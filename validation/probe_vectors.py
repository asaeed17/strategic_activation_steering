#!/usr/bin/env python3
"""
probe_vectors.py

Validates steering vectors by:
  1. Training Logistic Regression probes per layer on the contrastive pairs.
  2. Plotting accuracy curves to find where traits are actually encoded.
  3. Running control probes for Verbosity and Formality.
  4. Checking for bias: projecting control samples onto negotiation steering vectors.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_vectors import (
    MODELS, HF_TOKEN, format_sample, extract_hidden_states, ModelConfig
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def train_probe(
    pos_h: np.ndarray,   # (N, H)
    neg_h: np.ndarray,   # (N, H)
    cv: int = 5,
) -> float:
    """Trains a Logistic Regression probe and returns CV accuracy."""
    X = np.concatenate([pos_h, neg_h], axis=0)
    y = np.concatenate([np.ones(len(pos_h)), np.zeros(len(neg_h))], axis=0)
    
    # Simple Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    
    # Handle small datasets
    n_samples = len(X)
    actual_cv = min(cv, n_samples // 2)
    if actual_cv < 2:
        # Too few samples for CV, just train and test on the same (not ideal but better than nothing)
        clf.fit(X, y)
        return clf.score(X, y)
        
    scores = cross_val_score(clf, X, y, cv=actual_cv)
    return scores.mean()


def run_probing(
    model,
    tokenizer,
    config: ModelConfig,
    dimensions: List[Dict],
    batch_size: int = 4,
) -> Dict[str, np.ndarray]:
    """Runs probing for all dimensions and returns accuracies per layer."""
    results = {}
    
    for dim in tqdm(dimensions, desc="Probing dimensions"):
        dim_id = dim["id"]
        pairs = dim["pairs"]
        
        pos_texts = [format_sample(p["context"], p["positive"], tokenizer, config) for p in pairs]
        neg_texts = [format_sample(p["context"], p["negative"], tokenizer, config) for p in pairs]
        
        pos_h = extract_hidden_states(model, tokenizer, pos_texts, batch_size) # (N, L, H)
        neg_h = extract_hidden_states(model, tokenizer, neg_texts, batch_size) # (N, L, H)
        
        n_layers = pos_h.shape[1]
        accuracies = np.zeros(n_layers)
        
        for l in range(n_layers):
            accuracies[l] = train_probe(pos_h[:, l, :], neg_h[:, l, :])
            
        results[dim_id] = accuracies
        
    return results


# ---------------------------------------------------------------------------
# Bias Detection
# ---------------------------------------------------------------------------

def check_steering_bias(
    steering_vecs: np.ndarray,  # (L, H)
    control_pos_h: np.ndarray,  # (N, L, H)
    control_neg_h: np.ndarray,  # (N, L, H)
) -> np.ndarray:
    """
    Checks if a steering vector is biased toward a control trait.
    Returns the separation (mean pos projection - mean neg projection) at each layer.
    """
    n_layers = steering_vecs.shape[0]
    separations = np.zeros(n_layers)
    
    for l in range(n_layers):
        vec = steering_vecs[l]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        
        pos_projs = control_pos_h[:, l, :] @ vec
        neg_projs = control_neg_h[:, l, :] @ vec
        
        separations[l] = pos_projs.mean() - neg_projs.mean()
        
    return separations


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracies(results: Dict[str, np.ndarray], output_path: Path, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not found, skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    for dim_id, accs in results.items():
        plt.plot(accs, label=dim_id)
    
    plt.xlabel("Layer")
    plt.ylabel("Probe Accuracy")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.4, 1.05)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    log.info(f"Plot saved to {output_path}")


def print_summary(neg_results: Dict[str, np.ndarray], con_results: Dict[str, np.ndarray]):
    print("\n" + "=" * 60)
    print(f"{'Dimension':<25} | {'Best Layer':<10} | {'Max Accuracy':<12}")
    print("-" * 60)
    
    for results in [neg_results, con_results]:
        for dim_id, accs in results.items():
            best_l = np.argmax(accs)
            max_acc = accs[best_l]
            print(f"{dim_id:<25} | {best_l:<10} | {max_acc:<12.3f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS.keys()), default="qwen2.5-3b")
    p.add_argument("--negotiation_pairs", default="negotiation_steering_pairs.json")
    p.add_argument("--control_pairs", default="control_steering_pairs.json")
    p.add_argument("--output_dir", default="probe_results")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--dimensions", nargs="+", default=["firmness", "empathy"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODELS[args.model]
    output_dir = Path(args.output_dir) / cfg.alias
    output_dir.mkdir(parents=True, exist_ok=True)
    
    token = HF_TOKEN if cfg.requires_token else None
    
    # Load pairs
    with open(args.negotiation_pairs) as f:
        neg_data = json.load(f)
    with open(args.control_pairs) as f:
        con_data = json.load(f)
        
    all_neg_dims = neg_data["dimensions"]
    target_dims = [d for d in all_neg_dims if d["id"] in args.dimensions]
    control_dims = con_data["dimensions"]
    
    log.info(f"Loading model {cfg.hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, token=token, torch_dtype=cfg.torch_dtype, device_map="auto"
    )
    model.eval()
    
    # 1. Probe target negotiation dimensions
    log.info("Probing target negotiation dimensions...")
    neg_results = run_probing(model, tokenizer, cfg, target_dims, args.batch_size)
    plot_accuracies(neg_results, output_dir / "negotiation_probes.png", f"Negotiation Trait Probes ({cfg.alias})")
    
    # 2. Probe control dimensions
    log.info("Probing control dimensions...")
    con_results = run_probing(model, tokenizer, cfg, control_dims, args.batch_size)
    plot_accuracies(con_results, output_dir / "control_probes.png", f"Control Trait Probes ({cfg.alias})")
    
    # Print table
    print_summary(neg_results, con_results)
    
    # 3. Bias analysis
    log.info("Running bias analysis...")
    bias_results = {}
    
    # For each target dimension, check if it separates control dimensions
    for dim in target_dims:
        dim_id = dim["id"]
        # Load the steering vector (using mean_diff as baseline)
        vec_path = Path(f"vectors/{cfg.alias}/mean_diff/{dim_id}_all_layers.npy")
        if not vec_path.exists():
            log.warning(f"Steering vector not found for {dim_id}, skipping bias check.")
            continue
            
        steering_vecs = np.load(vec_path) # (L, H)
        
        for c_dim in control_dims:
            c_id = c_dim["id"]
            pairs = c_dim["pairs"]
            pos_texts = [format_sample(p["context"], p["positive"], tokenizer, cfg) for p in pairs]
            neg_texts = [format_sample(p["context"], p["negative"], tokenizer, cfg) for p in pairs]
            
            c_pos_h = extract_hidden_states(model, tokenizer, pos_texts, args.batch_size)
            c_neg_h = extract_hidden_states(model, tokenizer, neg_texts, args.batch_size)
            
            sep = check_steering_bias(steering_vecs, c_pos_h, c_neg_h)
            bias_results[f"{dim_id}_vs_{c_id}"] = sep
            
    # Plot bias results
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for key, sep in bias_results.items():
            plt.plot(sep, label=key)
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.xlabel("Layer")
        plt.ylabel("Projection Separation (Positive - Negative)")
        plt.title(f"Steering Vector Bias Analysis ({cfg.alias})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "bias_analysis.png")
        plt.close()
    except ImportError:
        pass
    
    # Save raw results
    raw_results = {
        "negotiation": {k: v.tolist() for k, v in neg_results.items()},
        "control": {k: v.tolist() for k, v in con_results.items()},
        "bias": {k: v.tolist() for k, v in bias_results.items()},
    }
    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(raw_results, f, indent=2)
        
    log.info(f"All results saved to {output_dir}")

if __name__ == "__main__":
    main()
