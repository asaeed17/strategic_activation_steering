#!/usr/bin/env python3
"""
extract_vectors.py

Pulls behavioural steering vectors out of open-source LLMs for each
negotiation dimension defined in negotiation_steering_pairs.json.

Two methods are implemented — I went with both because they have different
failure modes and it's useful to compare them:

  1. Mean Difference (MD)
       direction_l = mean(pos_hiddens_l) - mean(neg_hiddens_l)
       This is the standard CAA approach from Panickssery et al. 2024.
       Simple and usually works well.

  2. PCA on Difference Vectors
       differences = [pos_i - neg_i  for each pair i]
       direction_l = PCA(differences_l).components_[0]
       Sign is resolved by aligning with the mean difference.
       Follows Zou et al. 2023 (RepE). More robust when individual pairs
       are noisy, since it finds the dominant axis of variation.

Activations are taken from the last real token of each formatted input.
Batches are left-padded so index [-1] always lands on the right token.

Example usage:
  # Qwen only — no token needed
  python extract_vectors.py --models qwen2.5-3b

  # Multiple models, all dimensions
  python extract_vectors.py --models qwen2.5-7b llama3-8b

  # Just a couple of dimensions, 4-bit quant to save VRAM
  python extract_vectors.py --models llama3-8b --dimensions firmness empathy --quantize

  # Only save specific layers (e.g. every 4th)
  python extract_vectors.py --models qwen2.5-7b --layers 8 12 16 20 24

Output layout:
  vectors/
    {model_alias}/
      mean_diff/
        {dimension_id}_all_layers.npy   # shape: (n_layers, hidden_dim)
        {dimension_id}_layer08.npy      # shape: (hidden_dim,)
        ...
      pca/
        {dimension_id}_all_layers.npy
        {dimension_id}_layer08.npy
        ...
      metadata.json
"""

import os
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace token — see the docstring above for setup options
# ---------------------------------------------------------------------------
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN", None)
# Uncomment and paste your token here if you prefer:
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    hf_id: str               # HuggingFace repo ID
    alias: str               # short name used as the output sub-directory
    requires_token: bool     # True for gated models (Llama, Gemma, Mistral v0.3)
    use_chat_template: bool  # wrap texts in the model's chat template
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16,
                "float16":  torch.float16,
                "float32":  torch.float32}[self.dtype]


MODELS: Dict[str, ModelConfig] = {
    # Llama 3 ------------------------------------------------- needs token --
    "llama3-8b": ModelConfig(
        hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        alias="llama3-8b",
        requires_token=True,
        use_chat_template=True,
    ),
    "llama3-3b": ModelConfig(
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        alias="llama3-3b",
        requires_token=True,
        use_chat_template=True,
    ),
    # Qwen 2.5 ----------------------------------------- no token required --
    "qwen2.5-7b": ModelConfig(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        alias="qwen2.5-7b",
        requires_token=False,
        use_chat_template=True,
    ),
    "qwen2.5-3b": ModelConfig(
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        alias="qwen2.5-3b",
        requires_token=False,
        use_chat_template=True,
    ),
    "qwen2.5-1.5b": ModelConfig(
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        alias="qwen2.5-1.5b",
        requires_token=False,
        use_chat_template=True,
    ),
    # Gemma 2 ------------------------------------------------ needs token --
    "gemma2-9b": ModelConfig(
        hf_id="google/gemma-2-9b-it",
        alias="gemma2-9b",
        requires_token=True,
        use_chat_template=True,
    ),
    "gemma2-2b": ModelConfig(
        hf_id="google/gemma-2-2b-it",
        alias="gemma2-2b",
        requires_token=True,
        use_chat_template=True,
    ),
    # Mistral ------------------------------------------------ needs token --
    "mistral-7b": ModelConfig(
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        alias="mistral-7b",
        requires_token=True,
        use_chat_template=True,
    ),
}


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def format_sample(
    context: str,
    response: str,
    tokenizer,
    config: ModelConfig,
) -> str:
    """
    Format a (context, response) pair for the model.

    For instruct models we use the model's chat template: context goes in
    the user turn, response in the assistant turn. This means the last
    token in the sequence is the last token of the assistant's response,
    which is where we want to read out the model's representation.

    Falls back to plain-text concatenation if the template errors out.
    """
    if config.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user",      "content": context},
            {"role": "assistant", "content": response},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            log.warning("Chat template failed (%s) — falling back to plain text.", exc)

    return f"{context}\n\nResponse: {response}"


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,
) -> np.ndarray:
    """
    Forward-pass over `texts` and return the last-token hidden state from
    every transformer layer.

    Left-padding means index [-1] always points to the final real token,
    regardless of sequence length — that's why we set padding_side='left'
    when loading the tokenizer.

    Returns
    -------
    np.ndarray  shape (n_texts, n_layers, hidden_dim), dtype float32, on CPU
    """
    device = next(model.parameters()).device
    all_hiddens: List[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size),
                      desc="  batches", leave=False):
        batch = texts[start : start + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # out.hidden_states is a tuple of length (n_layers + 1):
        #   [0] = embedding output, [1:] = transformer block outputs
        block_hiddens = out.hidden_states[1:]   # each is (B, T, H)

        # grab last token from each layer, then rearrange to (B, n_layers, H)
        last_tok = np.stack(
            [h[:, -1, :].cpu().float().numpy() for h in block_hiddens],
            axis=0,
        ).transpose(1, 0, 2)

        all_hiddens.append(last_tok)

    return np.concatenate(all_hiddens, axis=0)  # (N, n_layers, H)


# ---------------------------------------------------------------------------
# Direction vector computation
# ---------------------------------------------------------------------------

def compute_mean_diff(
    pos: np.ndarray,   # (N, n_layers, H)
    neg: np.ndarray,   # (N, n_layers, H)
) -> np.ndarray:       # (n_layers, H), unit-normed
    """
    Mean Difference steering vector.

        direction_l = normalise( mean(pos_l) - mean(neg_l) )

    The simplest approach and usually a solid baseline.
    """
    direction = pos.mean(axis=0) - neg.mean(axis=0)   # (n_layers, H)
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return direction / norms


def compute_pca_direction(
    pos: np.ndarray,   # (N, n_layers, H)
    neg: np.ndarray,   # (N, n_layers, H)
) -> np.ndarray:       # (n_layers, H), unit-normed
    """
    PCA on difference vectors, per Zou et al. 2023 (RepE).

        differences_l = [pos_i_l - neg_i_l  for each pair i]
        direction_l   = PCA(differences_l).components_[0]

    Sign ambiguity is resolved by aligning PC1 with the mean difference.

    The intuition: rather than just averaging, PCA finds the single axis
    that explains most of the variance across contrastive pairs — which
    tends to be cleaner when individual pairs are noisy.
    """
    diffs     = pos - neg                  # (N, n_layers, H)
    mean_diff = diffs.mean(axis=0)         # (n_layers, H), used for sign alignment

    n_layers   = diffs.shape[1]
    H          = diffs.shape[2]
    directions = np.zeros((n_layers, H), dtype=np.float32)

    for l in range(n_layers):
        d_l = diffs[:, l, :]               # (N, H)

        if d_l.shape[0] < 2:
            # degenerate case — only one pair, just use it directly
            directions[l] = d_l[0] / max(np.linalg.norm(d_l[0]), 1e-8)
            continue

        pca = PCA(n_components=1, svd_solver="full")
        pca.fit(d_l)
        pc1 = pca.components_[0]           # (H,)

        # flip sign if needed so it points the same way as the mean diff
        if np.dot(pc1, mean_diff[l]) < 0:
            pc1 = -pc1

        norm = np.linalg.norm(pc1)
        directions[l] = pc1 / max(norm, 1e-8)

    return directions


# ---------------------------------------------------------------------------
# Per-model extraction pipeline
# ---------------------------------------------------------------------------

def extract_for_model(
    config: ModelConfig,
    dimensions: List[Dict],
    output_dir: Path,
    batch_size: int = 4,
    use_quantization: bool = False,
    target_layers: Optional[List[int]] = None,
) -> None:
    """Load one model, extract vectors for all requested dimensions, save to disk."""

    # check token before we bother downloading anything
    token = HF_TOKEN if config.requires_token else None
    if config.requires_token and token is None:
        log.error(
            "Model '%s' is gated and needs HF_TOKEN. "
            "See the top of this script for setup instructions.",
            config.alias,
        )
        return

    log.info("Loading tokenizer  →  %s", config.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        token=token,
        padding_side="left",   # critical: ensures last token is always at index -1
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model  →  %s", config.hf_id)
    if use_quantization:
        try:
            from bitsandbytes import BitsAndBytesConfig as _  # noqa: F401
        except ImportError:
            log.error(
                "bitsandbytes not installed. "
                "Install with: pip install bitsandbytes"
            )
            return
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            token=token,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            token=token,
            dtype=config.torch_dtype,
            device_map="auto",
        )
    model.eval()

    # set up output directories
    model_dir = output_dir / config.alias
    (model_dir / "mean_diff").mkdir(parents=True, exist_ok=True)
    (model_dir / "pca").mkdir(parents=True, exist_ok=True)

    saved_n_layers:   Optional[int] = None
    saved_hidden_dim: Optional[int] = None

    for dim in tqdm(dimensions, desc=f"[{config.alias}] dimensions"):
        dim_id   = dim["id"]
        dim_name = dim["name"]
        pairs    = dim["pairs"]

        log.info("  ├─ %s  (%d pairs)", dim_name, len(pairs))

        pos_texts = [
            format_sample(p["context"], p["positive"], tokenizer, config)
            for p in pairs
        ]
        neg_texts = [
            format_sample(p["context"], p["negative"], tokenizer, config)
            for p in pairs
        ]

        log.info("  │    extracting positive activations …")
        pos_h = extract_hidden_states(model, tokenizer, pos_texts, batch_size)

        log.info("  │    extracting negative activations …")
        neg_h = extract_hidden_states(model, tokenizer, neg_texts, batch_size)

        # pos_h, neg_h: (N, n_layers, hidden_dim)
        n_layers   = pos_h.shape[1]
        hidden_dim = pos_h.shape[2]
        saved_n_layers   = n_layers
        saved_hidden_dim = hidden_dim

        # figure out which layers to save as individual files
        if target_layers is not None:
            save_layers = sorted(set(l for l in target_layers if 0 <= l < n_layers))
        else:
            save_layers = list(range(n_layers))

        # method 1: mean difference
        md_vecs = compute_mean_diff(pos_h, neg_h)              # (n_layers, H)
        np.save(model_dir / "mean_diff" / f"{dim_id}_all_layers.npy", md_vecs)
        for l in save_layers:
            np.save(
                model_dir / "mean_diff" / f"{dim_id}_layer{l:02d}.npy",
                md_vecs[l],
            )

        # method 2: PCA on differences
        pca_vecs = compute_pca_direction(pos_h, neg_h)         # (n_layers, H)
        np.save(model_dir / "pca" / f"{dim_id}_all_layers.npy", pca_vecs)
        for l in save_layers:
            np.save(
                model_dir / "pca" / f"{dim_id}_layer{l:02d}.npy",
                pca_vecs[l],
            )

        log.info("  │    saved  mean_diff + pca  →  %s/", model_dir.name)

    # write metadata so we know later what we extracted and how
    metadata = {
        "model_id":   config.hf_id,
        "alias":      config.alias,
        "n_layers":   saved_n_layers,
        "hidden_dim": saved_hidden_dim,
        "dimensions": [d["id"] for d in dimensions],
        "methods":    ["mean_diff", "pca"],
        "saved_layers": target_layers if target_layers else list(range(saved_n_layers or 0)),
        "notes": {
            "mean_diff":  "direction_l = normalise(mean(pos_l) - mean(neg_l))",
            "pca":        "direction_l = PCA(pos_i_l - neg_i_l).components_[0], sign-aligned",
            "last_token": "activations taken at the last real token (left-padded)",
        },
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("  └─ metadata saved  →  %s/metadata.json", model_dir.name)

    # free VRAM before the next model
    del model
    torch.cuda.empty_cache()
    log.info("Model unloaded.\n")


# ---------------------------------------------------------------------------
# Quick diagnostic: cosine similarity matrix across dimensions
# ---------------------------------------------------------------------------

def print_similarity_matrix(model_dir: Path, method: str, layer: int) -> None:
    """
    Print cosine similarities between all dimension vectors at a given layer.

    High similarity between two dimensions is a sign they might be partially
    redundant in this particular model — worth knowing before you use them
    together in a steering experiment.
    """
    vec_dir = model_dir / method
    files   = sorted(vec_dir.glob(f"*_layer{layer:02d}.npy"))
    if not files:
        return

    names  = [f.stem.replace(f"_layer{layer:02d}", "") for f in files]
    vecs   = np.stack([np.load(f) for f in files])              # (D, H)
    norms  = np.linalg.norm(vecs, axis=-1, keepdims=True)
    vecs_n = vecs / np.where(norms == 0, 1.0, norms)
    sims   = vecs_n @ vecs_n.T                                   # (D, D)

    col_w  = max(len(n) for n in names) + 2
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
    print(f"\nCosine similarity  [{method}  layer {layer}]\n{header}")
    for i, name in enumerate(names):
        row = f"{name:<{col_w}}" + "".join(
            f"{sims[i, j]:>{col_w}.2f}" for j in range(len(names))
        )
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract negotiation steering vectors from open-source LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=["qwen2.5-3b"],
        metavar="MODEL",
        help=(
            "Which models to process. "
            f"Choices: {', '.join(MODELS)}. "
            "Qwen models need no HF token."
        ),
    )
    p.add_argument(
        "--pairs_file",
        default="negotiation_steering_pairs.json",
        help="Path to the contrastive pairs JSON (default: negotiation_steering_pairs.json)",
    )
    p.add_argument(
        "--pairs_dir",
        default=None,
        help=(
            "Directory containing subdirectories, each with a negotiation_steering_pairs.json. "
            "Overrides --pairs_file. Vectors are saved to output_dir/{variant_name}/. "
            "E.g.: --pairs_dir steering_pairs"
        ),
    )
    p.add_argument(
        "--output_dir",
        default="vectors",
        help="Root directory for saved vectors (default: vectors/)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size. Reduce if you get OOM (default: 4)",
    )
    p.add_argument(
        "--dimensions",
        nargs="+",
        default=None,
        metavar="DIM",
        help=(
            "Only process these dimension IDs. Default: all. "
            "E.g.: --dimensions firmness empathy assertiveness"
        ),
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Save individual .npy files only for these layer indices. "
            "The *_all_layers.npy file is always written regardless. "
            "E.g.: --layers 8 16 24"
        ),
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help=(
            "Load in 4-bit NF4 (needs bitsandbytes). "
            "Cuts VRAM by ~4× at a small quality cost."
        ),
    )
    p.add_argument(
        "--sim_matrix",
        action="store_true",
        help="After extraction, print the cosine similarity matrix across dimensions.",
    )
    p.add_argument(
        "--sim_layer",
        type=int,
        default=16,
        help="Layer to use when printing the similarity matrix (default: 16)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build list of (pairs_path, output_dir) jobs
    jobs = []
    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir)
        if not pairs_dir.is_dir():
            log.error("--pairs_dir is not a directory: %s", pairs_dir)
            return
        for variant_dir in sorted(pairs_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            pairs_file = variant_dir / "negotiation_steering_pairs.json"
            if not pairs_file.exists():
                log.warning("No negotiation_steering_pairs.json in %s — skipping", variant_dir.name)
                continue
            out_dir = Path(args.output_dir) / variant_dir.name
            jobs.append((pairs_file, out_dir, variant_dir.name))
        if not jobs:
            log.error("No valid variants found in %s", pairs_dir)
            return
        log.info("Found %d variants: %s", len(jobs), [j[2] for j in jobs])
    else:
        pairs_path = Path(args.pairs_file)
        if not pairs_path.exists():
            log.error("Pairs file not found: %s", pairs_path)
            return
        jobs.append((pairs_path, Path(args.output_dir), None))

    for pairs_path, output_dir, variant_name in jobs:
        if variant_name:
            log.info("=" * 60)
            log.info("VARIANT: %s", variant_name)
            log.info("=" * 60)

        with open(pairs_path) as f:
            data = json.load(f)

        dimensions = data["dimensions"]

        if args.dimensions:
            requested  = set(args.dimensions)
            dimensions = [d for d in dimensions if d["id"] in requested]
            missing    = requested - {d["id"] for d in dimensions}
            if missing:
                log.warning("Unknown dimension IDs (skipping): %s", missing)

        log.info(
            "Loaded %d dimension(s): %s",
            len(dimensions),
            [d["id"] for d in dimensions],
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        for model_key in args.models:
            cfg = MODELS[model_key]
            log.info("=" * 60)
            log.info("MODEL:  %s", cfg.hf_id)
            log.info("ALIAS:  %s", cfg.alias)
            log.info("=" * 60)

            extract_for_model(
                config=cfg,
                dimensions=dimensions,
                output_dir=output_dir,
                batch_size=args.batch_size,
                use_quantization=args.quantize,
                target_layers=args.layers,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        cfg = MODELS[model_key]
        log.info("=" * 60)
        log.info("MODEL:  %s", cfg.hf_id)
        log.info("ALIAS:  %s", cfg.alias)
        log.info("=" * 60)

        extract_for_model(
            config=cfg,
            dimensions=dimensions,
            output_dir=output_dir,
            batch_size=args.batch_size,
            use_quantization=args.quantize,
            target_layers=args.layers,
        )
        if args.sim_matrix:
            model_dir = output_dir / cfg.alias
            for method in ("mean_diff", "pca"):
                print_similarity_matrix(model_dir, method, args.sim_layer)

        log.info("Vectors saved under: %s/", output_dir)

    log.info("All done.")


if __name__ == "__main__":
    main()
