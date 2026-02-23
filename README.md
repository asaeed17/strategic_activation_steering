# comp0087_snlp_cwk

## Overview

This project extracts behavioural steering vectors from open-source LLMs and uses them to evaluate whether those vectors make a model a better price negotiator, using the CraigslistBargains dataset as the arena.

There are two main scripts:

---

### 1. `extract_vectors.py` — extract steering vectors

Runs contrastive pairs from `negotiation_steering_pairs.json` through a model and saves a direction vector per layer per dimension. Two methods are computed for each: mean difference (simple, usually solid) and PCA on difference vectors (more robust when pairs are noisy).

```bash
# quickstart
python extract_vectors.py --models qwen2.5-3b

# multiple models, specific dimensions, 4-bit quantised to save VRAM
python extract_vectors.py --models qwen2.5-7b llama3-8b --dimensions firmness empathy --quantize

# only save specific layers
python extract_vectors.py --models qwen2.5-7b --layers 8 12 16 20
```
Vectors are saved under `vectors/{model_alias}/mean_diff/` and `vectors/{model_alias}/pca/` as `.npy` files. Run `extract_vectors.py` before `apply_steering.py`.


---

### 2. `apply_steering.py` — run the negotiation evaluation

Pits a steered agent against a baseline agent across a sample of Craigslist scenarios. The steered agent alternates between seller and buyer roles each game to avoid role bias. Scores measure how close the agreed price was to each agent's private target.

```bash
# basic run — 50 games, firmness dimension, layers 12/16/20, alpha=20
python apply_steering.py \
    --model qwen2.5-7b \
    --dimension firmness \
    --alpha 20 \
    --layers 12 16 20 \
    --use_craigslist \
    --num_samples 50 \
    --output_file results.json
```

Results are written to `results.json` with a summary and full per-game transcripts. The key metric is `advantage`: mean steered score minus mean baseline score across games that reached a deal. Positive means the steered agent won more of the price range on average.