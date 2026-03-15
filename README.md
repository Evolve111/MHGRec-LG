# MHGRec‑LG

## Installation
- Prerequisites: Python 3.9+
- Install PyTorch (follow the official instructions for your CUDA setup, or install the CPU build)

```bash
pip install torch torchvision torchaudio
```

- Install common dependencies

```bash
pip install -U numpy scipy scikit-learn networkx matplotlib requests
```

Optional: if you use a local/remote OpenAI-compatible LLM service, set environment variables:

```powershell
# Windows PowerShell example
$env:LLM_BASE_URL="http://localhost:11434/v1"   # default: local Ollama
$env:LLM_MODEL="deepseek-r1:latest"             # default
$env:LLM_API_KEY="<your_key_if_needed>"         # if a remote service requires it
```

## Data Preparation
- Example datasets are included at `data_recommendation/<DatasetName>/` (Bookcrossing / Movielens / Amazons)
- Optional multimodal directory `--mm_dir`, expected to contain:
  - `embed_text.npy` (text embeddings, shape: [num_items, d_text])
  - `embed_image.npy` (image embeddings, shape: [num_items, d_img])
  - Use `--mm_item_type` to specify the item node type id (default 1)

## Quick Start

```bash
python main_recommendation.py --dataset Bookcrossing
```

Enable multimodal fusion (average fusion example):

```bash
python main_recommendation.py --dataset Bookcrossing \
  --mm_dir path/to/bookcrossing-vit_bert \
  --mm_fusion avg --mm_norm
```

Full-item evaluation (instead of sampled candidates):

```bash
python main_recommendation.py --dataset Bookcrossing --full_item_eval
```

Ablation without LLM (random search):

```bash
python main_recommendation.py --dataset Bookcrossing --random_search
```

## Common Arguments
- Training
  - `--dataset`: Bookcrossing | Movielens | Amazons
  - `--gpu`: GPU index (default 0; falls back to CPU if CUDA is unavailable)
  - `--epochs` (default 200), `--patience` (default 30)
  - `--lr` (0.01), `--wd` (0.001), `--n_hid` (64), `--dropout` (0.6), `--seed` (1)
  - `--dataset_seed`: split seed (default 2)
- Search & LLM
  - `--num_generations` (20), `--population_size` (10)
  - `--delta`: path equivalence threshold (0.8)
  - `--w1 --w2 --w3`: weights for predicted performance / semantic prior / calibration
  - `--random_search`: disable LLM, use random candidates
  - `--no_grammar_translator`: feed raw graph structure without grammar translator
  - `--llm_temperature`, `--llm_top_p`, `--llm_max_tokens`, `--prompt_format {A|B|C|D}`, `--few_shot_k`
  - Cost tracking: `--llm_prompt_price_per_1k`, `--llm_completion_price_per_1k`
- Multimodal
  - `--mm_dir`; `--mm_item_type` (default 1)
  - `--mm_fusion`: sum | avg | weighted | concat | text | image; `--mm_alpha` for weighted fusion
  - `--mm_norm`: row-wise L2 normalization; `--disable_mm`: disable multimodal during search
  - `--user_mm_precomputed`: precomputed user profiles path; `--user_type` (default 0)
- Evaluation
  - `--full_item_eval`: rank over all items (default evaluates on sampled candidates)
- Aliases
  - `--num_iters` ≡ `--num_generations`; `--multimodal_dir` ≡ `--mm_dir`; `--fusion_method` ≡ `--mm_fusion`

More options and help:

```bash
python main_recommendation.py -h
```

## Reproducibility Examples
- Bookcrossing with default search:

```bash
python main_recommendation.py --dataset Bookcrossing --num_generations 20 --population_size 10 --seed 1
```

- Enable multimodal (average fusion):

```bash
python main_recommendation.py --dataset Bookcrossing \
  --mm_dir path/to/bookcrossing-vit_bert --mm_fusion avg --mm_norm
```

- Full-item evaluation:

```bash
python main_recommendation.py --dataset Bookcrossing --full_item_eval
```

- Ablation: random search (no LLM):

```bash
python main_recommendation.py --dataset Bookcrossing --random_search
```

- Movielens example:

```bash
python main_recommendation.py --dataset Movielens --seed 1
```

## Outputs & Logs
- Log directory: `log_recommendation/train_threshold_2_delta_{delta}_datasetseed_{dataset_seed}_changeinit/<Dataset>/pf_<...>/fs_<...>/seed_<...>/`
- Key artifacts:
  - `*.txt`: training & search logs
  - `performance_dict.pkl` / `gene_pools_history_dict.pkl` / `gene_pools_performance_dict.pkl`
  - `cost_stats.jsonl`: per-generation timing and LLM usage
  - `prompt_metrics.json`: prompt statistics (if available)
  - `seed_summary_*.json`: run summary

## Notes
- Use `--random_search` for LLM-free ablation experiments
- Specify `--gpu` for GPU acceleration; otherwise it runs on CPU
