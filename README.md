# KLASS: KL-Adaptive Stability Sampling for Fast Inference in Masked Diffusion Models (NeurIPS 2025 Spotlight)
[![arXiv](https://img.shields.io/badge/arXiv-2511.05664-red)](https://arxiv.org/abs/2511.05664)

🎉 **Accepted at NeurIPS 2025 (Spotlight)**  

**Authors:** Seo Hyun Kim*, Sunwoo Hong*, Hojung Jung, Youngrok Park, Se-Young Yun  
\*Equal contribution

**KLASS (KL-Adaptive Stability Sampling)** is a fast inference method designed to accelerate generation in masked diffusion models while maintaining high-quality outputs.

This repository provides an implementation of **KLASS** on **LLaDA 8B Instruct** and **Dream 7B Instruct**, along with evaluation scripts for standard benchmarks including **GSM8K**, **MATH**, **HumanEval**, and **MBPP**.

## 🚀 Installation

1. Create and activate the conda environment:
    
    ```coffeescript
    conda create -n klass python=3.12
    conda activate klass
    ```
    
2. Install dependencies and models:    
    ```coffeescript
    bash install.sh
    ```
    This script updates `generation_utils.py` in **Dream** with a customized version adapted for **KLASS**.

## 📊 Evaluation

We provide ready-to-run evaluation scripts for all supported models and datasets.

### LLaDA

```coffeescript
# GSM8K
bash scripts/llada_gsm8k.sh

# MATH
bash scripts/llada_math.sh

# Humaneval
bash scripts/llada_humaneval.sh

# MBPP
bash scripts/llada_mbpp.sh
```

### Dream

```coffeescript
# GSM8K
bash scripts/dream_gsm8k.sh

# MATH
bash scripts/dream_math.sh

# Humaneval
bash scripts/dream_humaneval.sh

# MBPP
bash scripts/dream_mbpp.sh
```

## ⚙️ Configuration & Arguments

You can customize the sampling behavior using the following arguments.

### Main Sampling Algorithm

- `alg`: Choose the unmasking algorithm.
  - `klass`: Uses KLASS sampling, which unmask tokens based on a combination of confidence and KL-divergence stability.
  - `default` (for LLaDA) / `maskgit_plus` (for Dream): Top-K confidence-based unmasking.
  - `random` (for LLaDA) / `origin` (for Dream): Random unmasking order.

### KLASS-Specific Parameters

These arguments are used only when `alg="klass"`.

- `conf_threshold`: Filter out tokens with confidence *lower than* this value.
- `kl_threshold`: Filter out tokens with a KL score *higher than* this value (calculated over `history_length`).
- `history_length`: Number of recent steps to use for the KL divergence stability calculation.
- `unmask_strategy`: Defines the strategy for unmasking the tokens that satisfy both the confidence and KL thresholds:
    - `all`: Unmask all tokens that satisfy the thresholds. (Default)
    - `max_conf`: Among the tokens satisfying the thresholds, unmask only the one with the maximum confidence.
    - `min_kl`: Among the tokens satisfying the thresholds, unmask only the one with the minimum KL score.

### Debugging

- `save_steps`: If set, this flag saves the detailed results of each generation step (including position, token ID, confidence, and KL divergence for all tokens) for analysis.


## 🙏 Acknowledgements

This codebase builds upon the official implementations of [**LLaDA**](https://github.com/ML-GSAI/LLaDA), [**Dream**](https://github.com/DreamLM/Dream), and [**HumanEval**](https://github.com/openai/human-eval).
We thank the original authors for their open-source contributions.
