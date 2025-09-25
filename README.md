# SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs

This repository contains the **core implementation** of the methods described in our paper, "SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs".

## Overview

Due to the limited storage space of anonymous GitHub uploads, **full datasets and some SAE-related resources** cannot be publicly provided at this time. However, all key training scripts have been included, and example training datasets are provided in our supplementary materials in OpenReview.

### Repository Content

- **train_open_source_v2.py**: Main script for supervised finetuning (SFT).
- **train_grpo_open_source_vllm.py**: Main script for GRPO-based training.
- **utils.py**: Utilities shared by training scripts.

> **Note:**  
> The directory `./data` is meant to store training datasets, and `./sae_acts` stores SAE-associated data used during training and analysis. These are **not included** in the repository due to anonymization requirements and size constraints. However, a key training dataset is made available with the supplementary materials of our paper, so you can refer to that for data formatting.

## Usage

1. **Key code is provided** for both the SFT and GRPO methods.
2. **Example data and formatting instructions** are in the supplementary materials.
3. To run the scripts, please prepare your data according to the sample files described.

## Code Overview

```
.
├── train_open_source_v2.py            # SFT training implementation
├── train_grpo_open_source_vllm.py     # GRPO training implementation
├── utils.py                           # Utility functions
├── data/                              # (Not included) training data
└── sae_acts/                          # (Not included) SAE-related data & activations
```

## Disclaimer

Due to anonymity and data privacy reasons, only key components and code files are published. Please refer to our supplementary materials in OpenReview for training datasets.
