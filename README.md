# SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2507.14894-b31b1b.svg)](https://arxiv.org/abs/2507.14894)
[![Conference](https://img.shields.io/badge/ICLR-2026-4b44ce.svg)](https://iclr.cc/Conferences/2026)

This repository contains the official implementation of "SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs", accepted at ICLR 2026.

## Installation

```bash
git clone https://github.com/Aatrox103/SASFT
cd SASFT
pip install -r requirements.txt
```

### Additional Setup for Llama-3.1-8B

If you plan to work with Llama-3.1-8B, you'll need to install the SAE library following the guidelines from [OpenMOSS/Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs)

## Downloading Pre-trained SAEs

To download the pre-trained Sparse Autoencoders, simply run:

```bash
python download.py
```

This will automatically download the following SAE models into the `./SAE/` directory:
- `Llama3_1-8B-Base-LXR-8x`
- `gemma-scope-2b-pt-res`
- `gemma-scope-9b-pt-res`

> **Note:** If you encounter network issues, you can uncomment the alternative download URLs in `download.py` to use mirror sites.

## Data

All data files are located in the `./data/` directory:

| File | Description |
|---|---|
| `sft_data_zh_110k.jsonl` | ~110k Chinese-target SFT training data — used for main paper results |
| `sft_data_ko_110k.jsonl` | ~110k Korean-target SFT training data — used for main paper results |
| `sft_data_ru_110k.jsonl` | ~110k Russian-target SFT training data — used for main paper results |
| `multilingual_data.jsonl` | ~1k multilingual samples used to identify language-specific SAE features |
| `multilingual_data_new_SFTData.jsonl` | ~7.8k multilingual samples used to estimate per-language average feature activations (loaded by `find_lan_feature.py`) |
| `cs_test/` | Test set for code-switching detection and evaluation (`cs_detect.py`) |

## Usage

### 1. Finding Language-Specific Features

To identify language-specific features in LLMs (e.g., for Gemma-2-2B):

```bash
python find_lan_feature.py --model gemma-2-2b --model_path YOUR_MODEL_PATH
```

Results will be saved in `./sae_acts/{model}/layer_{layer}/`:
```
sae_acts/
└── gemma-2-2b/
    └── layer_20/
        ├── sae_acts_without_relu_per_lan_avg_new_SFTData.pth  # Required for SASFT training
        ├── top_index_per_lan_magnitude.pth
        └── ...
```

- `top_index_per_lan_magnitude.pth` has shape `11 × feature_num`, where the 11 languages are: `['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar', 'ru']`. It stores the indices of language-specific SAE features ranked by the metric proposed in our paper, used to select which features to penalize during SASFT training.
- `sae_acts_without_relu_per_lan_avg_new_SFTData.pth` stores the average pre-activation values per language per feature, loaded by `train_open_source_v2.py` to compute the SAE-guided loss.

### 2. Training

```bash
accelerate launch train.py \
  --model gemma-2-2b \
  --model_path google/gemma-2-2b \
  --data_set zh_200k \
  --lr 5e-5 \
  --whether_sae True \
  --sae_method SASFT \
  --reduced_lan zh \
  --layer 24 25 \
  --loss_weight 0.0005
```

Checkpoints are saved to `./checkpoints/{data_set}/{model}/{method}/`. For more examples (baseline SFT, larger models with DeepSpeed, etc.), see `train.sh`.

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `gemma-2-2b` | Model name (`gemma-2-2b`, `gemma-2-9b`, `Meta-Llama-3.1-8B`) |
| `--model_path` | — | Path or HuggingFace ID of base model |
| `--data_set` | `zh_110k` | Training dataset name |
| `--whether_sae` | `False` | Enable SAE-guided loss |
| `--sae_method` | `SASFT` | SAE loss method (`SASFT` or `SASFT_zero`) |
| `--reduced_lan` | `ko` | Target language to reduce code-switching for |
| `--layer` | `[20]` | Layer indices to apply SAE loss |
| `--loss_weight` | `0.001` | Weight of SAE loss term |

### 3. Evaluation

To evaluate code-switching rate on a trained checkpoint:

```bash
python cs_detect.py --model_path ./checkpoints/zh_110k/gemma-2-2b/SASFT/SAE-True_lr_5e-05_epoch_1-layer-num-2-loss-weight-0.0005-layer-24-25_top_feature_idx_0-1/checkpoint-439
```

Results are saved to `./cs_results/` including:
- `cs_gen_results.jsonl`: Generated outputs with script detection
- `cs_results_all.json`: Aggregated code-switching statistics by language pair
- `samples/`: Example code-switching instances (e.g., `zh_to_ko.jsonl`)

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{dengsasft,
  title={SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs},
  author={Deng, Boyi and Wan, Yu and Yang, Baosong and Huang, Fei and Wang, Wenjie and Feng, Fuli},
  year={2026}
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```

## Contact

For questions or feedback, feel free to reach out to Boyi Deng at dengboyi@mail.ustc.edu.cn.
