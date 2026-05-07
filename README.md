# Analyzing and Mitigating Object Hallucination: A Training Bias Perspective

<div align="center">
  <a href="https://arxiv.org/abs/2508.04567"><img src="https://img.shields.io/badge/ArXiv-2508.04567-b31b1b.svg"></a>
  <a href="https://huggingface.co/datasets/Monosail/POPEv2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-yellow"></a>
</div>

Official code and benchmark release for our AAAI 2026 paper **Analyzing and Mitigating Object Hallucination: A Training Bias Perspective**.

## Overview

We study object hallucination in large vision-language models from a training-bias perspective and propose **Obliviate**, an unlearning-style mitigation method that suppresses hallucinated generations while preserving normal behavior.

This repository is organized for public release:

- training code lives in `LLaMA-Factory-for-obliviate/`
- the public **POPEv2** benchmark is released on Hugging Face at [`Monosail/POPEv2`](https://huggingface.co/datasets/Monosail/POPEv2)
- raw benchmark image assets are intentionally not versioned in this GitHub repo
- the released training data is stored in `training_data/`

## Repository Layout

| Path | Description |
| --- | --- |
| `LLaMA-Factory-for-obliviate/` | Modified LLaMA-Factory codebase used to implement Obliviate. |
| `LLaMA-Factory-for-obliviate/examples/train_full/qwen2vl_72B.yaml` | Example training config for the Qwen2-VL-72B setup used in our experiments. |
| `scripts/run_qwen2vl.sh` | One-command example launcher for Qwen2-VL training on the released training data. |
| `scripts/prepare_popev2_hf_parquet.py` | Utility to package paired images and annotations into the Hugging Face parquet release. |
| `POPEv2/README.md` | Benchmark note and loading instructions for the public release. |
| `training_data/ul_2k_8k_qwen.json` | Training-data release used by the Qwen2-VL Obliviate setup. |
| `training_data/dataset_info.json` | LLaMA-Factory dataset registry entry for the released training data. |
| `training_data/README.md` | Notes on the released training-data format and image-path assumptions. |
| `Obliviate_sup.pdf` | Supplementary material for the paper. |

## Benchmark

The **POPEv2** benchmark is publicly available on Hugging Face:

- **Dataset:** <https://huggingface.co/datasets/Monosail/POPEv2>
- **Format:** `test.parquet`
- **Per-example fields:** `normal_image`, `counterfactual_image`, `question`, `normal_label`, `counterfactual_label`, `image_id`, `example_id`, `target_object`

You can load it directly with:

```python
from datasets import load_dataset

dataset = load_dataset("Monosail/POPEv2", split="test")
```

## Project-specific Code Entry Points

Obliviate is implemented on top of LLaMA-Factory. The main project-specific entry points are:

| File | Role |
| --- | --- |
| `src/llamafactory/data/loader.py` | Adds the unlearning-aware dataset loading path via `get_ul_dataset`. |
| `src/llamafactory/data/processors/supervised.py` | Builds paired supervised / unlearning token sequences and token masks. |
| `src/llamafactory/data/preprocess.py` | Routes `lm_head_only` training to the unlearning preprocessing pipeline. |
| `src/llamafactory/train/sft/workflow.py` | Switches SFT training to the unlearning dataset path and passes `lm_head_only` to model loading. |
| `src/llamafactory/model/loader.py` | Freezes all parameters except `lm_head` when `lm_head_only=true`. |
| `src/llamafactory/hparams/training_args.py` | Exposes the unlearning strength coefficient `alpha`. |
| `src/llamafactory/hparams/finetuning_args.py` | Adds the `lm_head_only` training option. |

## Getting Started

1. Follow the environment setup in `LLaMA-Factory-for-obliviate/README.md`.
2. Make sure you have the released training file in `training_data/ul_2k_8k_qwen.json` and the corresponding COCO `train2017` images locally.
3. Launch the example with:

```bash
bash scripts/run_qwen2vl.sh /path/to/Qwen2-VL-72B-Instruct /path/to/coco/train2017 [output_dir]
```

The script rewrites the training-data image paths into a temporary portable dataset directory and then runs `llamafactory-cli train` with the bundled `qwen2vl_72B.yaml` config.

## License

The original code, scripts, and documentation in this repository are released under the **MIT License**. See [`LICENSE`](LICENSE).

Vendored third-party code keeps its original license:

- `LLaMA-Factory-for-obliviate/` retains its included Apache-2.0 license
- embedded third-party components inside that directory retain their own upstream licenses

## Citation

If you find this repository or the **POPEv2** benchmark useful in your research, please consider citing our paper:


```bibtex
@inproceedings{DBLP:conf/aaai/LiZZFW26,
  author       = {Yifan Li and
                  Kun Zhou and
                  Xin Zhao and
                  Lei Fang and
                  Jirong Wen},
  title        = {Analyzing and Mitigating Object Hallucination: {A} Training Bias Perspective},
  booktitle    = {{AAAI}},
  pages        = {6636--6643},
  publisher    = {{AAAI} Press},
  year         = {2026}
}
```
