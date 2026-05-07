# POPEv2

The canonical public release of **POPEv2** is hosted on Hugging Face:

- **Dataset:** <https://huggingface.co/datasets/Monosail/POPEv2>
- **Paper:** <https://arxiv.org/abs/2508.04567>

This GitHub repository intentionally does **not** version the raw benchmark assets in order to keep the code release lightweight. The following local files are expected to stay untracked in the GitHub release:

- `annotations.json`
- `test.parquet`
- `images/`
- `normal_images/`

The Hugging Face release packages **500** paired examples. Each sample contains:

| Column | Description |
| --- | --- |
| `example_id` | Stable POPEv2 example identifier |
| `image_id` | COCO image id shared by the pair |
| `normal_image` | Original image |
| `counterfactual_image` | Counterfactual image built from the original image |
| `target_object` | Queried object category |
| `question` | Binary object-presence question |
| `normal_label` | Ground-truth answer for the original image |
| `counterfactual_label` | Ground-truth answer for the counterfactual image |

## Loading the dataset

Load the benchmark directly from the Hub with:

```python
from datasets import load_dataset

dataset = load_dataset("Monosail/POPEv2", split="test")
```

## Notes

- The published benchmark lives on Hugging Face rather than in this GitHub repo.

## Citation

If you find this benchmark useful in your research, please consider citing our paper:

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
