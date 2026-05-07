from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Image, Value


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "POPEv2"
ANNOTATIONS_PATH = DATASET_DIR / "annotations.json"
PARQUET_PATH = DATASET_DIR / "test.parquet"
NORMAL_IMAGE_DIR = DATASET_DIR / "normal_images"
COUNTERFACTUAL_IMAGE_DIR = DATASET_DIR / "images"


def build_rows() -> list[dict[str, object]]:
    records = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    paired_examples: dict[int, dict[str, object]] = defaultdict(dict)

    for record in records:
        image_id = int(record["image_id"])
        example = paired_examples[image_id]
        example["example_id"] = f"popev2-{image_id:012d}"
        example["image_id"] = image_id
        example["target_object"] = record["target_object"]
        example["question"] = record["query"]

        file_name = f"{image_id:012d}.jpg"
        image_name = record["image_name"].lstrip("/")
        if image_name.startswith("images/"):
            counterfactual_path = COUNTERFACTUAL_IMAGE_DIR / file_name
            example["counterfactual_image"] = {
                "bytes": counterfactual_path.read_bytes(),
                "path": f"images/{file_name}",
            }
            example["counterfactual_label"] = record["label"]
        else:
            normal_path = NORMAL_IMAGE_DIR / file_name
            example["normal_image"] = {
                "bytes": normal_path.read_bytes(),
                "path": f"normal_images/{file_name}",
            }
            example["normal_label"] = record["label"]

    rows = []
    missing_fields = []
    required_fields = {
        "example_id",
        "image_id",
        "target_object",
        "question",
        "normal_image",
        "counterfactual_image",
        "normal_label",
        "counterfactual_label",
    }

    for image_id, example in sorted(paired_examples.items()):
        missing = sorted(required_fields - example.keys())
        if missing:
            missing_fields.append((image_id, missing))
            continue

        rows.append(example)

    if missing_fields:
        details = "; ".join(
            f"{image_id}: {', '.join(fields)}" for image_id, fields in missing_fields[:10]
        )
        raise ValueError(f"Incomplete paired examples found: {details}")

    return rows


def main() -> None:
    features = Features(
        {
            "example_id": Value("string"),
            "image_id": Value("int64"),
            "target_object": Value("string"),
            "question": Value("string"),
            "counterfactual_image": Image(),
            "counterfactual_label": Value("string"),
            "normal_image": Image(),
            "normal_label": Value("string"),
        }
    )
    dataset = Dataset.from_list(build_rows(), features=features)
    dataset.to_parquet(str(PARQUET_PATH))
    print(f"Wrote {dataset.num_rows} paired examples to {PARQUET_PATH}")


if __name__ == "__main__":
    main()
