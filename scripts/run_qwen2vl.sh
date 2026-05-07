#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_TEMPLATE="${ROOT_DIR}/LLaMA-Factory-for-obliviate/examples/train_full/qwen2vl_72B.yaml"
TRAINING_JSON="${ROOT_DIR}/training_data/ul_2k_8k_qwen.json"

MODEL_NAME_OR_PATH="${1:-}"
COCO_TRAIN2017_DIR="${2:-}"
OUTPUT_DIR="${3:-${ROOT_DIR}/outputs/qwen2vl_72B_obliviate}"

if [[ -z "${MODEL_NAME_OR_PATH}" || -z "${COCO_TRAIN2017_DIR}" ]]; then
  echo "Usage: bash scripts/run_qwen2vl.sh /path/to/Qwen2-VL-72B-Instruct /path/to/coco/train2017 [output_dir]" >&2
  exit 1
fi

if [[ ! -d "${COCO_TRAIN2017_DIR}" ]]; then
  echo "COCO train2017 directory not found: ${COCO_TRAIN2017_DIR}" >&2
  exit 1
fi

if [[ ! -f "${TRAINING_JSON}" ]]; then
  echo "Training data not found: ${TRAINING_JSON}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_TEMPLATE}" ]]; then
  echo "Config template not found: ${CONFIG_TEMPLATE}" >&2
  exit 1
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli is not available in PATH." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/popev2-qwen2vl.XXXXXX")"
cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

export ROOT_DIR WORK_DIR MODEL_NAME_OR_PATH COCO_TRAIN2017_DIR OUTPUT_DIR CONFIG_TEMPLATE TRAINING_JSON

python - <<'PY'
import json
import os
from pathlib import Path

root_dir = Path(os.environ["ROOT_DIR"])
work_dir = Path(os.environ["WORK_DIR"])
training_json = Path(os.environ["TRAINING_JSON"])
config_template = Path(os.environ["CONFIG_TEMPLATE"])

try:
    data = json.loads(training_json.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"Failed to parse {training_json}: {exc}")

for item in data:
    image_path = item.get("images")
    if isinstance(image_path, str):
        item["images"] = Path(image_path).name

(work_dir / "ul_2k_8k_qwen.json").write_text(
    json.dumps(data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

dataset_info = {
    "ul_2k_8k_qwen": {
        "file_name": "ul_2k_8k_qwen.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
}
(work_dir / "dataset_info.json").write_text(
    json.dumps(dataset_info, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

config_text = config_template.read_text(encoding="utf-8")
replacements = {
    "__MODEL_NAME_OR_PATH__": os.environ["MODEL_NAME_OR_PATH"],
    "__DEEPSPEED_CONFIG__": str(root_dir / "LLaMA-Factory-for-obliviate" / "examples" / "deepspeed" / "ds_z3_config.json"),
    "__DATASET_DIR__": str(work_dir),
    "__IMAGE_DIR__": os.environ["COCO_TRAIN2017_DIR"],
    "__OUTPUT_DIR__": os.environ["OUTPUT_DIR"],
}
for source, target in replacements.items():
    config_text = config_text.replace(source, target)

(work_dir / "qwen2vl_72B.yaml").write_text(config_text, encoding="utf-8")
PY

cd "${ROOT_DIR}/LLaMA-Factory-for-obliviate"
llamafactory-cli train "${WORK_DIR}/qwen2vl_72B.yaml"
