# Training Data

This directory stores the released training data used for **Obliviate**.

The main file used in our Qwen2-VL setup is:

- `ul_2k_8k_qwen.json`

It follows a paired supervision / unlearning format:

- `images`: image path
- `messages`: standard instruction-following conversation
- `unlearning_messages`: alternative hallucinated conversation used for unlearning
- `unlearning_messages[*].mask`: token-level binary mask indicating which generated tokens are treated as hallucinated

Additional files:

- `dataset_info.json`: dataset registry entry for LLaMA-Factory

Notes:

- The released JSON currently uses COCO-style image paths such as `/coco/train2017/000000539056.jpg`.
- `scripts/run_qwen2vl.sh` creates a temporary portable copy of the dataset so you can point training to any local `train2017` directory.
