import subprocess
import os

# 设置CUDA设备
cuda_devices = "5,6,7,8"
# 训练命令
command = [
    "/anaconda3/envs/llama_factory/bin/llamafactory-cli", 
    "train", 
    "/code/LLaMA-Factory/examples/train_full/qwen2vl_full_sft.yaml"
]

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = cuda_devices
env['PATH'] = '/anaconda3/envs/llama_factory/bin/' + env['PATH'] 

subprocess.run(command, env=env, check=True)
