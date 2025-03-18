#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

# 设置GPU环境
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 训练使用前6张卡
#export CUDA_VISIBLE_DEVICES_FOR_VLLM=6,7  # vLLM使用后2张卡
# export VLLM_TENSOR_PARALLEL_SIZE=2  # 必须与VLLM使用的GPU数量匹配

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=6 src/open_r1/grpo.py \
    --config ./grpo_config.yaml
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=6 src/open_r1/faster_grpo.py \
    --config ./config_fast_grpo.yaml

CUDA_VISIBLE_DEVICES=0,1,2,37 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=4 src/open_r1/f_grpo.py \
    --config ./config.yaml


docker run --runtime nvidia --shm-size=32g  --gpus '"device=6,7"' -p 5903:30000 \
    -v /home/ubuntu/33b_coder/models/merged_model:/models \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path /models --host 0.0.0.0 --port 30000 --tp 2 


export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
 python -m sglang.launch_server --model-path /home/ubuntu/33b_coder/models/merged_model --host 0.0.0.0 --port 30000 --tp 2 --enable-p2p-check 




python -m open_r1.run_pipeline \
  --model "Qwen/Qwen2-7B-Instruct" \
  --dataset "/path/to/your/dataset.jsonl" \
  --output-dir "./output/run1" \
  --num-train-epochs 2 \
  --batch-size 32 \
  --dataset-split "train" \
  --eval-split "test" \
  --include-eval \
  --eval-batch-size 16 \
  --num-generations 16 \
  --temperature 0.8 \
  --vllm-gpu-memory-utilization 0.9 \
  --reward-funcs "accuracy" "format" "reasoning_steps" "cosine" "repetition_penalty" "length" \
  --learning-rate 1e-5 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-steps 500 \
  --save-steps 200 \
  --beta 0.04 \
  --do-eval \
  --eval-strategy "steps" \
  --eval-steps 100 \
  --use-peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --seed 42 