# ORPO训练指南

本文档介绍如何使用ORPO (Odds Ratio Preference Optimization) 方法训练语言模型。该训练脚本支持LoRA参数高效微调、Flash Attention 2优化、断点续训练和保存LoRA适配器到本地。

## 特性

- **ORPO训练**: 使用无需参考模型的ORPO方法进行偏好对齐训练
- **LoRA训练**: 使用参数高效微调技术，减少显存占用
- **Flash Attention 2**: 使用优化的注意力实现，加速训练
- **断点续训**: 支持从最新检查点或指定检查点恢复训练
- **适配器保存**: 单独保存LoRA适配器，方便部署
- **自动划分评估集**: 支持从训练数据中自动划分一定比例作为评估集

## 环境准备

```bash
pip install transformers trl datasets peft accelerate torch flash-attn
```

## 数据格式

训练数据应为JSON格式，结构如下：

```json
[
  {
    "conversations": [
      {
        "from": "system",
        "value": "您是一名AI助手。用户会给您一个任务。您的目标是尽量忠实地完成任务。在执行任务时，要逐步思考并合理地解释您的步骤。"
      },
      {
        "from": "human",
        "value": "用户问题..."
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "更好的回答..."
    },
    "rejected": {
      "from": "gpt",
      "value": "较差的回答..."
    }
  },
  ...
]
```

## 使用方法

### 基本训练

使用accelerate启动分布式训练：

```bash
accelerate launch src/orpo/train.py \
  --model_name_or_path llama3-8b-hf \
  --train_file data.json \
  --eval_ratio 0.05 \
  --output_dir ./orpo_output \
  --beta 0.1 \
  --max_length 2048 \
  --max_prompt_length 1024 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
  --use_flash_attention True \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3
```

### 自定义评估集比例

可以通过`eval_ratio`参数控制从训练数据中划分多少比例作为评估集：

```bash
accelerate launch src/orpo/train.py \
  --model_name_or_path llama3-8b-hf \
  --train_file data.json \
  --eval_ratio 0.1 \
  --output_dir ./orpo_output \
  --beta 0.1 \
  --lora_r 16 \
  --use_flash_attention True
```

### 量化训练

使用4bit量化减少显存占用：

```bash
accelerate launch src/orpo/train.py \
  --model_name_or_path llama3-8b-hf \
  --train_file data.json \
  --eval_ratio 0.05 \
  --output_dir ./orpo_output \
  --use_4bit True \
  --beta 0.1 \
  --lora_r 16 \
  --use_flash_attention True \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5
```

### 断点续训

从最新检查点恢复训练：

```bash
accelerate launch src/orpo/train.py \
  --model_name_or_path llama3-8b-hf \
  --train_file data.json \
  --eval_ratio 0.05 \
  --output_dir ./orpo_output \
  --resume_from_checkpoint latest \
  --beta 0.1 \
  --lora_r 16 \
  --use_flash_attention True \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5
```

### 自定义LoRA适配器保存位置

指定LoRA适配器保存目录：

```bash
accelerate launch src/orpo/train.py \
  --model_name_or_path llama3-8b-hf \
  --train_file data.json \
  --eval_ratio 0.05 \
  --output_dir ./orpo_output \
  --lora_adapter_dir ./lora_adapters/my_model \
  --beta 0.1 \
  --lora_r 16 \
  --use_flash_attention True \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5
```

## 主要参数说明

### 模型和数据相关
- `--model_name_or_path`: 预训练模型名称或路径
- `--train_file`: 训练数据JSON文件路径
- `--eval_ratio`: 评估数据集占训练数据的比例，范围0-1，默认为0.05
- `--eval_file`: 评估数据JSON文件路径（可选，如果提供则使用此文件而不是从train_file划分）
- `--output_dir`: 保存模型的目录
- `--lora_adapter_dir`: 保存LoRA适配器的目录（可选，默认为output_dir/adapter）

### ORPO特定参数
- `--beta`: ORPO的beta参数（λ），控制偏好优化的强度，默认为0.1
- `--max_length`: 最大序列长度，默认为2048
- `--max_prompt_length`: 最大prompt长度，默认为1024

### LoRA相关参数
- `--lora_r`: LoRA的秩，默认为16
- `--lora_alpha`: LoRA的alpha参数，默认为32
- `--lora_dropout`: LoRA的dropout率，默认为0.05
- `--lora_target_modules`: LoRA目标模块，逗号分隔

### 量化和优化相关
- `--use_4bit`: 是否使用4bit量化，默认为False
- `--use_8bit`: 是否使用8bit量化，默认为False
- `--use_flash_attention`: 是否使用Flash Attention 2，默认为True

### 断点续训参数
- `--resume_from_checkpoint`: 从检查点恢复训练，'latest'表示最新检查点，或指定具体路径

### 训练相关（TransformerTrainer参数）
- `--num_train_epochs`: 训练轮数
- `--per_device_train_batch_size`: 每个设备的训练批量大小
- `--per_device_eval_batch_size`: 每个设备的评估批量大小
- `--gradient_accumulation_steps`: 梯度累积步数
- `--learning_rate`: 学习率
- `--warmup_ratio`: 预热比例

## 最佳实践

1. **Flash Attention 2**: 默认启用，可显著加速训练，建议开启
2. **显存优化**: 对于大模型，推荐使用4bit量化 + LoRA训练
3. **定期保存检查点**: 默认会保存检查点，可通过`--save_steps`控制保存频率
4. **选择合适的beta值**: beta值控制ORPO的优化强度，建议在0.05-0.2范围内调整
5. **自动评估集划分**: 默认会从训练集划分5%作为评估集，可通过`--eval_ratio`调整
6. **LoRA参数选择**: 
   - 对于小模型：r=8, alpha=16
   - 对于中型模型：r=16, alpha=32
   - 对于大模型：r=32, alpha=64

## 注意事项

1. 确保安装了Flash Attention 2：`pip install flash-attn --no-build-isolation`
2. 训练后LoRA适配器会同时保存到指定目录和主输出目录的adapter子目录中
3. 使用量化训练时，建议适当降低学习率
4. 断点续训功能会自动检测最新检查点，无需手动指定检查点编号
5. 评估集默认从训练集中随机划分，每次运行会得到不同的划分结果，可以通过设置种子确保结果一致 