#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from trl import ORPOTrainer
from trl.trainer.orpo_config import ORPOConfig

from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers.utils import is_liger_kernel_available

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """
    ORPO训练的命令行参数
    """
    model_name_or_path: str = field(
        metadata={"help": "预训练模型名称或路径"}
    )
    train_file: str = field(
        metadata={"help": "训练数据JSON文件路径"}
    )
    eval_ratio: float = field(
        default=0.05,
        metadata={"help": "评估数据集占训练数据的比例，范围0-1"}
    )
    eval_file: str = field(
        default=None,
        metadata={"help": "评估数据JSON文件路径（可选，如果提供则使用此文件而不是从train_file划分）"}
    )
    lora_adapter_dir: str = field(
        default=None,
        metadata={"help": "保存LoRA adapter的目录，若不指定则使用output_dir/adapter"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "ORPO的beta参数（λ）"}
    )
    # LoRA参数
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA的秩"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA的alpha参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA的dropout率"}
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA目标模块，逗号分隔"}
    )
    # 量化参数
    use_4bit: bool = field(
        default=False,
        metadata={"help": "是否使用4bit量化"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "是否使用8bit量化"}
    )
    # Flash Attention相关参数
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "是否使用Flash Attention 2"}
    )

def load_dataset_from_json(file_path: str) -> Dataset:
    """
    从JSON文件加载数据集并转换为适合ORPO训练的格式
    
    参数:
        file_path: JSON文件路径
        
    返回:
        Dataset对象，包含标准格式的数据
    """
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据为ORPO格式
    processed_data = []
    for item in data:
        # 标准化聊天格式，把from改为role，把value改为content
        messages = []
        for conv in item["conversations"]:
            messages.append({
                "role": "system" if conv["from"] == "system" else (
                    "user" if conv["from"] == "human" else "assistant"
                ),
                "content": conv["value"]
            })
        
        # 获取chosen和rejected回答
        chosen_content = item["chosen"]["value"]
        rejected_content = item["rejected"]["value"]
        
        # 构建完整对话序列 - 包括chosen和rejected回答
        chosen_messages = messages.copy()
        chosen_messages.append({"role": "assistant", "content": chosen_content})
        
        rejected_messages = messages.copy()
        rejected_messages.append({"role": "assistant", "content": rejected_content})
        
        # TRL库期望的格式是包含chosen和rejected字段的字典，其中每个字段是完整的对话消息列表
        processed_data.append({
            "chosen": chosen_messages,
            "rejected": rejected_messages,
        })
    
    # 创建Dataset对象
    return Dataset.from_list(processed_data)


def split_dataset(dataset: Dataset, eval_ratio: float) -> Tuple[Dataset, Dataset]:
    """
    将数据集分割为训练集和评估集
    
    参数:
        dataset: 原始数据集
        eval_ratio: 评估集占比，0-1之间
        
    返回:
        训练集和评估集的元组 (train_dataset, eval_dataset)
    """
    if eval_ratio <= 0 or eval_ratio >= 1:
        logger.warning(f"无效的eval_ratio值: {eval_ratio}，应在0-1之间。使用默认值0.05")
        eval_ratio = 0.05
    
    # 计算训练集和评估集的大小
    dataset_size = len(dataset)
    eval_size = int(dataset_size * eval_ratio)
    train_size = dataset_size - eval_size
    
    # 随机打乱并分割数据集
    shuffled_dataset = dataset.shuffle(seed=42)
    train_dataset = shuffled_dataset.select(range(train_size))
    eval_dataset = shuffled_dataset.select(range(train_size, dataset_size))
    
    logger.info(f"数据集分割完成: 训练集大小 {len(train_dataset)}，评估集大小 {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """查找最新的检查点"""
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoint_dirs:
        return None
    
    # 按检查点编号排序
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
    
    logger.info(f"找到最新检查点: {latest_checkpoint}")
    return latest_checkpoint


def apply_liger_kernel(model, script_args):
    """应用Liger Kernel优化到模型"""
    if not script_args.use_liger_kernel:
        return model
        
    if not is_liger_kernel_available():
        raise ImportError(
            "您设置了 `use_liger_kernel=True` 但 liger-kernel >= 0.3.0 未安装。"
            "请使用 `pip install liger-kernel` 安装"
        )
        
    from liger_kernel.transformers import _apply_liger_kernel_to_instance
    
    # 检查是否为PEFT模型
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        _apply_liger_kernel_to_instance(model=base_model)
        logger.info("已对PEFT模型的基础模型应用Liger Kernel优化")
    else:
        _apply_liger_kernel_to_instance(model=model)
        logger.info("已对基础模型应用Liger Kernel优化")
    
    return model


def main():
    # 解析命令行参数
    parser = HfArgumentParser([ScriptArguments, ORPOConfig])
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 更新ORPOConfig参数
    training_args.beta = script_args.beta
    
    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 如果指定了lora_adapter_dir，确保它存在
    if script_args.lora_adapter_dir:
        os.makedirs(script_args.lora_adapter_dir, exist_ok=True)
    else:
        script_args.lora_adapter_dir = os.path.join(training_args.output_dir, "adapter")
        os.makedirs(script_args.lora_adapter_dir, exist_ok=True)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # 如果没有eos_token，使用一个常见的token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 准备模型加载参数
    model_kwargs = {}
    
    # 处理量化参数
    if script_args.use_4bit:
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
        })
    elif script_args.use_8bit:
        model_kwargs.update({"load_in_8bit": True})
    
    # 处理Flash Attention 2
    if script_args.use_flash_attention:
        model_kwargs.update({
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,  # 使用bf16可以获得更好的性能
        })
    
    # 加载模型
    logger.info(f"正在加载模型 {script_args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # 如果模型是一个PEFT模型并且有adapter_model路径，加载adapter
    if training_args.resume_from_checkpoint and "adapter_model" in os.listdir(training_args.resume_from_checkpoint):
        logger.info(f"从检查点加载adapter: {training_args.resume_from_checkpoint}/adapter_model")
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()  # 先合并adapter再加载新的
        
        # 准备模型进行LoRA训练
        if script_args.use_4bit or script_args.use_8bit:
            model = prepare_model_for_kbit_training(model)
            
        # 转换target_modules为列表
        target_modules = script_args.lora_target_modules.split(",")
        
        # 创建LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=target_modules,
        )
        
        # 先应用LoRA配置
        model = get_peft_model(model, peft_config)
        
        # 再次应用Liger Kernel到PEFT模型（如果启用）
        model = apply_liger_kernel(model, script_args)
        
        # 然后加载现有的adapter权重
        adapter_path = os.path.join(script_args.resume_from_checkpoint, "adapter_model")
        model.load_adapter(adapter_path)
        
    else:
        # 配置PEFT (LoRA)
        if script_args.use_4bit or script_args.use_8bit:
            model = prepare_model_for_kbit_training(model)
            
        # 转换target_modules为列表
        target_modules = script_args.lora_target_modules.split(",")
        
        # 创建LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=target_modules,
        )
        
        # 应用LoRA配置
        model = get_peft_model(model, peft_config)
        
        # 再次应用Liger Kernel到PEFT模型（如果启用）
        model = apply_liger_kernel(model, script_args)
    
    # 处理断点续训
    if script_args.resume_from_checkpoint:
        if script_args.resume_from_checkpoint == "latest":
            resume_from_checkpoint = find_latest_checkpoint(script_args.output_dir)
        else:
            resume_from_checkpoint = script_args.resume_from_checkpoint
        
        if resume_from_checkpoint is None:
            logger.warning("未找到检查点，将从头开始训练")
    else:
        resume_from_checkpoint = None
    
    # 加载数据集
    logger.info(f"加载数据 {script_args.train_file}...")
    full_dataset = load_dataset_from_json(script_args.train_file)
    
    # 处理评估数据集
    if script_args.eval_file:
        # 如果提供了单独的评估文件，则加载它
        logger.info(f"加载单独的评估数据 {script_args.eval_file}...")
        eval_dataset = load_dataset_from_json(script_args.eval_file)
        train_dataset = full_dataset
    else:
        # 否则，从训练集中划分出一部分作为评估集
        logger.info(f"从训练数据中划分 {script_args.eval_ratio*100:.1f}% 作为评估数据...")
        train_dataset, eval_dataset = split_dataset(full_dataset, script_args.eval_ratio)
    
    logger.info(f"训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")
    
    # 创建ORPO训练器
    logger.info("初始化ORPOTrainer...")
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # 注意这里使用processing_class而不是tokenizer
        peft_config=peft_config if not isinstance(model, PeftModel) else None,  # 如果已经是PeftModel，不需要再传配置
    )
    
    # 执行训练
    logger.info("开始训练...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 保存模型
    logger.info(f"保存模型到 {training_args.output_dir}...")
    trainer.save_model()
    
    # 单独保存adapter
    if isinstance(trainer.model, PeftModel):
        adapter_save_path = script_args.lora_adapter_dir
        logger.info(f"保存LoRA adapter到 {adapter_save_path}...")
        trainer.model.save_pretrained(adapter_save_path)
        
        # 同时保存tokenizer以便适配
        tokenizer.save_pretrained(adapter_save_path)
        logger.info(f"保存tokenizer到 {adapter_save_path}...")


if __name__ == "__main__":
    main()
