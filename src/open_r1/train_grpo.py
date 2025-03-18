#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRPO训练脚本 - 使用从vLLM采样器生成的样本进行GRPO训练

此脚本不包含采样逻辑，专注于高效训练。
"""

import json
import logging
import os
import sys
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from tqdm.auto import tqdm

import datasets
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser, set_seed

from open_r1.trainer.fast_grpo import FastGRPOTrainer, FastGRPOConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)

@dataclass
class TrainerArguments:
    """
    GRPO训练器的命令行参数
    """
    model_path: str = field(
        metadata={"help": "模型路径或名称"}
    )
    output_dir: str = field(
        metadata={"help": "输出目录，用于保存模型"}
    )
    samples_file: str = field(
        metadata={"help": "包含生成样本的文件路径(.json或.jsonl)"}
    )
    eval_samples_file: Optional[str] = field(
        default=None,
        metadata={"help": "包含验证样本的文件路径(.json或.jsonl)"}
    )
    # 常用训练参数
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "学习率"}
    )
    per_device_train_batch_size: int = field(
        default=4, 
        metadata={"help": "每个设备的训练批次大小"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "每个设备的评估批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=8, 
        metadata={"help": "梯度累积步数"}
    )
    max_steps: int = field(
        default=1000, 
        metadata={"help": "最大训练步数"}
    )
    logging_steps: int = field(
        default=1, 
        metadata={"help": "日志记录步数"}
    )
    save_steps: int = field(
        default=200, 
        metadata={"help": "保存模型的步数"}
    )
    eval_strategy: str = field(
        default="no",
        metadata={"help": "评估策略: 'no'、'steps'或'epoch'"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "每隔多少步评估一次"}
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "GRPO的KL散度系数"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "是否启用fp16训练"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "是否启用bf16训练"}
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "使用的聊天模板"}
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "提示的最大长度"}
    )
    max_completion_length: int = field(
        default=8192,
        metadata={"help": "生成的最大长度"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )
    report_to: str = field(
        default="none",
        metadata={"help": "报告训练指标的工具，例如'wandb'或'tensorboard'"}
    )
    # PEFT参数
    use_peft: bool = field(
        default=False,
        metadata={"help": "是否使用PEFT进行训练"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA的秩"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout率"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "LoRA目标模块"}
    )

def load_samples(samples_file: str) -> List[Dict[str, Any]]:
    """从文件加载样本数据"""
    logger.info(f"从 {samples_file} 加载样本...")
    
    if not os.path.exists(samples_file):
        raise FileNotFoundError(f"样本文件不存在: {samples_file}")
    
    # 根据文件扩展名选择加载方式
    if samples_file.endswith('.jsonl'):
        with open(samples_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
    else:
        with open(samples_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    
    logger.info(f"已加载 {len(samples)} 个样本")
    return samples

def prepare_training_data(samples: List[Dict[str, Any]]) -> Dataset:
    """将样本转换为训练数据格式"""
    logger.info("准备训练数据...")
    
    # 可以在这里实现样本筛选逻辑，例如只选择正优势的样本
    # positive_samples = [s for s in samples if s["advantage"] > 0]
    positive_samples = samples
    logger.info(f"使用 {len(positive_samples)}/{len(samples)} 个样本进行训练")
    
    # 准备数据集
    train_data = []
    for sample in positive_samples:
        train_data.append({
            "prompt": sample["prompt"],
            "completion": sample["completion"],
            "advantage": sample["advantage"],
            "reward": sample["reward"],
            # 确保保留其他可能的字段（如参考答案等）
            **{k: v for k, v in sample.items() 
               if k not in ["prompt", "completion", "advantage", "reward", "prompt_idx"]}
        })
    
    # 转换为Dataset格式
    return Dataset.from_list(train_data)

def main():
    # 解析命令行参数
    parser = HfArgumentParser((TrainerArguments, ModelConfig))
    args, model_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载样本数据
    train_samples = load_samples(args.samples_file)
    
    # 准备训练数据
    train_dataset = prepare_training_data(train_samples)
    
    # 如果提供了验证样本文件，加载并准备验证数据
    eval_dataset = None
    if args.eval_samples_file:
        eval_samples = load_samples(args.eval_samples_file)
        eval_dataset = prepare_training_data(eval_samples)
        logger.info(f"已准备 {len(eval_dataset)} 个验证样本")
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {args.model_path}")
    tokenizer = get_tokenizer(model_args, trust_remote_code=args.trust_remote_code)
    if args.chat_template and tokenizer.chat_template is None:
        tokenizer.chat_template = args.chat_template
    
    # 配置训练参数
    training_args = FastGRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
        chat_template=args.chat_template,
    )
    
    # 如果使用wandb，初始化
    if "wandb" in args.report_to:
        init_wandb_training(training_args)
    
    # 初始化并编译空白的奖励函数，因为样本已经包含计算好的奖励
    # 在这里，我们只需要一个占位奖励函数
    def dummy_reward_func(*args, **kwargs):
        return [0.0]
    
    # 获取PEFT配置
    peft_config = get_peft_config(model_args) if args.use_peft else None
    
    logger.info("初始化FastGRPOTrainer...")
    trainer = FastGRPOTrainer(
        model=args.model_path,
        reward_funcs=[dummy_reward_func],  # 只是一个占位符，不会被使用
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=get_callbacks(training_args, model_args) if "wandb" in args.report_to else None,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 如果有评估数据集，运行一次最终评估
    if eval_dataset and args.eval_strategy != "no":
        logger.info("运行最终评估...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # 保存最终模型
    logger.info("训练完成，保存模型...")
    trainer.save_model(args.output_dir)
    
    # 保存分词器
    tokenizer.save_pretrained(args.output_dir)
    
    # 如果使用了PEFT，创建合并模型
    if args.use_peft:
        logger.info("合并LoRA适配器和基础模型...")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        
        # 检查保存的是否为适配器
        if os.path.exists(os.path.join(args.output_dir, "adapter_config.json")):
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype="auto",
                trust_remote_code=args.trust_remote_code
            )
            # 加载适配器
            adapter_model = PeftModel.from_pretrained(base_model, args.output_dir)
            # 合并模型
            merged_model = adapter_model.merge_and_unload()
            # 保存合并后的模型
            merged_dir = os.path.join(args.output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
            merged_model.save_pretrained(merged_dir)
            # 同时保存分词器
            tokenizer.save_pretrained(merged_dir)
            logger.info(f"合并模型已保存到: {merged_dir}")
    
    logger.info(f"模型已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 