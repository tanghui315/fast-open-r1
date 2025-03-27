#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRPO训练流水线协调脚本

此脚本协调vLLM采样和GRPO训练的整个流程，实现了以下功能：
1. 使用vLLM进行批量采样
2. 使用采样结果进行GRPO训练
3. 迭代处理多个批次
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def run_command(cmd, desc=None):
    """运行命令并记录输出"""
    if desc:
        logger.info(f"执行: {desc}")
    
    logger.info(f"命令: {cmd}")
    
    # 创建一个新的进程并捕获输出
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # 实时输出命令结果
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        sys.stdout.flush()
    
    # 等待命令完成并检查退出码
    exit_code = process.wait()
    if exit_code != 0:
        logger.error(f"命令执行失败，退出码: {exit_code}")
        raise subprocess.CalledProcessError(exit_code, cmd)
    
    return exit_code

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GRPO训练流水线")
    
    # 必需参数
    parser.add_argument("--model", required=True, help="初始模型路径或名称")
    parser.add_argument("--dataset", required=True, help="数据集名称或路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    
    # 流水线控制参数
    parser.add_argument("--num-batches", type=int, default=10, help="处理的批次数量")
    parser.add_argument("--batch-size", type=int, default=32, help="每批处理的样本数量")
    parser.add_argument("--resume-from", type=int, default=0, help="从指定批次恢复")
    
    # 采样参数
    parser.add_argument("--dataset-config", help="数据集配置名称")
    parser.add_argument("--dataset-split", default="train", help="使用的数据集分割")
    parser.add_argument("--eval-split", default="test", help="使用的测试/验证集分割名称，通常为test或validation")
    parser.add_argument("--include-eval", action="store_true", help="是否也处理测试/验证数据集")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="测试/验证数据集每批处理的样本数量")
    parser.add_argument("--num-generations", type=int, default=16, help="每个提示生成的样本数量")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=-1, help="vLLM张量并行大小")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.95, help="vLLM GPU内存利用率")
    parser.add_argument("--reward-funcs", nargs="+", default=["accuracy", "format", "tag_count"], 
                       help="奖励函数列表")
    
    # 训练参数
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="每个设备的训练批次大小")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="每个设备的评估批次大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--max-steps", type=int, default=-1, help="每批次的最大训练步数")
    parser.add_argument("--beta", type=float, default=0.04, help="GRPO的KL散度系数")
    parser.add_argument("--fp16", action="store_true", help="是否启用fp16训练")
    parser.add_argument("--bf16", action="store_true", help="是否启用bf16训练")
    parser.add_argument("--do-eval", action="store_true", help="是否在训练期间进行评估")
    parser.add_argument("--eval-strategy", default="steps", choices=["no", "steps", "epoch"], 
                       help="评估策略")
    parser.add_argument("--eval-steps", type=int, default=100, help="每隔多少步进行一次评估")
    
    # PEFT参数
    parser.add_argument("--use-peft", action="store_true", help="是否使用PEFT进行训练")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA的秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha参数")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout率")
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj","o_proj","k_proj","gate_proj","down_proj","up_proj"], help="LoRA目标模块")
    
    # 其他参数
    parser.add_argument("--chat-template", help="使用的聊天模板")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="是否信任远程代码")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--report-to", default="none", help="报告训练指标的工具")
    parser.add_argument("--accelerate-args", default="", help="传递给accelerate launch的额外参数")
    parser.add_argument("--save-steps", type=int, default=200, help="每隔多少步保存一次模型检查点")
    
    # 添加新参数
    parser.add_argument("--num-train-epochs", type=int, default=1, 
                       help="处理整个数据集的轮数，默认为1轮")
    parser.add_argument("--total-dataset-size", type=int, default=None,
                       help="数据集总大小，如果不提供将在首次运行时自动计算")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(args.output_dir, "pipeline_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 计算训练集大小
    train_dataset_size = args.total_dataset_size
    if train_dataset_size is None and args.resume_from == 0:
        # 只在第一次运行时计算数据集大小
        logger.info(f"计算数据集大小...")
        import datasets
        from open_r1.utils.felix import load_dataset_from_source
        
        dataset = load_dataset_from_source(args.dataset, name=args.dataset_config)
        if args.dataset_split in dataset:
            train_dataset_size = len(dataset[args.dataset_split])
            logger.info(f"训练数据集'{args.dataset_split}'分割共包含{train_dataset_size}个样本")
        else:
            logger.error(f"找不到数据集分割'{args.dataset_split}'，使用默认批次数量")
            train_dataset_size = args.batch_size * args.num_batches
    elif train_dataset_size is None:
        # 如果从中间恢复且未提供数据集大小，使用默认批次
        logger.warning(f"从批次{args.resume_from}恢复但未提供数据集大小，使用默认批次数量")
        train_dataset_size = args.batch_size * args.num_batches
    
    # 计算训练集的每轮批次数
    train_batches_per_epoch = (train_dataset_size + args.batch_size - 1) // args.batch_size
    total_batches = train_batches_per_epoch * args.num_train_epochs
    
    # 如果需要处理评估集，也计算其大小和批次
    eval_dataset_size = None
    eval_batches_per_epoch = None
    
    if args.include_eval and args.resume_from == 0:
        try:
            dataset = load_dataset_from_source(args.dataset, name=args.dataset_config)
            
            # 确定要使用的评估分割名称
            eval_split_name = None
            if args.eval_split in dataset:
                eval_split_name = args.eval_split
            elif "test" in dataset:
                eval_split_name = "test"
                logger.info(f"找不到评估分割'{args.eval_split}'，使用'test'分割代替")
            elif "validation" in dataset:
                eval_split_name = "validation"
                logger.info(f"找不到评估分割'{args.eval_split}'或'test'，使用'validation'分割代替")
            
            if eval_split_name:
                eval_dataset_size = len(dataset[eval_split_name])
                logger.info(f"评估数据集'{eval_split_name}'分割共包含{eval_dataset_size}个样本")
                eval_batches_per_epoch = (eval_dataset_size + args.eval_batch_size - 1) // args.eval_batch_size
            else:
                logger.warning(f"找不到有效的评估分割，可用分割: {', '.join(dataset.keys())}")
        except Exception as e:
            logger.warning(f"获取评估集大小时出错: {e}")
    
    logger.info(f"训练数据集大小: {train_dataset_size}, 每轮批次数: {train_batches_per_epoch}, 总批次数: {total_batches}")
    if eval_dataset_size:
        logger.info(f"评估数据集大小: {eval_dataset_size}, 每轮批次数: {eval_batches_per_epoch}")
    
    # 设置初始模型路径
    current_model_path = args.model
    
    # 批次处理 - 修改循环逻辑
    for global_batch_idx in range(args.resume_from, total_batches):
        # 计算当前轮次(epoch)和轮内批次索引
        current_epoch = global_batch_idx // train_batches_per_epoch
        train_batch_in_epoch = global_batch_idx % train_batches_per_epoch
        
        # 如果处理评估集，计算对应的评估集批次索引
        eval_batch_in_epoch = None
        if eval_batches_per_epoch:
            # 按比例映射 - 保持相对位置一致
            eval_batch_in_epoch = min(
                int(train_batch_in_epoch * eval_batches_per_epoch / train_batches_per_epoch),
                eval_batches_per_epoch - 1
            )
        
        logger.info(f"===== 处理轮次 {current_epoch+1}/{args.num_train_epochs}, "
                   f"批次 {train_batch_in_epoch+1}/{train_batches_per_epoch} "
                   f"(全局批次 {global_batch_idx+1}/{total_batches}) =====")
        
        # 创建全局批次目录
        batch_dir = os.path.join(args.output_dir, f"batch-{global_batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # 使用train_batch_in_epoch创建训练样本文件
        train_samples_file = os.path.join(batch_dir, f"samples_train_batch{train_batch_in_epoch}.jsonl")
        
        logger.info(f"正在生成训练样本到 {train_samples_file}...")
        
        vllm_cmd = f"""
        python -m open_r1.vllm_sampler \\
          --model_path {current_model_path} \\
          --output_file {train_samples_file} \\
          --dataset_name {args.dataset} \\
          --dataset_split {args.dataset_split} \\
          --batch_size {args.batch_size} \\
          --batch_idx {train_batch_in_epoch} \\
          --num_generations {args.num_generations} \\
          --temperature {args.temperature} \\
          --vllm_tensor_parallel_size {args.vllm_tensor_parallel_size} \\
          --vllm_gpu_memory_utilization {args.vllm_gpu_memory_utilization} \\
          --reward_funcs {' '.join(args.reward_funcs)} \\
          --seed {args.seed}"""
        
        # 添加评估分割处理
        eval_samples_file = None
        if args.include_eval and eval_batch_in_epoch is not None:
            # 使用eval_batch_in_epoch创建评估样本文件
            eval_samples_file = os.path.join(batch_dir, f"samples_{args.eval_split}_batch{eval_batch_in_epoch}.jsonl")
            
            eval_cmd_part = f" \\\n  --include_eval --eval_split {args.eval_split} --eval_batch_size {args.eval_batch_size}"
            eval_cmd_part += f" \\\n  --eval_batch_idx {eval_batch_in_epoch} --eval_output_file {eval_samples_file}"
            vllm_cmd += eval_cmd_part
        
        # 添加可选参数
        if args.dataset_config:
            vllm_cmd += f" \\\n  --dataset_config {args.dataset_config}"
        if args.chat_template:
            vllm_cmd += f" \\\n  --chat_template {args.chat_template}"
        if not args.trust_remote_code:
            vllm_cmd += " \\\n  --no-trust_remote_code"
        
        # 运行vLLM采样命令
        run_command(vllm_cmd.strip(), "vLLM采样")
        
        # 2. 训练阶段
        train_output_dir = os.path.join(batch_dir, "model")
        os.makedirs(train_output_dir, exist_ok=True)
        
        logger.info(f"使用样本 {train_samples_file} 进行训练...")
        
        # 构建基本训练命令
        train_cmd = f"""
        accelerate launch {args.accelerate_args} --config_file recipes/accelerate_configs/zero3.yaml \\
          --num_processes=4 src/open_r1/train_grpo.py \\
          --model_path {current_model_path} \\
          --output_dir {train_output_dir} \\
          --samples_file {train_samples_file} \\
          --save_steps {args.save_steps} \\
          --learning_rate {args.learning_rate} \\
          --per_device_train_batch_size {args.per_device_train_batch_size} \\
          --gradient_accumulation_steps {args.gradient_accumulation_steps} \\
          --max_steps {args.max_steps} \\
          --beta {args.beta} \\
          --seed {args.seed} \\
          --report_to {args.report_to}
        """
        
        # 添加验证数据
        if args.do_eval and args.include_eval and eval_samples_file and os.path.exists(eval_samples_file):
            logger.info(f"使用评估样本 {eval_samples_file} 进行评估...")
            train_cmd += f" --eval_samples_file {eval_samples_file}"
            train_cmd += f" --per_device_eval_batch_size {args.per_device_eval_batch_size}"
            train_cmd += f" --eval_strategy {args.eval_strategy}"
            if args.eval_strategy == "steps":
                train_cmd += f" --eval_steps {args.eval_steps}"
        
        # 添加可选训练参数
        if args.fp16:
            train_cmd += " --fp16"
        if args.bf16:
            train_cmd += " --bf16"
        if args.chat_template:
            train_cmd += f" --chat_template {args.chat_template}"
        if args.trust_remote_code:
            train_cmd += " --trust_remote_code"
        
        # 添加PEFT参数
        if args.use_peft:
            train_cmd += f" --use_peft --lora_r {args.lora_r} --lora_alpha {args.lora_alpha} --lora_dropout {args.lora_dropout}"
            if args.lora_target_modules:
                train_cmd += f" --lora_target_modules {' '.join(args.lora_target_modules)}"
        
        # 运行训练命令
        run_command(train_cmd.strip(), "GRPO训练")
        
        # 3. 更新模型路径
        # 如果使用了PEFT且生成了merged模型，使用合并后的模型
        merged_dir = os.path.join(train_output_dir, "merged")
        if args.use_peft and os.path.exists(merged_dir):
            current_model_path = merged_dir
        else:
            current_model_path = train_output_dir
        
        logger.info(f"批次 {global_batch_idx+1} 完成，下一批次将使用模型: {current_model_path}")
    
    # 创建最终模型（复制最后一个批次的模型）
    final_model_path = os.path.join(args.output_dir, "final_model")
    if os.path.exists(final_model_path):
        shutil.rmtree(final_model_path)
    
    logger.info(f"复制最终模型从 {current_model_path} 到 {final_model_path}...")
    shutil.copytree(current_model_path, final_model_path)
    
    logger.info(f"训练完成！最终模型已保存到: {final_model_path}")

if __name__ == "__main__":
    main() 