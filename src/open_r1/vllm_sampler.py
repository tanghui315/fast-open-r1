#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vLLM采样器 - 使用vLLM引擎生成样本并计算奖励

此脚本独立于训练过程，专注于高效生成样本。
"""

import argparse
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
from transformers import HfArgumentParser, set_seed, AutoTokenizer

# 导入奖励函数
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    tag_count_reward,
)
from open_r1.utils.felix import (
    make_conversation,
    reasoning_steps_reward,
    format_reward,
    load_dataset_from_source,
)
from open_r1.utils import get_tokenizer

# vLLM导入
try:
    from vllm import LLM, SamplingParams
    is_vllm_available = True
except ImportError:
    is_vllm_available = False

logger = logging.getLogger(__name__)

@dataclass
class SamplerArguments:
    """
    vLLM采样器的命令行参数
    """
    model_path: str = field(
        metadata={"help": "模型路径或名称"}
    )
    output_file: str = field(
        metadata={"help": "生成样本的输出文件路径（.json或.jsonl格式）"}
    )
    dataset_name: str = field(
        metadata={"help": "数据集名称或路径"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "数据集配置名称"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "使用的数据集分割"}
    )
    eval_split: str = field(
        default="test",
        metadata={"help": "使用的测试/验证集分割名称，通常为test或validation"}
    )
    include_eval: bool = field(
        default=False,
        metadata={"help": "是否也处理测试/验证数据集"}
    )
    eval_batch_size: int = field(
        default=32,
        metadata={"help": "测试/验证数据集每批处理的样本数量"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "每批处理的样本数量"}
    )
    batch_idx: int = field(
        default=0,
        metadata={"help": "当前处理的批次索引"}
    )
    max_batches: int = field(
        default=-1,
        metadata={"help": "最大处理批次数量，-1表示处理整个数据集"}
    )
    num_generations: int = field(
        default=16,
        metadata={"help": "每个提示生成的样本数量"}
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "采样温度"}
    )
    vllm_tensor_parallel_size: int = field(
        default=-1,
        metadata={"help": "vLLM张量并行大小，-1表示使用所有可用GPU"}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.95,
        metadata={"help": "vLLM GPU内存利用率"}
    )
    vllm_max_model_len: int = field(
        default=16384,
        metadata={"help": "vLLM最大模型长度"}
    )
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "奖励函数列表。可选值: 'accuracy', 'format', 'reasoning_steps', 'cosine', "
                   "'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "奖励函数的权重，如果为None则平均所有奖励"}
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "错误答案的最小奖励值"}
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "错误答案的最大奖励值"}
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "正确答案的最小奖励值"}
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "正确答案的最大奖励值"}
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "缩放的最大长度"}
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "重复惩罚的n-gram大小"}
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "重复惩罚的最大(负)惩罚"}
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "代码格式奖励的语言"
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "使用的聊天模板"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )
    eval_batch_idx: Optional[int] = field(
        default=None,
        metadata={"help": "评估数据集的批次索引，如果为None则使用batch_idx"}
    )

def generate_samples_with_vllm(
    model_path: str, 
    prompts: List[Union[str, Dict[str, Any]]], 
    reward_funcs: List[callable], 
    reward_weights: torch.Tensor,
    args: SamplerArguments,
    tokenizer
) -> List[Dict[str, Any]]:
    """
    使用vLLM生成样本并计算奖励

    Args:
        model_path: 模型路径
        prompts: 提示列表
        reward_funcs: 奖励函数列表
        reward_weights: 奖励权重
        args: 采样器参数
        tokenizer: 分词器

    Returns:
        包含生成样本和奖励的列表
    """
    if not is_vllm_available:
        raise ImportError("请安装vLLM: pip install vllm")
        
    logger.info(f"vLLM将加载模型: {model_path}")
    
    # 计算vLLM应该使用的GPU数量
    available_gpus = torch.cuda.device_count()
    if args.vllm_tensor_parallel_size == -1:
        # 使用全部可用GPU
        tensor_parallel_size = available_gpus
    else:
        # 使用用户指定的值（但不超过可用数量）
        tensor_parallel_size = min(args.vllm_tensor_parallel_size, available_gpus)
        
    logger.info(f"系统检测到{available_gpus}个GPU，vLLM将使用{tensor_parallel_size}个GPU")
    
    # 初始化vLLM引擎
    try:
        # 修补vLLM的torch.distributed.get_world_size检查
        from unittest.mock import patch
        world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        profiling_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", 
            return_value=None
        )
        
        with world_size_patch, profiling_patch:
            vllm_engine = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
                enable_prefix_caching=True,
                trust_remote_code=args.trust_remote_code,
            )
        
        logger.info(f"vLLM服务已启动，使用{tensor_parallel_size}个GPU")
        
    except Exception as e:
        logger.error(f"vLLM启动失败: {e}")
        raise
    
    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.vllm_max_model_len,
    )
    
    # 保存原始提示的所有字段（包括可能的额外信息）
    original_samples = []
    if hasattr(prompts[0], "__getitem__") and isinstance(prompts[0], dict):
        # 如果prompts是字典格式（可能包含prompt以外的字段）
        prompt_texts = [p["prompt"] for p in prompts]
        original_samples = prompts  # 保留所有原始数据
    else:
        # 如果prompts已经是文本格式
        prompt_texts = prompts
        original_samples = [{"prompt": p} for p in prompts]  # 创建简单的字典
        
    # 扩展提示以生成多个样本
    expanded_prompts = []
    for prompt in prompt_texts:
        expanded_prompts.extend([prompt] * args.num_generations)
    
    logger.info(f"为{len(prompt_texts)}个提示生成{len(expanded_prompts)}个样本")
    
    # 使用vLLM批量生成
    outputs = vllm_engine.generate(
        expanded_prompts, 
        sampling_params=sampling_params,
        use_tqdm=True  # vLLM已内置进度条
    )
    
    # 整理结果
    results = []
    # 添加进度条用于处理生成结果
    logger.info("处理生成的结果...")
    for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="处理生成结果"):
        prompt_idx = i // args.num_generations
        original_prompt = prompt_texts[prompt_idx]
        original_sample = original_samples[prompt_idx]  # 获取原始样本的所有字段
        
        for completion_output in output.outputs:
            completion = completion_output.text
            result_item = {
                "prompt": original_prompt,
                "completion": completion,
                "prompt_idx": prompt_idx,
            }
            
            # 添加原始样本中的所有其他字段（保留额外信息，如参考答案等）
            for key, value in original_sample.items():
                if key != "prompt" and key not in result_item:
                    result_item[key] = value
                    
            results.append(result_item)
            
    # 计算奖励和优势
    logger.info("计算奖励和优势...")
    
    # 按照prompt_idx分组样本
    grouped_samples = {}
    for sample in results:
        prompt_idx = sample["prompt_idx"]
        if prompt_idx not in grouped_samples:
            grouped_samples[prompt_idx] = []
        grouped_samples[prompt_idx].append(sample)
    
    # 对每个样本计算奖励
    for sample in tqdm(results, desc="计算奖励"):
        rewards = []
        for reward_func in reward_funcs:
            # 收集除了"prompt"、"completion"和"prompt_idx"之外的所有字段作为额外参数
            extra_keys = [key for key in sample if key not in ["prompt", "completion", "prompt_idx", "reward", "advantage"]]
            reward_kwargs = {key: sample[key] for key in extra_keys}
            
            # 为了兼容reward_func的接口，将单个样本包装成列表
            reward = reward_func(
                prompts=[sample["prompt"]], 
                completions=[sample["completion"]],
                **{k: [v] for k, v in reward_kwargs.items()}  # 将每个值包装成列表
            )[0]
            rewards.append(reward)
        
        # 如果有多个奖励函数，使用权重计算总体奖励
        if reward_weights is not None:
            weighted_rewards = [r * w for r, w in zip(rewards, reward_weights)]
            sample["reward"] = sum(weighted_rewards) / sum(reward_weights)
        else:
            sample["reward"] = sum(rewards) / len(rewards)
    
    # 计算优势
    logger.info("计算组内优势值...")
    for prompt_idx, group in tqdm(grouped_samples.items(), desc="计算优势值", total=len(grouped_samples)):
        rewards = [s["reward"] for s in group]
        mean_reward = sum(rewards) / len(rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
        
        for sample in group:
            # 计算标准化的优势值
            sample["advantage"] = (sample["reward"] - mean_reward) / (std_reward + 1e-4)
            
    # 关闭vLLM引擎
    logger.info("正在关闭vLLM服务...")
    del vllm_engine
    # 强制执行垃圾收集
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("已关闭vLLM服务并释放GPU资源")
    
    return results

def save_samples(samples: List[Dict[str, Any]], output_file: str):
    """保存样本到文件"""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 根据文件扩展名选择保存格式
    if output_file.endswith('.jsonl'):
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已将{len(samples)}个样本保存到 {output_file}")

def process_dataset_split(
    dataset_split,
    split_name,
    args,
    batch_idx,
    batch_size,
    tokenizer,
    reward_funcs,
    reward_weights,
    output_file
):
    """处理指定的数据集分割"""
    logger.info(f"处理{split_name}数据集...")
    
    # 应用会话模板
    processed_dataset = dataset_split.map(make_conversation)
    if "messages" in processed_dataset.column_names:
        processed_dataset = processed_dataset.remove_columns("messages")
    
    # 计算批次范围
    total_samples = len(processed_dataset)
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, total_samples)
    
    if start_idx >= total_samples:
        logger.error(f"批次索引{batch_idx}超出{split_name}数据集范围(总样本数: {total_samples})")
        return None
    
    logger.info(f"处理{split_name}批次 {batch_idx}: 样本 {start_idx} 到 {end_idx} (共 {total_samples} 个样本)")
    current_batch = processed_dataset.select(range(start_idx, end_idx))
    
    # 生成样本
    logger.info(f"开始为{split_name}数据集使用vLLM生成样本...")
    samples = generate_samples_with_vllm(
        model_path=args.model_path,
        prompts=current_batch,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        args=args,
        tokenizer=tokenizer
    )
    
    # 保存样本
    save_samples(samples, output_file)
    
    return samples

def main():
    # 设置命令行参数
    parser = HfArgumentParser(SamplerArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset_name}")
    dataset = load_dataset_from_source(args.dataset_name, name=args.dataset_config)
    
    # 验证训练数据集
    if args.dataset_split not in dataset:
        available_splits = ", ".join(dataset.keys())
        raise ValueError(f"数据集分割'{args.dataset_split}'不存在。可用分割: {available_splits}")
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=args.trust_remote_code
    )
    if args.chat_template and tokenizer.chat_template is None:
        tokenizer.chat_template = args.chat_template
    
    # 获取奖励函数
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=args.cosine_min_value_wrong,
            max_value_wrong=args.cosine_max_value_wrong,
            min_value_correct=args.cosine_min_value_correct,
            max_value_correct=args.cosine_max_value_correct,
            max_len=args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=args.repetition_n_grams,
            max_penalty=args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=args.code_language),
        "tag_count": tag_count_reward,
    }
    
    # 获取用户指定的奖励函数
    reward_funcs = []
    for func_name in args.reward_funcs:
        if func_name in REWARD_FUNCS_REGISTRY:
            reward_funcs.append(REWARD_FUNCS_REGISTRY[func_name])
        else:
            available_funcs = ", ".join(REWARD_FUNCS_REGISTRY.keys())
            raise ValueError(f"奖励函数'{func_name}'不存在。可用函数: {available_funcs}")
    
    # 计算奖励权重
    if args.reward_weights:
        if len(args.reward_weights) != len(reward_funcs):
            raise ValueError(
                f"奖励权重数量({len(args.reward_weights)})必须匹配奖励函数数量({len(reward_funcs)})"
            )
        reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
    else:
        reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
    
    # 处理训练数据集
    process_dataset_split(
        dataset_split=dataset[args.dataset_split],
        split_name="训练",
        args=args,
        batch_idx=args.batch_idx,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        output_file=args.output_file
    )
    
    # 如果需要，处理测试/验证数据集
    if args.include_eval:
        # 首先检查指定的评估分割是否存在
        if args.eval_split in dataset:
            eval_split_name = args.eval_split
        # 如果没有找到指定的分割，尝试常见的评估分割名称
        elif "test" in dataset:
            eval_split_name = "test"
            logger.info(f"找不到分割'{args.eval_split}'，使用'test'分割代替")
        elif "validation" in dataset:
            eval_split_name = "validation"
            logger.info(f"找不到分割'{args.eval_split}'或'test'，使用'validation'分割代替")
        else:
            available_splits = ", ".join(dataset.keys())
            logger.warning(f"找不到有效的评估分割，可用分割: {available_splits}。跳过评估数据处理。")
            return
            
        # 准备评估数据集的输出文件
        eval_output_file = args.output_file
        if eval_output_file.endswith('.json'):
            eval_output_file = eval_output_file.replace('.json', f'_{eval_split_name}.json')
        elif eval_output_file.endswith('.jsonl'):
            eval_output_file = eval_output_file.replace('.jsonl', f'_{eval_split_name}.jsonl')
        else:
            eval_output_file = f"{eval_output_file}_{eval_split_name}"
        
        # 使用独立的评估批次索引
        batch_idx_for_eval = args.eval_batch_idx if args.eval_batch_idx is not None else args.batch_idx
        
        process_dataset_split(
            dataset_split=dataset[eval_split_name],
            split_name=f"评估({eval_split_name})",
            args=args,
            batch_idx=batch_idx_for_eval,  # 使用专门的评估批次索引
            batch_size=args.eval_batch_size,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            output_file=eval_output_file
        )
    
    logger.info("采样完成！")

if __name__ == "__main__":
    main() 