# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import shutil
import sys
import math
from tqdm.auto import tqdm
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    tag_count_reward,
)
from open_r1.utils.felix import (make_conversation,reasoning_steps_reward,format_reward,load_dataset_from_source)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.trainer.fast_grpo import FastGRPOTrainer, FastGRPOConfig
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
# vLLM 导入
try:
    from vllm import LLM, SamplingParams
    is_vllm_available = True
except ImportError:
    is_vllm_available = False

# 导入部分添加torch.distributed
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
        use_fast_trainer (`bool`):
            Whether to use the FastGRPOTrainer instead of the traditional GRPOTrainer.
        batch_size (`int`):
            Number of prompts to process in each batch during vLLM sampling.
        max_iterations (`int`):
            Maximum number of iterations to run. If -1, process the entire dataset.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
    use_fast_trainer: bool = field(
        default=False,
        metadata={"help": "Whether to use the FastGRPOTrainer instead of the traditional GRPOTrainer"},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Number of prompts to process in each batch during vLLM sampling"},
    )
    max_iterations: int = field(
        default=-1,
        metadata={"help": "Maximum number of iterations to run. If -1, process the entire dataset."},
    )
    vllm_tensor_parallel_size: int = field(
        default=-1,  # -1 means use all available GPUs
        metadata={"help": "vLLM tensor parallel size, -1 uses all GPUs"},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.95,
        metadata={"help": "vLLM GPU memory utilization ratio"},
    )
    vllm_max_model_len: int = field(
        default=16384,
        metadata={"help": "vLLM maximum model length"},
    )
    num_generations: int = field(
        default=16,
        metadata={"help": "Number of generations per prompt"},
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for generation"}
    )

def generate_samples_with_vllm(
    model_path, 
    prompts, 
    reward_funcs, 
    reward_weights,
    config,
    tokenizer,
    is_main_process=True,
    gpu_local_rank=0
):
    """使用vLLM生成样本并计算奖励，支持分布式环境

    Args:
        model_path: 模型路径
        prompts: 提示列表
        reward_funcs: 奖励函数列表
        reward_weights: 奖励权重
        config: 配置
        tokenizer: 分词器
        is_main_process: 是否是主进程
        gpu_local_rank: 本地GPU排序

    Returns:
        包含生成样本和奖励的列表
    """
    # 非主进程直接返回None
    if not is_main_process:
        return None
        
    # 以下代码只在主进程执行
    if not is_vllm_available:
        raise ImportError("请安装vLLM: pip install vllm")
        
    logger.info(f"vLLM将加载模型: {model_path}")
    
    # 计算vLLM应该使用的GPU数量
    available_gpus = torch.cuda.device_count()
    if config.vllm_tensor_parallel_size == -1:
        # 使用全部可用GPU
        tensor_parallel_size = available_gpus
    else:
        # 使用用户指定的值（但不超过可用数量）
        tensor_parallel_size = min(config.vllm_tensor_parallel_size, available_gpus)
        
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
                gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                max_model_len=config.vllm_max_model_len,
                enable_prefix_caching=True,
            )
        
        logger.info(f"vLLM服务已启动，使用{tensor_parallel_size}个GPU")
        
    except Exception as e:
        logger.error(f"vLLM启动失败: {e}")
        raise
    
    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.vllm_max_model_len,
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
        expanded_prompts.extend([prompt] * config.num_generations)
    
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
        prompt_idx = i // config.num_generations
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
            reward_kwargs = {key: [sample[key]] for key in extra_keys}
            
            reward = reward_func(
                prompts=[sample["prompt"]], 
                completions=[sample["completion"]],
                **reward_kwargs  # 传递额外参数
            )[0]
            rewards.append(reward)
        # 如果有多个奖励函数，使用权重计算总体奖励
        sample["reward"] = sum(r * w for r, w in zip(rewards, reward_weights)) / sum(reward_weights)
    
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

def prepare_training_data(samples):
    """将样本转换为训练数据格式"""
    logger.info("准备训练数据...")
    
    # 筛选样本，可以根据需要调整策略
    # positive_samples = [s for s in samples if s["advantage"] > 0]
    positive_samples = samples
    logger.info(f"使用{len(positive_samples)}/{len(samples)}个样本进行训练")
    
    # 准备数据集
    train_data = []
    
    for sample in tqdm(positive_samples, desc="处理训练样本"):
        # 创建基本数据项，包括必需字段
        sample_data = {
            "prompt": sample["prompt"],
            "completion": sample["completion"],
            "advantage": sample["advantage"],
            "reward": sample["reward"],
        }
        
        # 保留样本中的所有其他字段（除了prompt_idx，它只用于内部处理）
        for key in sample:
            if key not in sample_data and key != "prompt_idx":
                sample_data[key] = sample[key]
                
        train_data.append(sample_data)
    
    # 转换为Dataset格式
    return Dataset.from_list(train_data)

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset_from_source(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # 计算奖励权重
    if hasattr(script_args, "reward_weights") and script_args.reward_weights:
        if len(script_args.reward_weights) != len(reward_funcs):
            raise ValueError(
                f"奖励权重数量({len(script_args.reward_weights)})必须匹配奖励函数数量({len(reward_funcs)})"
            )
        reward_weights = torch.tensor(script_args.reward_weights, dtype=torch.float32)
    else:
        reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

    # 应用会话模板
    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    if script_args.use_fast_trainer:
        #############################
        # 使用分阶段的批量处理方法
        #############################
        logger.info("使用分阶段的批量处理方法进行训练")
        
        # 获取训练数据集
        train_dataset = dataset[script_args.dataset_train_split]
        eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
        
        # 计算批次总数
        total_samples = len(train_dataset)
        batch_size = script_args.batch_size
        total_batches = math.ceil(total_samples / batch_size)
        
        # 如果指定了最大迭代次数，限制批次数
        if script_args.max_iterations > 0:
            total_batches = min(total_batches, script_args.max_iterations)
        
        logger.info(f"数据集共有{total_samples}个样本，将分成{total_batches}个批次进行处理")
        
        # 创建输出目录
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # 初始化模型路径
        current_model_path = model_args.model_name_or_path
        
        # 创建一个FastGRPOConfig对象用于训练
        fast_config = FastGRPOConfig(
            output_dir=training_args.output_dir,
            learning_rate=training_args.learning_rate,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            max_steps=training_args.max_steps,
            num_train_epochs=training_args.num_train_epochs,
            max_prompt_length=training_args.max_prompt_length,
            max_completion_length=training_args.max_completion_length,
            beta=getattr(training_args, "beta", 0.1),
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            model_init_kwargs=training_args.model_init_kwargs,
            logging_steps=training_args.logging_steps,
            save_steps=training_args.save_steps,
            eval_steps=training_args.eval_steps,
            save_total_limit=training_args.save_total_limit,
            seed=training_args.seed,
            chat_template=training_args.chat_template,
        )
        
        # 批次迭代处理
        for batch_idx in tqdm(range(total_batches), desc="批次处理"):
            logger.info(f"===== 处理批次 {batch_idx+1}/{total_batches} =====")
            
            # 1. 获取当前批次的数据
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            current_batch = train_dataset.select(range(start_idx, end_idx))
            
            logger.info(f"当前批次包含{len(current_batch)}个样本")
            
            # 2. 使用vLLM生成样本（只在主进程执行）
            # 检测是否为主进程
            is_distributed = bool(training_args.local_rank != -1)
            is_main_process = not is_distributed or training_args.local_rank == 0
            gpu_local_rank = training_args.local_rank if is_distributed else 0
            
            # 执行采样
            if is_main_process:
                logger.info("主进程开始vLLM采样...")
                samples = generate_samples_with_vllm(
                    model_path=current_model_path,
                    prompts=current_batch,
                    reward_funcs=reward_funcs,
                    reward_weights=reward_weights,
                    config=script_args,
                    tokenizer=tokenizer,
                    is_main_process=is_main_process,
                    gpu_local_rank=gpu_local_rank
                )
                logger.info(f"采样完成，生成了{len(samples)}个样本")
            else:
                logger.info("非主进程等待vLLM采样结果...")
                samples = None
        
        # 创建FastGRPOTrainer
            logger.info("初始化训练器...")
            trainer = FastGRPOTrainer(
                model=current_model_path,
            reward_funcs=reward_funcs,
            args=fast_config,
                train_dataset=None,  # 先不设置数据集
                eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            )
            
            # 让训练器自己处理数据广播
            if is_main_process:
                logger.info("准备数据并交给训练器处理...")
                # 3. 准备训练数据
                training_data = prepare_training_data(samples)
                trainer.train_dataset = training_data
            
            # 等待主进程完成数据准备
            if is_distributed:
                logger.info("等待所有进程同步...")
                trainer.accelerator.wait_for_everyone()
            
            # 广播训练数据
            if is_distributed:
                if is_main_process:
                    training_data_list = [training_data]
                else:
                    training_data_list = [None]
                trainer.train_dataset = broadcast_object_list(training_data_list)[0]
            
            # 4. 训练模型
            logger.info("开始训练...")
            trainer.train()
            
            # 5. 保存模型
            batch_output_dir = os.path.join(training_args.output_dir, f"batch-{batch_idx}")
            if is_main_process:
                os.makedirs(batch_output_dir, exist_ok=True)
            trainer.save_model(batch_output_dir)
            if is_main_process:
                logger.info(f"模型已保存到 {batch_output_dir}")
            
            # 6. 更新模型路径用于下一批处理
            current_model_path = batch_output_dir
            
            # 清理内存
            del trainer
            if is_main_process and samples is not None:
                del training_data
                del samples
            torch.cuda.empty_cache()
            
            # 在批次结束后，合并LoRA适配器和基础模型
            if is_main_process:
                logger.info("合并LoRA适配器与基础模型...")
                # 检查是否使用了PEFT（LoRA）
                if model_args.use_peft:
                    from transformers import AutoModelForCausalLM
                    from peft import PeftModel, PeftConfig
                    
                    # 检查保存的是否为适配器
                    if os.path.exists(os.path.join(batch_output_dir, "adapter_config.json")):
                        # 加载基础模型
                        base_model = AutoModelForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            torch_dtype=torch_dtype,
                            trust_remote_code=model_args.trust_remote_code
                        )
                        # 加载适配器
                        adapter_model = PeftModel.from_pretrained(base_model, batch_output_dir)
                        # 合并模型
                        merged_model = adapter_model.merge_and_unload()
                        # 判断路径是否存在，存在则删除
                        if os.path.exists(os.path.join(training_args.output_dir, f"merged-batch")):
                            shutil.rmtree(os.path.join(training_args.output_dir, f"merged-batch"))
                        
                        # 保存合并后的模型
                        merged_dir = os.path.join(training_args.output_dir, f"merged-batch")
                        os.makedirs(merged_dir, exist_ok=True)
                        merged_model.save_pretrained(merged_dir)
                        # 同时保存分词器
                        tokenizer.save_pretrained(merged_dir)
                        # 更新模型路径
                        current_model_path = merged_dir
                        logger.info(f"LoRA适配器已与基础模型合并并保存到 {merged_dir}")
        
        # 训练完成后，保存最终模型
        if is_main_process:
            logger.info("所有批次处理完成，保存最终模型...")
            # os.system(f"cp -r {current_model_path}/* {training_args.output_dir}/")
            logger.info(f"最终模型已保存到 {training_args.output_dir}")
        
    else:
        #############################
        # Initialize the GRPO trainer
        #############################
        trainer = GRPOTrainer(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
        )

        ###############
        # Training loop
        ###############
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        ##################################
        # Save model and create model card
        ##################################
        logger.info("*** Save model ***")
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["open-r1"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

        ##########
        # Evaluate
        ##########
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        #############
        # push to hub
        #############
        if training_args.push_to_hub:
            logger.info("Pushing to hub...")
            trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, FastGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)