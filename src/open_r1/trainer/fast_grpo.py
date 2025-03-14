# fast_grpo_trainer.py
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from accelerate.utils import gather_object, set_seed
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

@dataclass
class FastGRPOConfig(TrainingArguments):
    """继承TrainingArguments并添加GRPO特定参数"""
    
    # GRPO算法参数
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "提示的最大长度"}
    )
    max_completion_length: int = field(
        default=8192,
        metadata={"help": "生成的最大长度"}
    )
    num_generations: int = field(
        default=16,
        metadata={"help": "每个提示生成的样本数量"},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "GRPO的KL散度系数"}
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "生成时的温度"}
    )
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "用于初始化模型的参数"}
    )
    # 奖励相关参数
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "奖励函数的权重，如果为None则平均所有奖励"}
    )
    chat_template: Optional[str] = field(
        default=None, 
        metadata={"help": "The chat template to use."}
    )

class FastGRPOTrainer(Trainer):
    """
    优化后的FastGRPOTrainer类，专注于GRPO训练过程。
    该类不再包含vLLM采样逻辑，采样过程由外部处理。
    
    工作流程:
    1. 接收已经生成好的包含奖励和优势值的训练样本
    2. 使用Trainer的功能进行训练
    3. 更新和保存模型
    """
    
    def __init__(
        self,
        model: Union[str, nn.Module],
        reward_funcs: Union[Callable, List[Callable]],
        args: Optional[FastGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[Any] = None,
        peft_config: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
    ):
        # 设置默认args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = FastGRPOConfig(f"{model_name}-FastGRPO")
        
        self.model_name = model if isinstance(model, str) else model.config._name_or_path
        
        # 设置随机种子确保再现性
        set_seed(args.seed)
        
        # 处理奖励函数
        self.reward_funcs = [reward_funcs] if not isinstance(reward_funcs, list) else reward_funcs
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(self.reward_funcs):
                raise ValueError(
                    f"奖励权重数量({len(args.reward_weights)})必须匹配奖励函数数量({len(self.reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)
        
        # 设置tokenizer
        self.processing_class = processing_class or AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token = self.processing_class.eos_token
            
        # 创建参考模型配置
        self.peft_config = peft_config
        
        def data_collator(features):  # 无需特殊的数据整理
            return features
        
        # 调用父类初始化，延迟加载model
        model_init = self._get_model_init(model) if isinstance(model, str) else None
    
        super().__init__(
            model=None if model_init else model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processing_class, 
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        # 初始化参考模型
        self.init_ref_model()
    
    def _get_model_init(self, model_name):
        """返回一个加载模型的函数"""
        def model_init():
            # 从model_init_kwargs获取torch_dtype
            torch_dtype = None
            if self.args.model_init_kwargs and 'torch_dtype' in self.args.model_init_kwargs:
                torch_dtype_str = self.args.model_init_kwargs['torch_dtype']
                if isinstance(torch_dtype_str, str) and torch_dtype_str not in ['auto', None] and hasattr(torch, torch_dtype_str):
                    torch_dtype = getattr(torch, torch_dtype_str)
                elif torch_dtype_str == 'auto':
                    torch_dtype = 'auto'
                
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                use_cache=False if self.args.gradient_checkpointing else True,
            )
        return model_init
    
    def init_ref_model(self):
        """初始化参考模型用于KL散度计算"""
        logger.info("初始化参考模型...")
        
        if self.model is None:
            self.call_model_init()  # 确保主模型已加载
            
        # 检查是否使用PEFT
        if hasattr(self.model, "is_peft_model") and self.model.is_peft_model:
            # 如果是PEFT模型，不需要创建参考模型
            self.ref_model = None
            logger.info("使用PEFT模型，将通过禁用adapter获得参考模型输出")
        else:
            # 从model_init_kwargs获取torch_dtype
            torch_dtype = None
            if self.args.model_init_kwargs and 'torch_dtype' in self.args.model_init_kwargs:
                torch_dtype_str = self.args.model_init_kwargs['torch_dtype']
                if isinstance(torch_dtype_str, str) and torch_dtype_str not in ['auto', None] and hasattr(torch, torch_dtype_str):
                    torch_dtype = getattr(torch, torch_dtype_str)
                elif torch_dtype_str == 'auto':
                    torch_dtype = 'auto'
            
            # 如果不是PEFT模型，创建一个参考模型副本
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
            )
            # 将参考模型移至适当设备并设为评估模式
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            logger.info("已创建参考模型副本")
    
    def _set_signature_columns_if_needed(self):
        """设置_signature_columns以保留GRPO所需的列"""
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "completion", "advantage", "reward"]
    
    def _remove_unused_columns(self, dataset, description=None):
        """确保GRPO所需的列不被删除"""
        if not self.args.remove_unused_columns:
            return dataset
        
        # 确保GRPO所需的列被保留
        self._set_signature_columns_if_needed()
        
        # 调用父类方法
        return super()._remove_unused_columns(dataset, description)
    
    def _prepare_inputs(self, inputs):
        """处理GRPO特定的输入格式"""
        # 获取设备
        device = self.accelerator.device
        
        # 处理输入格式
        prompts = [x["prompt"] for x in inputs]
        completions = [x["completion"] for x in inputs]
        advantages = torch.tensor([x["advantage"] for x in inputs], dtype=torch.float32, device=device)
        
        # 编码提示
        prompt_inputs = self.processing_class(
            prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # 截断提示到最大长度
        if self.args.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.args.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.args.max_prompt_length:]
        
        # 编码完成
        completion_inputs = self.processing_class(
            completions, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        )
        completion_inputs = super()._prepare_inputs(completion_inputs)
        completion_ids, completion_mask = completion_inputs["input_ids"], completion_inputs["attention_mask"]
        
        # 截断完成到最大长度
        if self.args.max_completion_length is not None:
            completion_ids = completion_ids[:, :self.args.max_completion_length]
            completion_mask = completion_mask[:, :self.args.max_completion_length]
        
        # 连接提示和完成
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # 存储提示和完成的长度信息用于损失计算
        prompt_lengths = prompt_mask.sum(dim=1)
        
        # 获取参考模型的logits
        with torch.no_grad():
            if self.ref_model is not None:
                # 使用单独的参考模型
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            else:
                # 对于PEFT模型，通过禁用adapter获取参考模型的输出
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                with unwrapped_model.disable_adapter():
                    ref_outputs = unwrapped_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
            ref_logits = ref_outputs.logits
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_lengths": prompt_lengths,
            "ref_logits": ref_logits,
            "advantages": advantages,
        }
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """实现GRPO的损失计算"""
        # 获取输入
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_lengths = inputs["prompt_lengths"]
        ref_logits = inputs["ref_logits"]
        advantages = inputs["advantages"]
        
        # 模型前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        
        # 计算GRPO损失
        batch_size = logits.shape[0]
        losses = []
        
        for i in range(batch_size):
            # 获取完成部分的logits (去除最后一个token，因为没有下一个token来预测)
            prompt_length = prompt_lengths[i]
            completion_logits = logits[i, prompt_length-1:-1, :]  # shift left for next token prediction
            completion_ref_logits = ref_logits[i, prompt_length-1:-1, :]
            completion_tokens = input_ids[i, prompt_length:]
            
            # 仅考虑有效的token (忽略padding)
            valid_mask = (completion_tokens != self.processing_class.pad_token_id)
            valid_tokens = completion_tokens[valid_mask]
            valid_logits = completion_logits[valid_mask]
            valid_ref_logits = completion_ref_logits[valid_mask]
            
            if len(valid_tokens) == 0:
                continue  # 跳过没有有效token的样本
            
            # 计算log概率
            log_probs = torch.log_softmax(valid_logits, dim=-1)
            ref_log_probs = torch.log_softmax(valid_ref_logits, dim=-1)
            
            # 获取实际生成的token的log概率
            selected_log_probs = log_probs.gather(-1, valid_tokens.unsqueeze(-1)).squeeze(-1)
            selected_ref_log_probs = ref_log_probs.gather(-1, valid_tokens.unsqueeze(-1)).squeeze(-1)
            
            # 计算KL散度: exp(ref - policy) - 1 - (ref - policy)
            kl_div = torch.exp(selected_ref_log_probs - selected_log_probs) - 1 - (selected_ref_log_probs - selected_log_probs)
            
            # 计算GRPO目标
            advantage = advantages[i]
            
            # GRPO损失计算
            policy_loss = -torch.exp(selected_log_probs - selected_log_probs.detach()) * advantage
            kl_penalty = self.args.beta * kl_div
            
            sample_loss = (policy_loss + kl_penalty).mean()
            losses.append(sample_loss)
        
        # 计算平均损失
        if losses:
            loss = torch.stack(losses).mean()
        else:
            # 如果所有样本都没有有效token，返回零损失
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss