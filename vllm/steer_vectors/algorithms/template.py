# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any
import torch
from abc import ABC, abstractmethod

from .base import BaseSteerVectorAlgorithm

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


class AlgorithmTemplate(BaseSteerVectorAlgorithm, ABC):
    """
    控制向量算法模板类
    
    为基于token级变换的算法提供统一模板，包括：
    - 触发器处理逻辑
    - 位置解析逻辑  
    - Prefill/Generate阶段处理逻辑
    - 前向传播处理逻辑
    
    算法实现者只需要实现3个核心方法：
    - _get_params(): 获取当前激活的算法参数
    - _is_valid(): 检查参数是否有效
    - _transform(): 对单个token进行变换
    """
    
    def __init__(self, layer_id: Optional[int] = None):
        super().__init__(layer_id)
    
    @abstractmethod
    def _get_params(self) -> Any:
        """获取当前激活的算法参数，由具体算法实现"""
        pass
    
    @abstractmethod
    def _is_valid(self, params: Any) -> bool:
        """检查算法参数是否有效，由具体算法实现"""
        pass
    
    @abstractmethod
    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """对单个token的hidden state进行变换，由具体算法实现"""
        pass
    
    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """统一的干预应用逻辑，所有使用此模板的算法共享"""
        # Skip if no trigger tokens or positions are configured
        if (self.prefill_trigger_tokens is None and self.generate_trigger_tokens is None and
                self.prefill_trigger_positions is None):
            return hidden_states

        # Get algorithm parameters
        params = self._get_params()
        if not self._is_valid(params):
            return hidden_states

        # 获取forward context信息
        if get_forward_context is None:
            return hidden_states

        forward_ctx = get_forward_context()
        current_tokens = forward_ctx.current_tokens
        is_decode = forward_ctx.is_decode
        attn_metadata = forward_ctx.attn_metadata

        if current_tokens is not None and attn_metadata is not None:
            if is_decode:
                # Generate/Decode阶段处理
                return self._apply_generate_phase(hidden_states, current_tokens, params)
            else:
                # Prefill阶段处理
                return self._apply_prefill_phase(hidden_states, current_tokens, attn_metadata, params)

        return hidden_states
    
    def _apply_generate_phase(self, hidden_states: torch.Tensor, current_tokens: torch.Tensor, params: Any) -> torch.Tensor:
        """Generate阶段的统一处理逻辑"""
        if self.generate_trigger_tokens is None:
            return hidden_states

        batch_size = hidden_states.shape[0]

        if current_tokens.numel() > 0:
            if current_tokens.dim() == 2:
                current_tokens = current_tokens.flatten()

            if self._should_apply_to_all_generate_tokens():
                if self.debug:
                    print(f"[{self.__class__.__name__}] Layer {self.layer_id}: "
                          f"Applying transformation to ALL {batch_size} samples in generate phase")
                # 对所有样本应用变换
                for sample_idx in range(batch_size):
                    hidden_states[sample_idx] = self._transform(hidden_states[sample_idx], params)
            else:
                # 检查每个样本的当前token
                samples_to_apply = set()
                for i in range(min(batch_size, current_tokens.shape[0])):
                    token_id = current_tokens[i].item()
                    if token_id in self.generate_trigger_tokens:
                        samples_to_apply.add(i)
                        if self.debug:
                            print(f"[{self.__class__.__name__}] Layer {self.layer_id}: "
                                  f"Sample {i} triggered by token {token_id}")

                if samples_to_apply:
                    if self.debug:
                        print(f"[{self.__class__.__name__}] Applying transformation to generate samples: {samples_to_apply}")
                    for sample_idx in samples_to_apply:
                        if sample_idx < hidden_states.shape[0]:
                            hidden_states[sample_idx] = self._transform(hidden_states[sample_idx], params)

        return hidden_states
    
    def _apply_prefill_phase(self, hidden_states: torch.Tensor, current_tokens: torch.Tensor, attn_metadata: Any, params: Any) -> torch.Tensor:
        """Prefill阶段的统一处理逻辑"""
        if not self._has_prefill_triggers():
            return hidden_states

        total_tokens = hidden_states.shape[0]

        if self._should_apply_to_all_prefill_tokens():
            if self.debug:
                print(f"[{self.__class__.__name__}] Layer {self.layer_id}: "
                      f"Applying transformation to ALL {total_tokens} tokens in prefill phase")
            for pos in range(total_tokens):
                hidden_states[pos] = self._transform(hidden_states[pos], params)
        else:
            # 收集所有需要应用变换的位置
            positions_to_apply = []

            # 获取seq_start_loc来分割不同样本的token（位置和token触发都需要这个信息）
            seq_start_loc = None
            if hasattr(attn_metadata, 'seq_start_loc') and attn_metadata.seq_start_loc is not None:
                seq_start_loc = attn_metadata.seq_start_loc
            elif hasattr(attn_metadata, 'prefill_metadata') and attn_metadata.prefill_metadata is not None:
                prefill_meta = attn_metadata.prefill_metadata
                if hasattr(prefill_meta, 'seq_start_loc'):
                    seq_start_loc = prefill_meta.seq_start_loc

            # 1. 从直接指定的位置添加（修复多prompt的bug）
            if self.prefill_trigger_positions is not None:
                if seq_start_loc is not None:
                    # 使用新的方法：为每个样本分别解析位置
                    resolved_positions = self._resolve_positions_per_sample(self.prefill_trigger_positions, seq_start_loc)
                    positions_to_apply.extend(resolved_positions)
                    if self.debug:
                        print(f"[{self.__class__.__name__}] Direct positions per sample: {self.prefill_trigger_positions} -> {resolved_positions}")
                else:
                    # 后备方案：使用旧的方法（仅适用于单prompt场景）
                    resolved_positions = self._resolve_positions(self.prefill_trigger_positions, total_tokens)
                    positions_to_apply.extend(resolved_positions)
                    if self.debug:
                        print(f"[{self.__class__.__name__}] Direct positions (fallback): {self.prefill_trigger_positions} -> {resolved_positions}")

            # 2. 从token触发添加位置
            if self.prefill_trigger_tokens is not None:
                if seq_start_loc is not None and current_tokens.numel() > 0:
                    num_samples = len(seq_start_loc) - 1
                    if self.debug:
                        print(f"[{self.__class__.__name__}] Layer {self.layer_id}: "
                              f"Prefill - Processing {total_tokens} tokens from {num_samples} samples")

                    # 为每个样本检查是否包含trigger token
                    for sample_idx in range(num_samples):
                        start_pos = seq_start_loc[sample_idx].item()
                        end_pos = seq_start_loc[sample_idx + 1].item()

                        if current_tokens.dim() == 1 and current_tokens.shape[0] >= end_pos:
                            sample_token_ids = current_tokens[start_pos:end_pos]

                            triggered_positions = []
                            for rel_pos, token_id in enumerate(sample_token_ids):
                                if token_id.item() in self.prefill_trigger_tokens:
                                    abs_pos = start_pos + rel_pos
                                    triggered_positions.append(abs_pos)
                                    positions_to_apply.append(abs_pos)

                            if self.debug and triggered_positions:
                                trigger_token_ids = [current_tokens[pos].item() for pos in triggered_positions]
                                print(f"[{self.__class__.__name__}] Sample {sample_idx}: positions {triggered_positions} "
                                      f"triggered by tokens {trigger_token_ids}")

            # 去重并排序
            positions_to_apply = sorted(set(positions_to_apply))

            if positions_to_apply:
                if self.debug:
                    print(f"[{self.__class__.__name__}] Applying transformation to {len(positions_to_apply)} token positions: {positions_to_apply}")
                for pos in positions_to_apply:
                    if pos < hidden_states.shape[0]:
                        hidden_states[pos] = self._transform(hidden_states[pos], params)

        return hidden_states

    def forward_decoder_layer(self, output: Any) -> Any:
        """Decoder层的forward处理"""
        from ..layers import _extract_hidden_states_and_residual, _reconstruct_output

        # 提取hidden_states和residual
        hidden_states, residual, other_outputs, original_format = _extract_hidden_states_and_residual(output)

        # 构建完整的hidden state
        if residual is not None:
            complete_hidden_states = hidden_states + residual
        else:
            complete_hidden_states = hidden_states

        # 应用算法变换
        modified_complete_hidden_states = self.apply_intervention(complete_hidden_states)

        # 重构输出格式
        if residual is not None:
            zero_residual = torch.zeros_like(residual)
            return _reconstruct_output(modified_complete_hidden_states, zero_residual, other_outputs,
                                       original_format, output)
        else:
            return _reconstruct_output(modified_complete_hidden_states, None, other_outputs, original_format,
                                       output)

    def forward_mlp_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP层的forward处理"""
        return self.apply_intervention(hidden_states) 