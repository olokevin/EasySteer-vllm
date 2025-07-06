# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Dict
import torch


class BaseSteerVectorAlgorithm(ABC):
    """控制向量算法的基础接口类"""

    def __init__(self, layer_id: Optional[int] = None):
        self.layer_id = layer_id
        self.debug = False
        self.prefill_trigger_tokens: Optional[set[int]] = None
        self.prefill_trigger_positions: Optional[list[int]] = None
        self.generate_trigger_tokens: Optional[set[int]] = None

    def set_debug(self, debug: bool) -> None:
        """设置调试模式"""
        self.debug = debug

    def set_prefill_trigger_tokens(self, token_ids: Optional[list[int]]) -> None:
        """设置prefill阶段的触发token"""
        self.prefill_trigger_tokens = set(token_ids) if token_ids is not None else None

    def set_prefill_trigger_positions(self, positions: Optional[list[int]]) -> None:
        """设置prefill阶段的触发位置"""
        self.prefill_trigger_positions = positions

    def set_generate_trigger_tokens(self, token_ids: Optional[list[int]]) -> None:
        """设置generate阶段的触发token"""
        self.generate_trigger_tokens = set(token_ids) if token_ids is not None else None

    def _should_apply_to_all_prefill_tokens(self) -> bool:
        """检查是否应该对所有prefill token应用控制向量"""
        return self.prefill_trigger_tokens is not None and -1 in self.prefill_trigger_tokens

    def _should_apply_to_all_generate_tokens(self) -> bool:
        """检查是否应该对所有generate token应用控制向量"""
        return self.generate_trigger_tokens is not None and -1 in self.generate_trigger_tokens

    def _has_prefill_triggers(self) -> bool:
        """检查是否配置了prefill触发器"""
        return (self.prefill_trigger_tokens is not None or
                self.prefill_trigger_positions is not None)

    def _resolve_positions(self, positions: list[int], total_length: int) -> list[int]:
        """解析负数位置索引为正数索引"""
        resolved = []
        for pos in positions:
            if pos < 0:
                resolved_pos = total_length + pos
                if resolved_pos >= 0:
                    resolved.append(resolved_pos)
            else:
                if pos < total_length:
                    resolved.append(pos)
        return resolved

    def _resolve_positions_per_sample(self, positions: list[int], seq_start_loc: torch.Tensor) -> list[int]:
        """
        为每个样本分别解析位置，修复多prompt拼接时的位置计算bug
        
        Args:
            positions: 相对于每个样本的位置列表（支持负数索引）
            seq_start_loc: 每个样本在拼接序列中的起始位置
            
        Returns:
            解析后的绝对位置列表
        """
        resolved_positions = []
        num_samples = len(seq_start_loc) - 1
        
        for sample_idx in range(num_samples):
            start_pos = seq_start_loc[sample_idx].item()
            end_pos = seq_start_loc[sample_idx + 1].item()
            sample_length = end_pos - start_pos
            
            # 为这个样本解析每个位置
            for pos in positions:
                if pos < 0:
                    # 负数索引：相对于样本末尾
                    resolved_pos = start_pos + sample_length + pos
                    if resolved_pos >= start_pos:  # 确保位置在样本范围内
                        resolved_positions.append(resolved_pos)
                else:
                    # 正数索引：相对于样本开头
                    resolved_pos = start_pos + pos
                    if resolved_pos < end_pos:  # 确保位置在样本范围内
                        resolved_positions.append(resolved_pos)
            
            if self.debug:
                sample_resolved = []
                for pos in positions:
                    if pos < 0:
                        resolved_pos = start_pos + sample_length + pos
                        if resolved_pos >= start_pos:
                            sample_resolved.append(resolved_pos)
                    else:
                        resolved_pos = start_pos + pos
                        if resolved_pos < end_pos:
                            sample_resolved.append(resolved_pos)
                if sample_resolved:
                    print(f"[Position Resolution] Sample {sample_idx} (length {sample_length}): "
                          f"positions {positions} -> absolute positions {sample_resolved}")
        
        return sorted(resolved_positions)

    @classmethod
    @abstractmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """从文件路径加载控制向量数据，并返回一个包含参数的字典"""
        pass

    @abstractmethod
    def set_steer_vector(self, index: int, **kwargs) -> None:
        """设置控制向量参数"""
        pass

    @abstractmethod
    def reset_steer_vector(self, index: int) -> None:
        """重置控制向量"""
        pass

    @abstractmethod
    def set_active_tensor(self, index: int) -> None:
        """设置激活的控制向量"""
        pass

    @abstractmethod
    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用控制向量干预"""
        pass

    @abstractmethod
    def forward_decoder_layer(self, output: Any) -> Any:
        """Decoder层的forward处理"""
        pass

    @abstractmethod
    def forward_mlp_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP层的forward处理"""
        pass
