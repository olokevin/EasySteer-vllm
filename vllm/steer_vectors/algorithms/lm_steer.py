# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any, Tuple
import torch
import logging

from .base import BaseSteerVectorAlgorithm
from .factory import register_algorithm

logger = logging.getLogger(__name__)


@register_algorithm("lm_steer")
class LMSteerAlgorithm(BaseSteerVectorAlgorithm):
    """
    LM-Steer算法
    
    使用低秩优化形式：h = h + α*((h·P1)·P2^T)
    其中P1和P2是低秩投影矩阵，·表示矩阵乘法。
    """

    def __init__(self, layer_id=None, normalize=False):
        super().__init__(layer_id)
        self.projector1 = {}  # 保存投影矩阵P1
        self.projector2 = {}  # 保存投影矩阵P2
        self.scale_factors = {}  # 保存缩放因子α
        self.active_tensor_index = None

    def set_steer_vector(self, index: int, **kwargs):
        """设置投影矩阵和缩放因子"""
        payload = kwargs.get("payload")
        scale_factor = kwargs.get("scale_factor", 1.0)
        
        if not payload:
            logger.warning(f"Missing payload for layer {self.layer_id}")
            return
            
        # 检查是否包含必需的projector矩阵
        if "projector1" in payload and "projector2" in payload:
            self.projector1[index] = payload["projector1"]
            self.projector2[index] = payload["projector2"]
            logger.info(f"Set projector matrices for index {index}")
        else:
            logger.warning(f"Missing required 'projector1'/'projector2' in payload for layer {self.layer_id}")
            return

        self.scale_factors[index] = scale_factor

    def reset_steer_vector(self, index: int):
        """重置特定索引的向量"""
        if index in self.projector1:
            del self.projector1[index]
        if index in self.projector2:
            del self.projector2[index]
        if index in self.scale_factors:
            del self.scale_factors[index]
        if self.active_tensor_index == index:
            self.active_tensor_index = None

    def set_active_tensor(self, index: int):
        """设置当前激活的张量索引"""
        self.active_tensor_index = index
        
    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用干预公式：h = h + α*((h·P1)·P2^T)"""
        if self.active_tensor_index is None:
            return hidden_states
            
        # 检查是否可以应用控制向量
        if not self._should_apply_steer_vector(hidden_states):
            return hidden_states

        # 获取当前激活的索引
        index = self.active_tensor_index
        alpha = self.scale_factors.get(index, 1.0)
        
        if index not in self.projector1 or index not in self.projector2:
            logger.warning(f"Projector matrices not found for index {index}")
            return hidden_states
            
        # 获取投影矩阵
        P1 = self.projector1[index]  # shape: [num_steers, embed_dim, rank]
        P2 = self.projector2[index]  # shape: [num_steers, embed_dim, rank]
        
        # 确保数据类型匹配
        if P1.dtype != hidden_states.dtype:
            P1 = P1.to(dtype=hidden_states.dtype)
            self.projector1[index] = P1
            
        if P2.dtype != hidden_states.dtype:
            P2 = P2.to(dtype=hidden_states.dtype)
            self.projector2[index] = P2
        
        # 选择第一个steer向量(索引0)
        P1_active = P1[0]  # shape: [embed_dim, rank]
        P2_active = P2[0]  # shape: [embed_dim, rank]
        
        # 保存原始维度
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])  # [batch*seq*..., hidden_dim]
        
        # 应用低秩变换: (h·P1)·P2^T
        transformed = torch.matmul(flat_hidden, P1_active)  # [batch*seq, rank]
        transformed = torch.matmul(transformed, P2_active.transpose(0, 1))  # [batch*seq, hidden_dim]
        
        # 应用缩放因子并添加原始隐藏状态：h = h + α*((h·P1)·P2^T)
        transformed = flat_hidden + alpha * transformed
        
        # 恢复原始维度
        transformed = transformed.reshape(orig_shape)
        
        # 调试日志
        if self.debug:
            logger.info(f"Layer {self.layer_id}: Applied LM-Steer transform with scale {alpha}")
            
        return transformed

    def _should_apply_steer_vector(self, hidden_states: torch.Tensor) -> bool:
        """检查当前token是否应该应用控制向量"""
        # 没有配置任何触发器时，对所有token都应用
        if (not self._has_prefill_triggers() and 
            self.generate_trigger_tokens is None):
            return True
            
        # 尝试从context获取当前token和位置信息
        try:
            from vllm.forward_context import get_forward_context
            if get_forward_context is None:
                # 如果无法获取context，默认应用控制向量
                return True
                
            ctx = get_forward_context()
            if ctx is None:
                return True
                
            # 检查是否在prefill或decode阶段
            # ForwardContext对象有is_decode属性，我们用它来区分阶段
            if not hasattr(ctx, 'is_decode') or not ctx.is_decode:
                # prefill阶段
                # 检查是否对所有prefill token应用
                if self._should_apply_to_all_prefill_tokens():
                    return True
                    
                # 检查token id触发器
                if (self.prefill_trigger_tokens is not None and 
                    hasattr(ctx, 'token_id') and
                    ctx.token_id in self.prefill_trigger_tokens):
                    return True
                    
                # 检查位置触发器
                if self.prefill_trigger_positions is not None:
                    # 获取seq_len和当前位置
                    if not hasattr(ctx, 'seq_start_loc'):
                        # 没有位置信息，保险起见应用控制向量
                        return True
                    
                    if not hasattr(ctx, 'position'):
                        return True
                        
                    # 解析相对位置为绝对位置
                    pos = ctx.position
                    trigger_positions = self._resolve_positions_per_sample(
                        self.prefill_trigger_positions, ctx.seq_start_loc)
                        
                    if pos in trigger_positions:
                        return True
                return False
            else:
                # decode/generate阶段
                # 检查是否对所有generate token应用
                if self._should_apply_to_all_generate_tokens():
                    return True
                    
                # 检查token id触发器
                if (self.generate_trigger_tokens is not None and 
                    hasattr(ctx, 'token_id') and
                    ctx.token_id in self.generate_trigger_tokens):
                    return True
                    
                return False
        except Exception as e:
            # 出现异常时保险起见应用控制向量
            if self.debug:
                logger.warning(f"Error in _should_apply_steer_vector: {e}")
            return True

    def forward_decoder_layer(self, output):
        """应用LM-Steer变换到隐藏状态"""
        # 没有激活的向量，直接返回原始输出
        if self.active_tensor_index is None:
            return output

        # 从DecoderLayer输出中提取hidden_states
        from vllm.steer_vectors.layers import _extract_hidden_states_and_residual, _reconstruct_output
        hidden_states, residual, other_outputs, original_format = _extract_hidden_states_and_residual(output)

        # 应用干预
        transformed = self.apply_intervention(hidden_states)
        
        # 如果没有变化，直接返回原始输出
        if transformed is hidden_states:
            return output
            
        # 重构并返回修改后的输出
        return _reconstruct_output(transformed, residual, other_outputs, original_format, output)
        
    def forward_mlp_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP层的forward处理，简单转发到apply_intervention"""
        return self.apply_intervention(hidden_states)

    @classmethod
    def load_from_path(cls, file_path: str, device: str, config=None, target_layers=None):
        """从pt文件加载LM-Steer参数"""
        import os
        
        try:
            # 加载pt文件，设置weights_only=False以允许加载argparse.Namespace等对象
            state_dict = torch.load(file_path, map_location=device, weights_only=False)
            
            # 提取投影矩阵
            projector1 = None
            projector2 = None
            
            # 检查是否为列表结构 (处理gpt2.pt特殊结构)
            if isinstance(state_dict, list) and len(state_dict) > 1:
                # logger.info(f"检测到列表结构，正在尝试从元素[1]中提取参数")
                params_dict = state_dict[1]
                
                if isinstance(params_dict, dict) and 'projector1' in params_dict and 'projector2' in params_dict:
                    # 这是低秩优化的形式
                    # logger.info(f"找到低秩优化的projector1和projector2参数")
                    projector1 = params_dict['projector1']
                    projector2 = params_dict['projector2']
            # 检查是否为字典结构
            elif isinstance(state_dict, dict):
                if "projector1" in state_dict and "projector2" in state_dict:
                    # 这是低秩优化的形式
                    # logger.info(f"找到低秩优化的projector1和projector2参数")
                    projector1 = state_dict["projector1"]
                    projector2 = state_dict["projector2"]
            
            # 如果没有找到投影矩阵，报错
            if projector1 is None or projector2 is None:
                logger.error(f"Could not find projector matrices in file {file_path}")
                raise ValueError(f"Projector matrices not found in pt file")
            
            # 从config获取数据类型，并使用默认值
            adapter_dtype = config.adapter_dtype if hasattr(config, 'adapter_dtype') else torch.float16
                
            # 创建每个目标层的payload
            layer_payloads = {}
            
            # 如果没有指定目标层，假设应用于所有层
            if target_layers is None:
                # 尝试从config中获取层数
                if hasattr(config, 'num_hidden_layers'):
                    target_layers = list(range(config.num_hidden_layers))
                else:
                    # 默认假设模型有32层
                    target_layers = list(range(32))
            
            # 确保是tensor并转换数据类型
            projector1_tensor = projector1.to(device=device, dtype=adapter_dtype)
            projector2_tensor = projector2.to(device=device, dtype=adapter_dtype)
            
            for layer_idx in target_layers:
                layer_payloads[layer_idx] = {
                    "projector1": projector1_tensor,
                    "projector2": projector2_tensor
                }
            # logger.info(f"已加载低秩投影矩阵 P1: {projector1_tensor.shape}, P2: {projector2_tensor.shape}")
                
            return {"layer_payloads": layer_payloads}
            
        except Exception as e:
            logger.error(f"Failed to load LM-Steer parameters from {file_path}: {e}")
            raise RuntimeError(f"Failed to load LM-Steer parameters") from e 