# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any, Tuple
import torch
import logging

from .base import BaseSteerVectorAlgorithm
from .factory import register_algorithm

logger = logging.getLogger(__name__)


@register_algorithm("linear")
class LinearTransformAlgorithm(BaseSteerVectorAlgorithm):
    """
    线性变换算法：wh+b
    
    对隐藏状态应用一个线性变换，包括权重矩阵乘法和偏置向量加法。
    使用同一个权重矩阵和偏置向量应用到所有目标层。
    """

    def __init__(self, layer_id=None, normalize=False):
        super().__init__(layer_id)
        self.weights = {}  # 保存权重矩阵W
        self.biases = {}   # 保存偏置向量b
        self.scale_factors = {}  # 保存缩放因子
        self.active_tensor_index = None

    def set_steer_vector(self, index: int, **kwargs):
        """设置权重矩阵和偏置向量"""
        payload = kwargs.get("payload")
        scale_factor = kwargs.get("scale_factor", 1.0)
        
        if not payload or "weight" not in payload:
            logger.warning(f"Missing required 'weight' in payload for layer {self.layer_id}")
            return

        # 保存权重矩阵和偏置向量
        self.weights[index] = payload["weight"]
        self.biases[index] = payload.get("bias", None)  # 偏置可选
        self.scale_factors[index] = scale_factor

    def reset_steer_vector(self, index: int):
        """重置特定索引的向量"""
        if index in self.weights:
            del self.weights[index]
        if index in self.biases:
            del self.biases[index]
        if index in self.scale_factors:
            del self.scale_factors[index]
        if self.active_tensor_index == index:
            self.active_tensor_index = None

    def set_active_tensor(self, index: int):
        """设置当前激活的张量索引"""
        self.active_tensor_index = index
        
    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用线性变换干预 wh+b"""
        if self.active_tensor_index is None or self.active_tensor_index not in self.weights:
            return hidden_states
            
        # 检查是否可以应用控制向量 (对应于触发条件逻辑)
        if not self._should_apply_steer_vector(hidden_states):
            return hidden_states

        # 获取当前激活的权重、偏置和缩放因子
        weight = self.weights[self.active_tensor_index]
        bias = self.biases.get(self.active_tensor_index, None)
        scale = self.scale_factors.get(self.active_tensor_index, 1.0)

        # 确保权重和偏置的数据类型与hidden_states匹配
        if weight.dtype != hidden_states.dtype:
            weight = weight.to(dtype=hidden_states.dtype)
            self.weights[self.active_tensor_index] = weight  # 更新缓存，避免重复转换
            
        if bias is not None and bias.dtype != hidden_states.dtype:
            bias = bias.to(dtype=hidden_states.dtype)
            self.biases[self.active_tensor_index] = bias  # 更新缓存，避免重复转换

        # 检查维度匹配
        if weight.shape[0] != hidden_states.shape[-1]:
            logger.error(f"维度不匹配: weight shape={weight.shape}, hidden_states shape={hidden_states.shape}")
            return hidden_states

        # 应用权重矩阵：矩阵乘法
        # 维度重排以进行批量矩阵乘法
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])  # [batch*seq*..., hidden_dim]
        transformed = torch.matmul(flat_hidden, weight.T)  # [batch*seq*..., hidden_dim]
        
        # 如果有偏置，添加偏置
        if bias is not None:
            transformed = transformed + bias
            
        # 应用缩放因子
        if scale != 1.0:
            transformed = transformed * scale
            
        # 恢复原始维度
        transformed = transformed.reshape(orig_shape)
        
        # 调试日志
        if self.debug:
            logger.info(f"Layer {self.layer_id}: Applied linear transform with scale {scale}")
            
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
        """应用线性变换 wh+b 到隐藏状态"""
        # 没有激活的向量，直接返回原始输出
        if self.active_tensor_index is None or self.active_tensor_index not in self.weights:
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
        """从pkl文件加载线性变换参数"""
        import pickle
        import numpy as np
        import os
        
        try:
            # 加载pkl文件
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # 提取权重和偏置
            # 检查是否为LinearTransport对象，直接访问属性而不是使用get方法
            if hasattr(data, 'A_') and hasattr(data, 'B_'):
                # 直接访问属性
                weight = data.A_
                bias = data.B_
            elif isinstance(data, dict):
                # 如果是字典，使用get方法
                weight = data.get("A_", None)
                bias = data.get("B_", None)
            else:
                # 尝试直接作为属性访问
                try:
                    weight = getattr(data, "A_", None)
                    bias = getattr(data, "B_", None)
                except AttributeError:
                    logger.error(f"Cannot extract A_ and B_ from data type: {type(data)}")
                    raise ValueError(f"Unsupported data format. Neither a dict nor has A_/B_ attributes: {type(data)}")
            
            if weight is None:
                logger.error(f"Failed to find weight (A_) in data of type {type(data)}")
                raise ValueError(f"Weight matrix (A_) not found in pkl file")
                
            # 确保数据是numpy数组或转换为numpy数组
            if not isinstance(weight, np.ndarray):
                weight = np.array(weight, dtype=np.float32)
                
            if bias is not None and not isinstance(bias, np.ndarray):
                bias = np.array(bias, dtype=np.float32)
            
            # 从config获取数据类型，并使用默认值
            adapter_dtype = config.adapter_dtype if hasattr(config, 'adapter_dtype') else torch.float16
                
            # 转换为torch张量并设置正确的dtype
            weight_tensor = torch.tensor(weight, device=device, dtype=adapter_dtype)
            bias_tensor = torch.tensor(bias, device=device, dtype=adapter_dtype) if bias is not None else None
            
            # 创建每个目标层的payload，使用相同的权重和偏置
            layer_payloads = {}
            
            # 如果没有指定目标层，假设应用于所有层
            # 我们这里假设模型有48层，基于错误信息中看到的配置
            if target_layers is None:
                target_layers = list(range(48))  # 假设模型有48层
                
            for layer_idx in target_layers:
                layer_payloads[layer_idx] = {
                    "weight": weight_tensor,
                    "bias": bias_tensor
                }
                
            return {"layer_payloads": layer_payloads}
            
        except Exception as e:
            logger.error(f"Failed to load linear transform parameters from {file_path}: {e}")
            raise RuntimeError(f"Failed to load linear transform parameters") from e 