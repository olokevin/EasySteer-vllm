# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any, Tuple
import torch
import logging

from .template import AlgorithmTemplate
from .factory import register_algorithm

logger = logging.getLogger(__name__)


@register_algorithm("linear")
class LinearTransformAlgorithm(AlgorithmTemplate):
    """
    线性变换算法：wh+b
    
    对隐藏状态应用一个线性变换，包括权重矩阵乘法和偏置向量加法。
    使用同一个权重矩阵和偏置向量应用到所有目标层。
    """

    def __init__(self, layer_id=None, normalize=False):
        super().__init__(layer_id)
        # normalize is accepted for signature consistency, but not used.
        self.weights = {}  # 保存权重矩阵W
        self.biases = {}   # 保存偏置向量b
        self.scale_factors = {}  # 保存缩放因子
        self.active_tensor_index = None
        self.active_params: Optional[dict] = None

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
        if index is not None and index in self.weights:
            weight = self.weights[index]
            bias = self.biases.get(index, None)
            scale = self.scale_factors.get(index, 1.0)
            
            self.active_params = {
                "weight": weight,
                "bias": bias,
                "scale": scale
            }
        else:
            self.active_params = None

    # 实现算法模板要求的抽象方法
    def _get_params(self) -> Optional[dict]:
        """获取当前激活的算法参数"""
        return self.active_params

    def _is_valid(self, params: Any) -> bool:
        """检查算法参数是否有效"""
        return (params is not None and 
                isinstance(params, dict) and 
                "weight" in params)

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        """对单个token进行线性变换: wh+b * scale"""
        weight = params["weight"]
        bias = params.get("bias", None)
        scale = params.get("scale", 1.0)
        
        # 确保数据类型匹配
        device = hidden_state.device
        dtype = hidden_state.dtype
        
        weight = weight.to(device).to(dtype)
        if bias is not None:
            bias = bias.to(device).to(dtype)
        
        # 检查维度匹配
        if weight.shape[0] != hidden_state.shape[-1]:
            logger.error(f"维度不匹配: weight shape={weight.shape}, hidden_state shape={hidden_state.shape}")
            return hidden_state
        
        # 应用权重矩阵：矩阵乘法
        transformed = torch.matmul(hidden_state, weight.T)
        
        # 如果有偏置，添加偏置
        if bias is not None:
            transformed = transformed + bias
            
        # 应用缩放因子
        if scale != 1.0:
            transformed = transformed * scale
            
        return transformed

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