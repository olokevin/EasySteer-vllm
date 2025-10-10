# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any, Tuple
import torch
import logging

from .template import AlgorithmTemplate
from .factory import register_algorithm

logger = logging.getLogger(__name__)


@register_algorithm("lm_steer")
class LMSteerAlgorithm(AlgorithmTemplate):
    """
    LM-Steer算法
    
    使用低秩优化形式：h = h + α*((h·P1)·P2^T)
    其中P1和P2是低秩投影矩阵，·表示矩阵乘法。
    """

    def __init__(self, layer_id=None, normalize=False):
        super().__init__(layer_id)
        # normalize is accepted for signature consistency, but not used.
        self.projector1 = {}  # 保存投影矩阵P1
        self.projector2 = {}  # 保存投影矩阵P2
        self.scale_factors = {}  # 保存缩放因子α
        self.active_tensor_index = None
        self.active_params: Optional[dict] = None

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
            # logger.info(f"Set projector matrices for index {index}")
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
        if index is not None and index in self.projector1 and index in self.projector2:
            P1 = self.projector1[index]
            P2 = self.projector2[index]
            alpha = self.scale_factors.get(index, 1.0)
            
            # 选择第一个steer向量(索引0) 如果是多维的
            if P1.dim() > 2:
                P1_active = P1[0]  # shape: [embed_dim, rank]
            else:
                P1_active = P1
                
            if P2.dim() > 2:
                P2_active = P2[0]  # shape: [embed_dim, rank]
            else:
                P2_active = P2
            
            self.active_params = {
                "P1": P1_active,
                "P2": P2_active,
                "alpha": alpha
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
                "P1" in params and 
                "P2" in params)

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        """对单个token进行LM-Steer变换: h = h + α*((h·P1)·P2^T)"""
        P1 = params["P1"]
        P2 = params["P2"]
        alpha = params.get("alpha", 1.0)
        
        # 确保数据类型匹配
        device = hidden_state.device
        dtype = hidden_state.dtype
        
        P1 = P1.to(device).to(dtype)
        P2 = P2.to(device).to(dtype)
        
        # 应用低秩变换: (h·P1)·P2^T
        transformed = torch.matmul(hidden_state, P1)  # [..., rank]
        transformed = torch.matmul(transformed, P2.transpose(-2, -1))  # [..., hidden_dim]
        
        # 添加原始隐藏状态：h = h + α*((h·P1)·P2^T)
        return hidden_state + alpha * transformed

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