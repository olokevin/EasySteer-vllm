# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Any
import torch

from .template import AlgorithmTemplate
from .factory import register_algorithm
import logging
logger = logging.getLogger(__name__)
@register_algorithm("direct")
class DirectAlgorithm(AlgorithmTemplate):
    """Direct控制向量算法实现"""

    def __init__(self, layer_id: Optional[int] = None, normalize: bool = False, **kwargs):
        super().__init__(layer_id)
        self.normalize = normalize
        self.steer_vectors: dict[int, torch.Tensor | int] = {}
        self.active_vector: Optional[torch.Tensor] = None

    def set_steer_vector(self, index: int, **kwargs) -> None:
        """设置控制向量"""
        payload = kwargs.get("payload")
        scale_factor = kwargs.get("scale_factor", 1.0)
        if payload is None or not isinstance(payload, torch.Tensor):
            raise ValueError("DirectAlgorithm requires 'payload' (torch.Tensor) in kwargs")
        self.steer_vectors[index] = payload * scale_factor

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """从GGUF文件加载Direct控制向量"""
        import gguf
        import numpy as np

        config = kwargs.get("config")
        if config is None:
            raise ValueError("DirectAlgorithm.load_from_path requires 'config' in kwargs")

        reader = gguf.GGUFReader(path)
        
        # 验证文件类型
        archf = reader.get_field("general.architecture")
        if archf and len(archf.parts):
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "steervector" and arch != "controlvector":
                # 仅记录日志，不强制要求
                # logger.warning(".gguf file with arch %s may not be a steer vector", arch)
                pass

        sv_weights = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (ValueError, IndexError) as e:
                raise ValueError(f".gguf file has invalid direction field name: {tensor.name}") from e
            
            np_copy = np.array(tensor.data, copy=True)
            sv_weights[layer] = torch.from_numpy(np_copy).to(device).to(config.adapter_dtype)
            
        return {"layer_payloads": sv_weights}

    def reset_steer_vector(self, index: int) -> None:
        """重置控制向量"""
        if index in self.steer_vectors:
            if isinstance(self.steer_vectors[index], torch.Tensor):
                shape = self.steer_vectors[index].shape
                device = self.steer_vectors[index].device
                dtype = self.steer_vectors[index].dtype
                self.steer_vectors[index] = torch.zeros(shape, device=device, dtype=dtype)
            else:
                self.steer_vectors[index] = None

    def set_active_tensor(self, index: int) -> None:
        """设置激活的控制向量"""
        if index is not None and index in self.steer_vectors:
            if not isinstance(self.steer_vectors[index], torch.Tensor):
                self.active_vector = None
            else:
                self.active_vector = self.steer_vectors[index]
        else:
            self.active_vector = None

    # 实现算法模板要求的抽象方法
    def _get_params(self) -> Optional[torch.Tensor]:
        """获取当前激活的算法参数"""
        return self.active_vector

    def _is_valid(self, params: Any) -> bool:
        """检查算法参数是否有效"""
        return params is not None and isinstance(params, torch.Tensor) and params.numel() > 0

    def _transform(self, hidden_state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """对单个token进行Direct变换: h = h + cv (可选归一化)"""
        if self.normalize:
            norm_pre = torch.norm(hidden_state, dim=-1, keepdim=True)
            transformed = hidden_state + params
            norm_post = torch.norm(transformed, dim=-1, keepdim=True)
            return transformed * norm_pre / norm_post
        else:
            logger.debug("yes!")
            return hidden_state + params

 