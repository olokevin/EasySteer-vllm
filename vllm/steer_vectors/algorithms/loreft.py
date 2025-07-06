# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Any
import torch

from .template import AlgorithmTemplate
from .factory import register_algorithm


@register_algorithm("loreft")
class LoReFTAlgorithm(AlgorithmTemplate):
    """LoReFT控制向量算法实现"""
    
    def __init__(self, layer_id: Optional[int] = None, normalize: bool = False, **kwargs):
        super().__init__(layer_id)
        # normalize is accepted for signature consistency, but not used.
        self.loreft_params: dict[int, dict] = {}
        self.active_loreft_params: Optional[dict] = None

    def set_steer_vector(self, index: int, **kwargs) -> None:
        """设置LoReFT参数"""
        payload = kwargs.get("payload")
        scale_factor = kwargs.get("scale_factor", 1.0)
        
        if payload is None or not isinstance(payload, dict):
            raise ValueError("LoReFTAlgorithm requires 'payload' (dict) in kwargs")
            
        self.loreft_params[index] = {
            "rotate_layer": payload.get("rotate_layer"),
            "learned_source_weight": payload.get("learned_source.weight"),
            "learned_source_bias": payload.get("learned_source.bias"),
            "scale_factor": scale_factor
        }

    def reset_steer_vector(self, index: int) -> None:
        """重置LoReFT参数"""
        if index in self.loreft_params:
            del self.loreft_params[index]

    def set_active_tensor(self, index: int) -> None:
        """设置激活的LoReFT参数"""
        if index is not None and index in self.loreft_params:
            self.active_loreft_params = self.loreft_params[index]
        else:
            self.active_loreft_params = None

    # 实现算法模板要求的抽象方法
    def _get_params(self) -> Optional[dict]:
        """获取当前激活的算法参数"""
        return self.active_loreft_params

    def _is_valid(self, params: Any) -> bool:
        """检查算法参数是否有效"""
        return params is not None

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        """对单个token进行LoReFT变换: h + R^T(Wh + b - Rh) * scale_factor"""
        return self._apply_loreft_transformation(hidden_state, params)

    def _apply_loreft_transformation(self, hidden_states: torch.Tensor, loreft_params: dict) -> torch.Tensor:
        """
        应用LoReFT变换: LoReFT(h) = h + R^T(Wh + b − Rh)
        
        Args:
            hidden_states: 输入的hidden states (单个样本或单个token)
            loreft_params: 包含rotate_layer, learned_source参数和scale_factor的字典
            
        Returns:
            变换后的hidden states
        """
        rotate_layer = loreft_params["rotate_layer"]
        learned_source_weight = loreft_params["learned_source_weight"]
        learned_source_bias = loreft_params["learned_source_bias"]
        scale_factor = loreft_params.get("scale_factor", 1.0)
        
        # 确保张量在相同的设备和数据类型上
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        rotate_layer = rotate_layer.to(device).to(dtype)
        learned_source_weight = learned_source_weight.to(device).to(dtype)
        if learned_source_bias is not None:
            learned_source_bias = learned_source_bias.to(device).to(dtype)
        
        # 计算旋转基: Rh
        rotated_base = torch.matmul(hidden_states, rotate_layer)
        
        # 计算学习的源: Wh + b
        learned_output = torch.matmul(hidden_states, learned_source_weight.T)
        if learned_source_bias is not None:
            learned_output = learned_output + learned_source_bias
            
        # 应用LoReFT公式和缩放因子
        delta = torch.matmul((learned_output - rotated_base), rotate_layer.T) * scale_factor
        
        return hidden_states + delta 

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """从目录加载LoReFT参数"""
        import os
        import glob
        import json

        config = kwargs.get("config")
        if config is None:
            raise ValueError("LoReFTAlgorithm.load_from_path requires 'config' in kwargs")
        
        target_layers = kwargs.get("target_layers")

        if not os.path.isdir(path):
            raise ValueError(f"For LoReFT algorithm, path must be a directory. Got: {path}")

        bin_files = glob.glob(os.path.join(path, "*.bin"))
        if not bin_files:
            raise ValueError(f"No .bin files found in directory: {path}")
        if len(bin_files) > 1:
            raise ValueError(f"Multiple .bin files found in directory {path}. Please ensure only one exists.")
        
        bin_file_path = bin_files[0]

        config_files = [os.path.join(path, f) for f in ["reft_config.json", "config.json"] if os.path.exists(os.path.join(path, f))]
        if not config_files:
            raise ValueError(f"No config file (reft_config.json or config.json) found in directory: {path}")
        if len(config_files) > 1:
            raise ValueError(f"Multiple config files found in directory {path}. Please ensure only one exists.")
        
        config_file_path = config_files[0]

        with open(config_file_path, 'r') as f:
            config_data = json.load(f)

        config_layer_idx = None
        if "representations" in config_data:
            representations = config_data.get("representations", [])
            if representations:
                first_repr = representations[0]
                if isinstance(first_repr, dict):
                    config_layer_idx = first_repr.get("layer")

        if config_layer_idx is None:
            bin_filename = os.path.basename(bin_file_path)
            if "intkey_layer_" in bin_filename:
                try:
                    layer_str = bin_filename.split("intkey_layer_")[1].split("_")[0]
                    config_layer_idx = int(layer_str)
                except (ValueError, IndexError):
                    pass
        
        if config_layer_idx is None:
            raise ValueError(f"Could not extract layer info from config {config_file_path} or filename {bin_filename}")

        if target_layers and config_layer_idx not in target_layers:
            raise ValueError(f"Layer mismatch: config specifies layer {config_layer_idx}, but target_layers is {target_layers}.")

        state_dict = torch.load(bin_file_path, map_location=device)
        
        rotate_layer, learned_source_weight, learned_source_bias = None, None, None
        adapter_dtype = config.adapter_dtype if hasattr(config, 'adapter_dtype') else torch.float16

        for key, value in state_dict.items():
            if "rotate_layer" in key:
                if "parametrizations.weight.original" in key or key.endswith("rotate_layer"):
                    rotate_layer = value.to(adapter_dtype)
            elif "learned_source" in key:
                if key.endswith("weight") and "parametrizations" not in key:
                    learned_source_weight = value.to(adapter_dtype)
                elif key.endswith("bias"):
                    learned_source_bias = value.to(adapter_dtype)
            elif key == "weight":
                learned_source_weight = value.to(adapter_dtype)
            elif key == "bias":
                learned_source_bias = value.to(adapter_dtype)
        
        if rotate_layer is None or learned_source_weight is None:
            raise ValueError(f"Could not find all required LoReFT params in {bin_file_path}. Keys: {list(state_dict.keys())}")
        
        loreft_params = {
            config_layer_idx: {
                "rotate_layer": rotate_layer,
                "learned_source.weight": learned_source_weight,
                "learned_source.bias": learned_source_bias
            }
        }
        
        return {"layer_payloads": loreft_params} 