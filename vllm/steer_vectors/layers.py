# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
from torch import nn

from .algorithms import BaseSteerVectorAlgorithm, create_algorithm

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


@dataclass
class SteerVectorMapping:
    layer_mapping: dict[int, torch.Tensor]


class BaseLayerWithSteerVector(nn.Module):
    pass


def _extract_hidden_states_and_residual(output):
    """
    从DecoderLayer输出中提取hidden_states和residual

    Args:
        output: DecoderLayer的输出，可能的格式：
               - (hidden_states, residual)  # Qwen2等模型
               - hidden_states             # Phi等模型
               - tuple with more elements  # 其他可能格式

    Returns:
        (hidden_states, residual, other_outputs, original_format)
    """
    if isinstance(output, tuple):
        if len(output) == 2:
            # 假设是(hidden_states, residual)格式
            hidden_states, residual = output
            if (isinstance(hidden_states, torch.Tensor) and
                    isinstance(residual, torch.Tensor) and
                    hidden_states.shape == residual.shape):
                return hidden_states, residual, None, "tuple_2"
            else:
                # 如果形状不匹配，可能不是(hidden_states, residual)格式
                return output[0], None, output[1:], "tuple_other"
        elif len(output) > 2:
            # 更复杂的tuple，假设第一个是hidden_states
            return output[0], None, output[1:], "tuple_multi"
        else:
            # 单元素tuple
            return output[0], None, None, "tuple_1"
    elif isinstance(output, torch.Tensor):
        # 直接是tensor，如Phi模型
        return output, None, None, "tensor"
    else:
        # 其他格式，尝试从属性中提取
        if hasattr(output, 'hidden_states'):
            hidden_states = output.hidden_states
            residual = getattr(output, 'residual', None)
            return hidden_states, residual, output, "object"
        else:
            # 无法识别的格式，返回原始输出
            return output, None, None, "unknown"


def _reconstruct_output(modified_hidden_states, residual, other_outputs, original_format, original_output):
    """
    根据原始格式重构输出

    Args:
        modified_hidden_states: 修改后的hidden_states
        residual: 残差（如果有）
        other_outputs: 其他输出元素
        original_format: 原始格式标识
        original_output: 原始输出（用于复杂对象的重构）

    Returns:
        重构后的输出
    """
    if original_format == "tuple_2":
        return (modified_hidden_states, residual)
    elif original_format == "tuple_other":
        return (modified_hidden_states,) + other_outputs
    elif original_format == "tuple_multi":
        return (modified_hidden_states,) + other_outputs
    elif original_format == "tuple_1":
        return (modified_hidden_states,)
    elif original_format == "tensor":
        return modified_hidden_states
    elif original_format == "object":
        # 对于对象格式，修改相应属性
        if hasattr(original_output, 'hidden_states'):
            original_output.hidden_states = modified_hidden_states
        return original_output
    else:
        # 未知格式，返回修改后的hidden_states
        return modified_hidden_states


class DecoderLayerWithSteerVector(BaseLayerWithSteerVector):
    """
    通用的DecoderLayer包装器，支持在完整hidden state上进行干预。
    采用懒加载机制，只在需要时创建对应的算法实例，节省内存。
    """

    def __init__(self, base_layer) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.active_algorithm_name: str = "direct"
        self.algorithms: Dict[str, BaseSteerVectorAlgorithm] = {}
        self.layer_id: Optional[int] = None

    def _get_or_create_algorithm(self, name: str, **kwargs) -> BaseSteerVectorAlgorithm:
        """懒加载或获取指定名称的算法实例"""
        if name not in self.algorithms:
            # 将从外部传入的kwargs（如normalize）传递给构造函数
            self.algorithms[name] = create_algorithm(name, layer_id=self.layer_id, **kwargs)
        return self.algorithms[name]

    def set_layer_id(self, layer_id: int) -> None:
        """为所有已创建的算法设置层ID"""
        self.layer_id = layer_id
        for algo in self.algorithms.values():
            algo.layer_id = layer_id

    def set_steer_vector(self, index: int, **kwargs):
        """
        通用方法：为指定算法设置控制向量参数。
        这个方法现在负责将所有相关参数分发到算法实例。
        """
        # 1. 确定算法并提取其独有关联参数
        algorithm_name = kwargs.pop("algorithm_name", "direct")
        self.active_algorithm_name = algorithm_name
        
        # 提取构造函数参数 (如 normalize)
        init_kwargs = {}
        if "normalize" in kwargs:
            init_kwargs["normalize"] = kwargs.get("normalize")

        algo = self._get_or_create_algorithm(algorithm_name, **init_kwargs)

        # 2. 设置核心向量参数 (payload) 和其他运行时参数
        # 我们将所有剩余的kwargs传递给set_steer_vector
        algo.set_steer_vector(index, **kwargs)

        # 3. 设置触发器和调试标志
        if "prefill_trigger_tokens" in kwargs:
            algo.set_prefill_trigger_tokens(kwargs["prefill_trigger_tokens"])
        if "prefill_trigger_positions" in kwargs:
            algo.set_prefill_trigger_positions(kwargs["prefill_trigger_positions"])
        if "generate_trigger_tokens" in kwargs:
            algo.set_generate_trigger_tokens(kwargs["generate_trigger_tokens"])
        if "debug" in kwargs:
            algo.set_debug(kwargs["debug"])

    def reset_steer_vector(self, index: int):
        """重置所有算法中指定索引的向量（或仅重置当前激活的）"""
        # 简单起见，我们重置所有已创建的算法中的向量
        # 也可以只重置当前激活的
        for algo in self.algorithms.values():
            algo.reset_steer_vector(index)

    def set_active_tensor(self, index: int):
        """为当前激活的算法设置激活张量"""
        algo = self._get_or_create_algorithm(self.active_algorithm_name)
        algo.set_active_tensor(index)

    def _apply_to_active_algorithm(self, method_name: str, *args, **kwargs):
        """辅助函数，将方法调用应用于当前激活的算法"""
        algo = self._get_or_create_algorithm(self.active_algorithm_name)
        method = getattr(algo, method_name, None)
        if method:
            method(*args, **kwargs)

    def set_prefill_trigger_tokens(self, token_ids: Optional[list[int]]):
        self._apply_to_active_algorithm("set_prefill_trigger_tokens", token_ids)

    def set_prefill_trigger_positions(self, positions: Optional[list[int]]):
        self._apply_to_active_algorithm("set_prefill_trigger_positions", positions)

    def set_generate_trigger_tokens(self, token_ids: Optional[list[int]]):
        self._apply_to_active_algorithm("set_generate_trigger_tokens", token_ids)

    def set_debug(self, debug: bool) -> None:
        self._apply_to_active_algorithm("set_debug", debug)

    def forward(self, *args, **kwargs):
        """包装DecoderLayer的forward方法"""
        output = self.base_layer(*args, **kwargs)

        # 动态获取当前激活的算法并应用
        active_algo = self._get_or_create_algorithm(self.active_algorithm_name)
        return active_algo.forward_decoder_layer(output)


