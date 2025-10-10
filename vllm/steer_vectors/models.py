# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple
import os

import gguf
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from vllm.adapter_commons.models import (AdapterLRUCache, AdapterModel,
                                         AdapterModelManager)
from vllm.adapter_commons.utils import (add_adapter, deactivate_adapter,
                                        get_adapter, list_adapters,
                                        remove_adapter, set_adapter_mapping)
from vllm.config import SteerVectorConfig
from vllm.steer_vectors.layers import (SteerVectorMapping,
                                         DecoderLayerWithSteerVector)
from vllm.sequence import Sequence
from vllm.steer_vectors.algorithms import create_algorithm
from vllm.steer_vectors.algorithms.factory import ALGORITHM_REGISTRY

logger = logging.getLogger(__name__)

_GLOBAL_STEER_VECTOR_ID = 0


def get_steer_vector_id():
    global _GLOBAL_STEER_VECTOR_ID
    _GLOBAL_STEER_VECTOR_ID += 1
    return _GLOBAL_STEER_VECTOR_ID


# 支持的控制向量包装类，仅支持DecoderLayer级别
_all_sv_classes = {
    "decoder_layer": DecoderLayerWithSteerVector
}

# DecoderLayer类名列表，用于自动识别
_decoder_layer_class_names = [
    "Qwen2DecoderLayer", "LlamaDecoderLayer", "PhiLayer", "PhiDecoderLayer",
    "MistralDecoderLayer", "Qwen2MoeDecoderLayer", "DecoderLayer",
    "BartDecoderLayer", "CohereDecoderLayer", "FalconDecoderLayer",
    "ExaoneDecoderLayer", "GemmaDecoderLayer", "Gemma2DecoderLayer",
    "Gemma3DecoderLayer", "GraniteMoeDecoderLayer", "GraniteMoeSharedDecoderLayer",
    "Grok1DecoderLayer", "GraniteDecoderLayer", "InternLMDecoderLayer",
    "InternLM2VEDecoderLayer", "JambaMambaDecoderLayer", "JambaAttentionDecoderLayer",
    "Mamba2DecoderLayer", "MambaDecoderLayer", "MiniCPMDecoderLayer",
    "MiniCPM3DecoderLayer", "MiniMaxText01DecoderLayer", "MixtralDecoderLayer",
    "MllamaCrossAttentionDecoderLayer", "DeepseekV2DecoderLayer",
    "MolmoDecoderLayer", "MolmoDecoderNormAfterLayer", "NemotronDecoderLayer",
    "DeciLMDecoderLayer", "DeepseekDecoderLayer", "OlmoDecoderLayer",
    "OPTDecoderLayer", "Olmo2DecoderLayer", "OlmoeDecoderLayer",
    "OrionDecoderLayer", "PersimmonDecoderLayer", "Phi3SmallDecoderLayer",
    "PhiMoEDecoderLayer", "SolarDecoderLayer", "Starcoder2DecoderLayer",
    "StablelmDecoderLayer", "WhisperDecoderLayer", "Zamba2AttentionDecoderLayer",
    "Zamba2MambaDecoderLayer", "ChameleonDecoderLayer", "ChameleonSwinDecoderLayer",
    "BambaMixerDecoderLayer", "BambaAttentionDecoderLayer", "BaiChuanDecoderLayer",
    "ArcticDecoderLayer", "AriaTextDecoderLayer", "GPT2Block"
]


def parse_number_from_string(s: str) -> int:
    parts = s.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def load_steer_vector_file(file_path, revision="main"):
    try:
        if Path(file_path).exists():
            return str(Path(file_path).resolve())
        parts = file_path.split("/")
        repo_id = "/".join(parts[:2])
        file_name = "/".join(parts[2:])

        return hf_hub_download(repo_id=repo_id,
                               filename=file_name,
                               revision=revision)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e


class SteerVectorModel(AdapterModel):

    def __init__(self,
                 steer_vector_id=None,
                 layer_payloads=None,
                 scale_factor=1.0,
                 algorithm="direct",
                 multi_vector_data=None) -> None:
        self.id = steer_vector_id
        self.layer_payloads = layer_payloads
        self.scale_factor = scale_factor
        self.algorithm = algorithm
        self.multi_vector_data = multi_vector_data  # For multi-vector mode: list of vector data
        
    @property
    def is_multi_vector(self) -> bool:
        """Check if this is a multi-vector model"""
        return self.multi_vector_data is not None

    @classmethod
    def from_local_checkpoint(
            cls,
            steer_vector_model_path: str,
            steer_vector_id: int,
            config: SteerVectorConfig,
            device: str = "cuda",
            scale_factor: float = 1.0,
            algorithm: str = "direct",
            target_layers: Optional[list[int]] = None,
    ) -> "SteerVectorModel":

        try:
            # Handle algorithm parameter in path
            if "|" in steer_vector_model_path:
                steer_vector_model_path, path_algorithm = steer_vector_model_path.split("|", 1)
                algorithm = path_algorithm

            # Resolve path (local or HF Hub)
            if os.path.exists(steer_vector_model_path):
                file_path = os.path.abspath(steer_vector_model_path)
            else:
                file_path = load_steer_vector_file(steer_vector_model_path)

            # --- Refactored Loading Logic ---
            # Dynamically get the algorithm class from the registry via the factory
            # (without creating an instance)
            if algorithm not in ALGORITHM_REGISTRY:
                 raise ValueError(f"Unsupported algorithm for loading: '{algorithm}'")
            
            algo_class = ALGORITHM_REGISTRY[algorithm]

            # Delegate loading to the algorithm's class method
            loaded_params = algo_class.load_from_path(
                file_path, device, config=config, target_layers=target_layers
            )
            
            # Create SteerVectorModel instance from loaded parameters
            return cls(
                steer_vector_id=steer_vector_id,
                layer_payloads=loaded_params.get("layer_payloads"),
                scale_factor=scale_factor,
                algorithm=algorithm,
            )
            # --- End of Refactored Logic ---

        except Exception as e:
            raise RuntimeError(f"Failed to load steer vector from {steer_vector_model_path} with algorithm '{algorithm}'") from e

    @classmethod
    def from_steer_vector_request(
            cls,
            request,  # SteerVectorRequest
            config: SteerVectorConfig,
            device: str = "cuda",
    ) -> "SteerVectorModel":
        """Create SteerVectorModel from SteerVectorRequest (supports both single and multi-vector)"""
        from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig
        
        if not request.is_multi_vector:
            # Single-vector mode: use existing logic
            return cls.from_local_checkpoint(
                steer_vector_model_path=request.steer_vector_local_path,
                steer_vector_id=request.steer_vector_id,
                config=config,
                device=device,
                scale_factor=request.scale,
                algorithm=request.algorithm,
                target_layers=request.target_layers,
            )
        else:
            # Multi-vector mode: load each vector individually
            multi_vector_data = []
            
            for i, vector_config in enumerate(request.vector_configs):
                try:
                    # Load individual vector
                    single_model = cls.from_local_checkpoint(
                        steer_vector_model_path=vector_config.path,
                        steer_vector_id=f"{request.steer_vector_id}_vec_{i}",  # Unique sub-ID
                        config=config,
                        device=device,
                        scale_factor=vector_config.scale,
                        algorithm=vector_config.algorithm,
                        target_layers=vector_config.target_layers,
                    )
                    
                    # Store vector data with its configuration
                    vector_data = {
                        'payloads': single_model.layer_payloads,
                        'scale': vector_config.scale,
                        'target_layers': vector_config.target_layers,
                        'prefill_trigger_tokens': vector_config.prefill_trigger_tokens,
                        'prefill_trigger_positions': vector_config.prefill_trigger_positions,
                        'prefill_exclude_tokens': vector_config.prefill_exclude_tokens,
                        'prefill_exclude_positions': vector_config.prefill_exclude_positions,
                        'generate_trigger_tokens': vector_config.generate_trigger_tokens,
                        'algorithm': vector_config.algorithm,
                        'path': vector_config.path,
                        'normalize': vector_config.normalize,  # Add normalize setting
                    }
                    multi_vector_data.append(vector_data)
                    
                    logger.debug(f"Loaded vector {i}: {vector_config.path} (algorithm: {vector_config.algorithm}, scale: {vector_config.scale})")
                    
                except Exception as e:
                    logger.error(f"Failed to load vector {i} from {vector_config.path}: {e}")
                    raise RuntimeError(f"Failed to load vector {i} from {vector_config.path}") from e
            
            logger.debug(f"Successfully loaded {len(multi_vector_data)} vectors for multi-vector request '{request.steer_vector_name}'")
            
            return cls(
                steer_vector_id=request.steer_vector_id,
                layer_payloads=None,
                scale_factor=1.0,  # Individual scales are stored in multi_vector_data
                algorithm="multi_vector",  # Special algorithm indicator
                multi_vector_data=multi_vector_data
            )


class SteerVectorModelManager(AdapterModelManager):

    def __init__(self, model: nn.Module,
                 steer_vector_config: SteerVectorConfig):
        self.model = model
        self._registered_adapters = {}
        self._active_adapters = {}
        self.steer_vector_config = steer_vector_config
        self._last_mapping = None
        self.model.steer_vector_manager = self
        self.steer_vector_index_to_id: list[
            Optional[int]] = [None] * self.adapter_slots
        self.modules: dict[str, nn.Module] = {}
        self._create_sv_modules()

    @property
    def adapter_slots(self) -> int:
        return self.capacity

    @property
    def capacity(self) -> int:
        return self.steer_vector_config.max_steer_vectors

    def activate_adapter(
            self,
            steer_vector_id: int,
            target_layers: Optional[list[int]] = None,
            prefill_trigger_tokens: Optional[list[int]] = None,
            prefill_trigger_positions: Optional[list[int]] = None,
            prefill_exclude_tokens: Optional[list[int]] = None,
            prefill_exclude_positions: Optional[list[int]] = None,
            generate_trigger_tokens: Optional[list[int]] = None,
            debug: bool = False,
            conflict_resolution: str = "priority",
            normalize: bool = False,
    ) -> bool:
        if steer_vector_id in self._active_adapters:
            self._deactivate_adapter(steer_vector_id)
            del self._active_adapters[steer_vector_id]

        first_free_slot = next(
            (i for i, slot_id in enumerate(self.steer_vector_index_to_id) if slot_id is None),
            None
        )
        if first_free_slot is None:
            raise ValueError("No free steer vector slots")
        index = first_free_slot
        
        steer_vector_model = self._registered_adapters.get(steer_vector_id)
        if not steer_vector_model:
            raise ValueError(f"Steer vector {steer_vector_id} not found.")

        # 统一准备参数字典
        params = {
            "algorithm_name": steer_vector_model.algorithm,
            "scale_factor": steer_vector_model.scale_factor,
            "prefill_trigger_tokens": prefill_trigger_tokens,
            "prefill_trigger_positions": prefill_trigger_positions,
            "prefill_exclude_tokens": prefill_exclude_tokens,
            "prefill_exclude_positions": prefill_exclude_positions,
            "generate_trigger_tokens": generate_trigger_tokens,
            "debug": debug,
            "normalize": normalize,
        }

        # 根据算法类型，将特定于算法的权重/参数应用到所有目标模块
        if steer_vector_model.is_multi_vector:
            self._activate_multi_vector_adapter(index, steer_vector_model, debug, conflict_resolution)
        
        elif steer_vector_model.layer_payloads:
            for layer_idx, payload in steer_vector_model.layer_payloads.items():
                if target_layers and layer_idx not in target_layers:
                    continue
                for module in self._get_modules_for_layer(layer_idx):
                    module.set_steer_vector(index, payload=payload, **params)
        
        else: # Fallback for models without payloads (e.g., empty multi-vector shell)
            pass # Or log a warning if needed

        self.steer_vector_index_to_id[index] = steer_vector_id
        self._active_adapters[steer_vector_id] = None
        self._set_adapter_mapping(steer_vector_id)

        logger.debug(f"Activated steer vector {steer_vector_id} in slot {index}")
        return True

    def _activate_multi_vector_adapter(self, index: int, steer_vector_model: SteerVectorModel, debug: bool, conflict_resolution: str):
        """专门处理多向量激活的逻辑"""
        layer_to_vectors: Dict[int, List[Tuple[int, Dict]]] = {}
        
        # 1. 收集每个层需要处理的向量
        for vector_idx, vector_data in enumerate(steer_vector_model.multi_vector_data):
            vector_target_layers = vector_data.get('target_layers')
            
            # 使用通用的 'payloads' 键
            affected_layers = list(vector_data.get('payloads', {}).keys())
            
            for layer_idx in affected_layers:
                if vector_target_layers is None or layer_idx in vector_target_layers:
                    if layer_idx not in layer_to_vectors:
                        layer_to_vectors[layer_idx] = []
                    layer_to_vectors[layer_idx].append((vector_idx, vector_data))

        # 2. 为每一层配置算法
        for layer_idx, vectors_for_layer in layer_to_vectors.items():
            for module in self._get_modules_for_layer(layer_idx):
                if len(vectors_for_layer) == 1:
                    # 单个向量，退化为单向量模式处理
                    _, vector_data = vectors_for_layer[0]
                    self._apply_single_vector_to_module(module, index, vector_data, debug, layer_idx)
                else:
                    # 多个向量，配置MultiVectorAlgorithm
                    # 先设置激活算法为multi_vector，以便正确获取实例
                    module.active_algorithm_name = "multi_vector"
                    multi_vector_algo = module._get_or_create_algorithm("multi_vector")
                    multi_vector_algo.set_conflict_resolution(conflict_resolution)
                    multi_vector_algo.set_debug(debug)
                    multi_vector_algo.reset_steer_vector(0) # 清理旧状态
                    
                    # 添加所有子向量
                    for vec_idx, vec_data in vectors_for_layer:
                        add_kwargs = vec_data.copy()
                        algorithm_type = add_kwargs.pop('algorithm')

                        # 使用 'payload' 作为统一的参数名
                        add_kwargs['payload'] = vec_data['payloads'][layer_idx]
                        add_kwargs['scale_factor'] = vec_data.get('scale', 1.0)
                        
                        # 从add_kwargs中移除不属于add_vector的旧参数
                        add_kwargs.pop('payloads', None) # 移除 'payloads'
                        add_kwargs.pop('weights', None)
                        add_kwargs.pop('loreft_params', None)
                        add_kwargs.pop('sv_vector', None) # 确保旧参数被移除
                        
                        multi_vector_algo.add_vector(
                            vector_idx=vec_idx, 
                            algorithm_type=algorithm_type, 
                            **add_kwargs
                        )

    def _get_modules_for_layer(self, layer_idx: int) -> List[nn.Module]:
        """获取指定层的所有模块"""
        modules = []
        for module_name, module in self.modules.items():
            if parse_number_from_string(module_name) == layer_idx:
                modules.append(module)
        return modules

    def _apply_single_vector_to_module(self, module, index, vector_data, debug, layer_idx):
        """辅助方法：应用单个向量到模块"""
        
        # 统一准备参数
        params = {
            "algorithm_name": vector_data['algorithm'],
            "scale_factor": vector_data.get('scale', 1.0), # 传递缩放因子
            "prefill_trigger_tokens": vector_data.get('prefill_trigger_tokens'),
            "prefill_trigger_positions": vector_data.get('prefill_trigger_positions'),
            "prefill_exclude_tokens": vector_data.get('prefill_exclude_tokens'),
            "prefill_exclude_positions": vector_data.get('prefill_exclude_positions'),
            "generate_trigger_tokens": vector_data.get('generate_trigger_tokens'),
            "debug": debug,
            "normalize": vector_data.get('normalize', False),  # Add normalize parameter
        }
        
        # 使用通用的payload
        params["payload"] = vector_data['payloads'][layer_idx]
        
        # 使用通用方法设置
        module.set_steer_vector(index, **params)

    def _deactivate_adapter(self, steer_vector_id: int) -> bool:
        index = self.get_index_from_id(steer_vector_id)
        if index is None:
            return False
        for k, v in self.modules.items():
            v.reset_steer_vector(index)
        self.steer_vector_index_to_id[index] = None
        return True

    def _add_adapter(self, steer_vector: SteerVectorModel) -> bool:
        """Add a SteerVectorModel to the manager."""
        self._registered_adapters[steer_vector.id] = steer_vector

    def get_index_from_id(self, id):
        for i in range(len(self.steer_vector_index_to_id)):
            if self.steer_vector_index_to_id[i] == id:
                return i
        return None

    def _set_adapter_mapping(self, id: int) -> None:
        index = self.get_index_from_id(id)
        
        if index is None:
            logger.warning(f"No slot found for steer vector ID {id}")
            return
            
        for k, v in self.modules.items():
            v.set_active_tensor(index)

    def _create_sv_modules(self):
        """
        创建控制向量模块，仅支持DecoderLayer级别的包装
        """
        self._wrap_decoder_layers()

    def _wrap_decoder_layers(self) -> None:
        """包装DecoderLayer"""
        decoder_layer_wrapped = False
        for module_name, module in self.model.named_modules():
            # 检查是否是DecoderLayer
            if any(class_name in module.__class__.__name__ for class_name in _decoder_layer_class_names):
                if isinstance(module, DecoderLayerWithSteerVector):
                    continue  # 已经被包装过了

                # 包装DecoderLayer
                new_module = self.replace_submodule(
                    self.model, module_name, DecoderLayerWithSteerVector(module))
                new_module.set_layer_id(parse_number_from_string(module_name))
                self.register_module(module_name, new_module)
                # Normalization is now set per-vector through SteerVectorRequest
                decoder_layer_wrapped = True
                logger.debug(f"Wrapped DecoderLayer: {module_name}")

        if decoder_layer_wrapped:
            logger.debug("Using DecoderLayer-level steer vector intervention")
        else:
            logger.warning("No DecoderLayer found for steer vector intervention")



    def replace_submodule(self, model: nn.Module, module_name: str,
                          new_module: nn.Module) -> nn.Module:
        """Replace a submodule in a model with a new module."""
        parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
        target_name = module_name.split(".")[-1]
        setattr(parent, target_name, new_module)
        return new_module

    def register_module(self, module_name: str, module: nn.Module):
        self.modules[module_name] = module

    def remove_all_adapters(self):
        """Remove all PromptAdapterModel from the manager."""
        self._registered_adapters.clear()
        self.steer_vector_index_to_id = [None] * self.adapter_slots
        self._active_adapters.clear()

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return deactivate_adapter(adapter_id, self._active_adapters,
                                  self._deactivate_adapter)

    def add_adapter(self, adapter: SteerVectorModel) -> bool:
        return add_adapter(adapter, self._registered_adapters, self.capacity,
                           self._add_adapter)

    def set_adapter_mapping(self, mapping: SteerVectorMapping) -> None:
        self._last_mapping = set_adapter_mapping(mapping, self._last_mapping,
                                                 self._set_adapter_mapping)

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(adapter_id, self._registered_adapters,
                              self.deactivate_adapter)

    def list_adapters(self) -> dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        return get_adapter(adapter_id, self._registered_adapters)

    def pin_adapter(self, adapter_id: int) -> bool:
        raise NotImplementedError


class SteerVectorLRUCache(AdapterLRUCache[SteerVectorModel]):

    def __init__(self, capacity: int, deactivate_sv_fn: Callable[[int], bool]):
        super().__init__(capacity, deactivate_sv_fn)


class LRUCacheSteerVectorModelManager(SteerVectorModelManager):

    def __init__(self, model: nn.Module,
                 steer_vector_config: SteerVectorConfig):
        self.steer_vector_config = steer_vector_config
        super().__init__(model, steer_vector_config)
        self._registered_adapters = SteerVectorLRUCache(
            self.capacity, self.deactivate_adapter)
        self._active_adapters = SteerVectorLRUCache(self.adapter_slots,
                                                      self._deactivate_adapter)

    def list_adapters(self) -> dict[int, SteerVectorModel]:
        """List all registered SteerVectorModel."""
        return dict(self._registered_adapters.cache)

    def activate_adapter(
            self,
            steer_vector_id: int,
            target_layers: Optional[list[int]] = None,
            prefill_trigger_tokens: Optional[list[int]] = None,
            prefill_trigger_positions: Optional[list[int]] = None,
            prefill_exclude_tokens: Optional[list[int]] = None,
            prefill_exclude_positions: Optional[list[int]] = None,
            generate_trigger_tokens: Optional[list[int]] = None,
            debug: bool = False,
            conflict_resolution: str = "priority",
            normalize: bool = False,
    ) -> bool:
        if steer_vector_id not in self._active_adapters and len(
                self._active_adapters) >= self.adapter_slots:
            self._active_adapters.remove_oldest()
        
        # 调用重构后的激活逻辑
        super().activate_adapter(steer_vector_id, target_layers,
                                          prefill_trigger_tokens,
                                          prefill_trigger_positions,
                                          prefill_exclude_tokens,
                                          prefill_exclude_positions,
                                          generate_trigger_tokens, debug,
                                          conflict_resolution, normalize)
        
        # We always touch to update the LRU cache order
        self._active_adapters.touch(steer_vector_id)
        return True
    
    def remove_oldest_adapter(self) -> bool:
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False


def create_sv_manager(
        model: nn.Module,
        steer_vector_config: SteerVectorConfig,
        steer_vector_manager_cls: type[
            SteerVectorModelManager] = SteerVectorModelManager,
) -> SteerVectorModelManager:
    steer_vector_manager = steer_vector_manager_cls(
        model=model, steer_vector_config=steer_vector_config)

    return steer_vector_manager
