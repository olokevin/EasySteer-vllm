# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List

import torch

from vllm.adapter_commons.utils import (add_adapter_worker,
                                        apply_adapters_worker,
                                        list_adapters_worker,
                                        set_active_adapters_worker)
from vllm.adapter_commons.worker_manager import AbstractWorkerManager
from vllm.config import SteerVectorConfig
from vllm.steer_vectors.models import (SteerVectorModel,
                                      SteerVectorModelManager,
                                      LRUCacheSteerVectorModelManager,
                                      create_sv_manager)
from vllm.steer_vectors.request import SteerVectorRequest

logger = logging.getLogger(__name__)


class WorkerSteerVectorManager(AbstractWorkerManager):
    """WorkerSteerVectorManager that manages
    steer vector models on the worker side.

    Every request, the requested steer vectors will be
    loaded (unless they are already loaded),
    and every other steer vector will be unloaded."""

    _manager_cls: type[SteerVectorModelManager] = SteerVectorModelManager

    def __init__(
        self,
        device: torch.device,
        steer_vector_config: SteerVectorConfig,
        steer_vector_model_cls: type[SteerVectorModel] = SteerVectorModel
    ):
        self._adapter_manager: SteerVectorModelManager
        self._steer_vector_model_cls = steer_vector_model_cls
        self.steer_vector_config = steer_vector_config
        super().__init__(device)

    @property
    def is_enabled(self) -> bool:
        return True

    def create_steer_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        steer_vector_manager = create_sv_manager(
            model,
            steer_vector_config=self.steer_vector_config,
            steer_vector_manager_cls=self._manager_cls,
        )
        self._adapter_manager = steer_vector_manager
        return steer_vector_manager.model

    def _load_adapter(
            self, steer_vector_request: SteerVectorRequest
    ) -> SteerVectorModel:
        try:
            # Use the new from_steer_vector_request method that supports both single and multi-vector
            steer_vector = (
                self._steer_vector_model_cls.from_steer_vector_request(
                    request=steer_vector_request,
                    config=self.steer_vector_config,
                    device=str(self.device)))
        except Exception as e:
            request_info = steer_vector_request.steer_vector_local_path if not steer_vector_request.is_multi_vector else f"multi-vector request with {len(steer_vector_request.vector_configs)} vectors"
            raise RuntimeError(
                f"Loading steer vector {request_info} failed") from e
        return steer_vector

    def add_dummy_steer_vector(
            self, steer_vector_request: SteerVectorRequest) -> bool:
        return True

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: set[Any]) -> None:
        mapping = next((request.adapter_id for request in requests), None)
        set_active_adapters_worker(requests, mapping, self._apply_adapters,
                                   self._adapter_manager.set_adapter_mapping)

    def add_adapter(self, adapter_request: Any) -> bool:
        # Support replacing adapters with the same ID by removing old one first
        if adapter_request.adapter_id in self.list_adapters():
            logger.debug(f"Replacing existing steer vector with ID {adapter_request.adapter_id}")
            self.remove_adapter(adapter_request.adapter_id)
        
        # Define activation function based on request type
        if adapter_request.is_multi_vector:
            # Multi-vector mode: activation is handled internally
            activate_fn = lambda adapter_id: self._adapter_manager.activate_adapter(
                adapter_id,
                debug=adapter_request.debug,
                conflict_resolution=adapter_request.conflict_resolution)
        else:
            # Single-vector mode: use request-level parameters (backward compatibility)
            activate_fn = lambda adapter_id: self._adapter_manager.activate_adapter(
                adapter_id,
                target_layers=adapter_request.target_layers,
                prefill_trigger_tokens=adapter_request.prefill_trigger_tokens,
                prefill_trigger_positions=adapter_request.prefill_trigger_positions,
                generate_trigger_tokens=adapter_request.generate_trigger_tokens,
                debug=adapter_request.debug,
                normalize=adapter_request.normalize)
        
        return add_adapter_worker(adapter_request, self.list_adapters,
                                  self._load_adapter,
                                  self._adapter_manager.add_adapter,
                                  activate_fn)

    def _apply_adapters(self, adapter_requests: set[Any]) -> None:
        apply_adapters_worker(adapter_requests, self.list_adapters,
                              self._adapter_manager.adapter_slots,
                              self.remove_adapter, self.add_adapter)

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> set[int]:
        return list_adapters_worker(self._adapter_manager.list_adapters)


class LRUCacheWorkerSteerVectorManager(WorkerSteerVectorManager):
    """WorkerSteerVectorManager that manages
    steer vector models on the worker side.

    Uses an LRU Cache. Every request, the requested
    steer vectors will be loaded (unless they are already loaded)
    and least recently used steer vectors will
    be unloaded if the cache is above capacity."""

    _steer_vector_manager_cls: type[
        LRUCacheSteerVectorModelManager] = LRUCacheSteerVectorModelManager

    def create_steer_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        steer_vector_manager = create_sv_manager(
            model,
            steer_vector_config=self.steer_vector_config,
            steer_vector_manager_cls=self._steer_vector_manager_cls)
        self._adapter_manager: LRUCacheSteerVectorModelManager = (
            steer_vector_manager)
        return steer_vector_manager.model

    def _apply_adapters(
            self, steer_vector_requests: set[SteerVectorRequest]) -> None:
        steer_vectors_map = {
            steer_vector_request.steer_vector_id: steer_vector_request
            for steer_vector_request in steer_vector_requests
            if steer_vector_request
        }
        if len(steer_vectors_map) > self._adapter_manager.adapter_slots:
            raise RuntimeError(f"Number of requested steer vectors "
                               f"({len(steer_vectors_map)}) is greater "
                               "than the number of GPU steer vector slots "
                               f"({self._adapter_manager.adapter_slots}).")
        for steer_vector in steer_vectors_map.values():
            self.add_adapter(steer_vector)

    def add_adapter(self,
                    steer_vector_request: SteerVectorRequest) -> bool:
        if steer_vector_request.steer_vector_id not in self.list_adapters():
            # Remove before we load the new steer vector to save memory
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                self._adapter_manager.remove_oldest_adapter()
            steer_vector = self._load_adapter(steer_vector_request)
            loaded = self._adapter_manager.add_adapter(steer_vector)
        else:
            # Support replacing adapters with the same ID by removing and reloading
            logger.debug(f"Replacing existing steer vector with ID {steer_vector_request.steer_vector_id}")
            self._adapter_manager.remove_adapter(steer_vector_request.steer_vector_id)
            steer_vector = self._load_adapter(steer_vector_request)
            loaded = self._adapter_manager.add_adapter(steer_vector)
        
        # For multi-vector mode, activation is handled differently
        if steer_vector_request.is_multi_vector:
            # Multi-vector mode: activation is handled internally
            self._adapter_manager.activate_adapter(
                steer_vector_request.steer_vector_id,
                debug=steer_vector_request.debug,
                conflict_resolution=steer_vector_request.conflict_resolution)
        else:
            # Single-vector mode: use request-level parameters (backward compatibility)
            self._adapter_manager.activate_adapter(
                steer_vector_request.steer_vector_id,
                target_layers=steer_vector_request.target_layers,
                prefill_trigger_tokens=steer_vector_request.prefill_trigger_tokens,
                prefill_trigger_positions=steer_vector_request.prefill_trigger_positions,
                generate_trigger_tokens=steer_vector_request.generate_trigger_tokens,
                debug=steer_vector_request.debug,
                normalize=steer_vector_request.normalize)
        return loaded
