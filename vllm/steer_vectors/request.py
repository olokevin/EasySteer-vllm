# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, List
from vllm.adapter_commons.request import AdapterRequest


@dataclass
class VectorConfig:
    """
    Configuration for a single vector in multi-vector mode.
    
    Args:
        path: Local path to the vector file
        scale: Scale factor for this vector (default: 1.0)
        target_layers: List of layer indices to apply this vector to. If None, apply to all layers
        prefill_trigger_tokens: List of token IDs that trigger vector application in prefill phase.
                               Use [-1] to apply to ALL tokens in prefill phase.
        prefill_trigger_positions: List of token positions that trigger vector application in prefill phase.
                                 Supports negative indexing (e.g., -1 for last token).
        generate_trigger_tokens: List of token IDs that trigger vector application in generate phase.
                                Use [-1] to apply to ALL tokens in generate phase.
        algorithm: Vector algorithm to use: 'direct' (default) or 'loreft'
        normalize: Whether to normalize the vector (default: False, only applies to 'direct' algorithm)
    """
    path: str
    scale: float = 1.0
    target_layers: Optional[List[int]] = None
    prefill_trigger_tokens: Optional[List[int]] = None
    prefill_trigger_positions: Optional[List[int]] = None
    generate_trigger_tokens: Optional[List[int]] = None
    algorithm: str = "direct"
    normalize: bool = False


@dataclass
class SteerVectorRequest(AdapterRequest):
    """
    Request for a Steer Vector adapter.
    Supports both single-vector mode (backward compatible) and multi-vector mode.

    Args:
        steer_vector_name: Name of the steer vector
        steer_vector_id: Unique ID for the steer vector
        debug: Whether to print debug information during forward pass (default: False)
        conflict_resolution: How to handle conflicts when multiple vectors target the same position.
                           'error': raise an error when conflicts occur
                           'priority': use the first vector and ignore others (default)
                           'sequential': apply all vectors in sequence (effects stack)
        
        Single-vector mode (backward compatible):
        steer_vector_local_path: Local path to the steer vector file
        scale: Scale factor for the steer vector (default: 1.0)
        target_layers: List of layer indices to apply the steer vector to. If None, apply to all layers
        prefill_trigger_tokens: List of token IDs that trigger steer vector application in prefill phase.
        prefill_trigger_positions: List of token positions that trigger steer vector application in prefill phase.
        generate_trigger_tokens: List of token IDs that trigger steer vector application in generate phase.
        algorithm: Steer vector algorithm to use: 'direct' (default) or 'loreft'
        normalize: Whether to normalize the steer vector (default: False, only applies to 'direct' algorithm)
        
        Multi-vector mode:
        vector_configs: List of VectorConfig objects for multi-vector control
    """

    steer_vector_name: str
    steer_vector_id: int
    debug: bool = False  # Global debug parameter
    conflict_resolution: str = "priority"  # 'error' or 'priority'
    
    # === Single-vector mode (backward compatible) ===
    steer_vector_local_path: Optional[str] = None
    scale: float = 1.0
    target_layers: Optional[List[int]] = None
    prefill_trigger_tokens: Optional[List[int]] = None
    prefill_trigger_positions: Optional[List[int]] = None
    generate_trigger_tokens: Optional[List[int]] = None
    algorithm: str = "direct"
    normalize: bool = False
    
    # === Multi-vector mode ===
    vector_configs: Optional[List[VectorConfig]] = None

    def __post_init__(self):
        """Validate configuration consistency"""
        if self.conflict_resolution not in ["error", "priority", "sequential"]:
            raise ValueError(f"conflict_resolution must be 'error', 'priority', or 'sequential', got '{self.conflict_resolution}'")
            
        if self.is_multi_vector:
            if self.steer_vector_local_path is not None:
                raise ValueError("Cannot specify both steer_vector_local_path and vector_configs")
            if not self.vector_configs:
                raise ValueError("vector_configs cannot be empty in multi-vector mode")
        else:
            if self.steer_vector_local_path is None:
                raise ValueError("Must specify steer_vector_local_path in single-vector mode")

    @property
    def is_multi_vector(self) -> bool:
        """Check if this is a multi-vector request"""
        return self.vector_configs is not None

    def __hash__(self):
        return super().__hash__()

    @property
    def adapter_id(self):
        return self.steer_vector_id

    @property
    def name(self):
        return self.steer_vector_name

    @property
    def local_path(self):
        """Backward compatibility property"""
        if self.is_multi_vector:
            return None  # Multi-vector mode doesn't have a single path
        return self.steer_vector_local_path

    @property
    def scale_factor(self):
        """Backward compatibility property"""
        if self.is_multi_vector:
            return 1.0  # Multi-vector mode uses individual scales
        return self.scale





