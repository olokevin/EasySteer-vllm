# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List, Dict, Set, Tuple, Any
import torch
from dataclasses import dataclass

from .template import AlgorithmTemplate
from .factory import register_algorithm, create_algorithm

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


@dataclass
class VectorInstance:
    """表示一个要应用的向量实例"""
    vector_idx: int
    algorithm: AlgorithmTemplate
    scale: float = 1.0


@register_algorithm("multi_vector")
class MultiVectorAlgorithm(AlgorithmTemplate):
    """多向量控制的算法实现，支持在同一层应用多个向量"""
    
    def __init__(self, layer_id: Optional[int] = None):
        super().__init__(layer_id)
        # 存储每个向量索引对应的算法实例
        self.vector_algorithms: Dict[int, AlgorithmTemplate] = {}
        # 冲突解决策略
        self.conflict_resolution: str = "priority"  # 'error', 'priority', or 'sequential'
        
    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """
        MultiVectorAlgorithm is a container and does not load from a single path.
        This method is implemented to satisfy the abstract base class contract.
        """
        return {}
        
    def set_conflict_resolution(self, conflict_resolution: str) -> None:
        """设置冲突解决策略"""
        self.conflict_resolution = conflict_resolution

    def add_vector(self, vector_idx: int, algorithm_type: str, **kwargs) -> None:
        """添加一个向量到多向量管理器"""
        # 提取构造函数参数 (如 normalize)
        init_kwargs = {}
        if "normalize" in kwargs:
            init_kwargs["normalize"] = kwargs.get("normalize")
        
        # 使用工厂创建算法实例
        algo = create_algorithm(algorithm_type, layer_id=self.layer_id, **init_kwargs)
        
        # 统一准备 set_steer_vector 的参数
        set_vector_kwargs = {}
        if "payload" in kwargs:
            set_vector_kwargs["payload"] = kwargs["payload"]
        if "scale_factor" in kwargs:
            set_vector_kwargs["scale_factor"] = kwargs.get("scale_factor", 1.0)

        if set_vector_kwargs:
            algo.set_steer_vector(0, **set_vector_kwargs)  # Use index 0 for internal storage
        
        # 设置触发器配置
        if "prefill_trigger_tokens" in kwargs:
            algo.set_prefill_trigger_tokens(kwargs["prefill_trigger_tokens"])
        if "prefill_trigger_positions" in kwargs:
            algo.set_prefill_trigger_positions(kwargs["prefill_trigger_positions"])
        if "generate_trigger_tokens" in kwargs:
            algo.set_generate_trigger_tokens(kwargs["generate_trigger_tokens"])
        if "debug" in kwargs:
            algo.set_debug(kwargs["debug"])
            
        # Store the algorithm instance
        self.vector_algorithms[vector_idx] = algo
        
    def remove_vector(self, vector_idx: int) -> None:
        """移除一个向量"""
        if vector_idx in self.vector_algorithms:
            del self.vector_algorithms[vector_idx]

    def _gather_applicable_vectors_prefill(self, hidden_states: torch.Tensor, forward_ctx: Any) -> Dict[int, List[VectorInstance]]:
        """
        收集阶段（Prefill）：计算每个位置需要应用的向量列表
        
        Returns:
            Dict[position -> List[VectorInstance]]
        """
        position_to_vectors: Dict[int, List[VectorInstance]] = {}
        
        current_tokens = forward_ctx.current_tokens
        attn_metadata = forward_ctx.attn_metadata
        total_tokens = hidden_states.shape[0]

        # Get seq_start_loc for multi-sample handling
        seq_start_loc = None
        if hasattr(attn_metadata, 'seq_start_loc') and attn_metadata.seq_start_loc is not None:
            seq_start_loc = attn_metadata.seq_start_loc
        elif hasattr(attn_metadata, 'prefill_metadata') and attn_metadata.prefill_metadata is not None:
            prefill_meta = attn_metadata.prefill_metadata
            if hasattr(prefill_meta, 'seq_start_loc'):
                seq_start_loc = prefill_meta.seq_start_loc

        # 遍历每个向量配置
        for vector_idx, algo in self.vector_algorithms.items():
            # 跳过没有prefill触发器的向量
            if not algo._has_prefill_triggers():
                continue

            positions_for_this_vector = []

            # 处理全局应用
            if algo._should_apply_to_all_prefill_tokens():
                positions_for_this_vector = list(range(total_tokens))
            else:
                # 1. 处理位置触发
                if algo.prefill_trigger_positions is not None:
                    if seq_start_loc is not None:
                        resolved_positions = algo._resolve_positions_per_sample(algo.prefill_trigger_positions, seq_start_loc)
                    else:
                        resolved_positions = algo._resolve_positions(algo.prefill_trigger_positions, total_tokens)
                    positions_for_this_vector.extend(resolved_positions)

                # 2. 处理token触发
                if algo.prefill_trigger_tokens is not None and seq_start_loc is not None and current_tokens is not None and current_tokens.numel() > 0:
                    num_samples = len(seq_start_loc) - 1
                    
                    for sample_idx in range(num_samples):
                        start_pos = seq_start_loc[sample_idx].item()
                        end_pos = seq_start_loc[sample_idx + 1].item()

                        if current_tokens.dim() == 1 and current_tokens.shape[0] >= end_pos:
                            sample_token_ids = current_tokens[start_pos:end_pos]

                            for rel_pos, token_id in enumerate(sample_token_ids):
                                if token_id.item() in algo.prefill_trigger_tokens:
                                    abs_pos = start_pos + rel_pos
                                    positions_for_this_vector.append(abs_pos)

            # 去重并添加到position_to_vectors映射
            for pos in set(positions_for_this_vector):
                if pos not in position_to_vectors:
                    position_to_vectors[pos] = []
                position_to_vectors[pos].append(VectorInstance(vector_idx=vector_idx, algorithm=algo))

        return position_to_vectors

    def _gather_applicable_vectors_generate(self, hidden_states: torch.Tensor, forward_ctx: Any) -> Dict[int, List[VectorInstance]]:
        """
        收集阶段（Generate）：计算每个样本需要应用的向量列表
        
        Returns:
            Dict[sample_idx -> List[VectorInstance]]
        """
        sample_to_vectors: Dict[int, List[VectorInstance]] = {}
        
        current_tokens = forward_ctx.current_tokens
        batch_size = hidden_states.shape[0]

        if current_tokens is None or current_tokens.numel() == 0:
            return sample_to_vectors

        if current_tokens.dim() == 2:
            current_tokens = current_tokens.flatten()

        # 遍历每个向量配置
        for vector_idx, algo in self.vector_algorithms.items():
            # 跳过没有generate触发器的向量
            if algo.generate_trigger_tokens is None:
                continue

            samples_for_this_vector = []

            # 处理全局应用
            if algo._should_apply_to_all_generate_tokens():
                samples_for_this_vector = list(range(batch_size))
            else:
                # 检查每个样本的当前token
                for i in range(min(batch_size, current_tokens.shape[0])):
                    token_id = current_tokens[i].item()
                    if token_id in algo.generate_trigger_tokens:
                        samples_for_this_vector.append(i)

            # 添加到sample_to_vectors映射
            for sample_idx in samples_for_this_vector:
                if sample_idx not in sample_to_vectors:
                    sample_to_vectors[sample_idx] = []
                sample_to_vectors[sample_idx].append(VectorInstance(vector_idx=vector_idx, algorithm=algo))

        return sample_to_vectors

    def _resolve_conflicts(self, position_to_vectors: Dict[int, List[VectorInstance]]) -> Dict[int, List[VectorInstance]]:
        """
        解决冲突阶段：根据conflict_resolution策略处理冲突
        
        Args:
            position_to_vectors: 每个位置/样本的向量列表
            
        Returns:
            解决冲突后的映射
        """
        resolved = {}
        
        for pos, vectors in position_to_vectors.items():
            if len(vectors) <= 1:
                # 没有冲突
                resolved[pos] = vectors
            else:
                # 存在冲突
                if self.conflict_resolution == "error":
                    vector_indices = [v.vector_idx for v in vectors]
                    raise ValueError(
                        f"Multiple vectors conflict at position/sample {pos}: "
                        f"vector indices {vector_indices}. "
                        f"Set conflict_resolution='priority' to use the first vector, "
                        f"or 'sequential' to apply all vectors in sequence."
                    )
                elif self.conflict_resolution == "priority":
                    # 使用第一个向量
                    resolved[pos] = [vectors[0]]
                    if self.debug:
                        vector_indices = [v.vector_idx for v in vectors]
                        print(f"[MultiVector] Conflict at position {pos}: "
                              f"vectors {vector_indices}, using vector {vectors[0].vector_idx} (priority mode)")
                elif self.conflict_resolution == "sequential":
                    # 保留所有向量，按顺序应用
                    resolved[pos] = vectors
                    if self.debug:
                        vector_indices = [v.vector_idx for v in vectors]
                        print(f"[MultiVector] Conflict at position {pos}: "
                              f"vectors {vector_indices}, applying all in sequence (sequential mode)")
                else:
                    raise ValueError(f"Unknown conflict resolution strategy: {self.conflict_resolution}")
                        
        return resolved

    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用多向量干预"""
        if not self.vector_algorithms:
            return hidden_states

        # 获取forward context信息
        if get_forward_context is None:
            return hidden_states

        forward_ctx = get_forward_context()
        if forward_ctx is None:
            return hidden_states

        is_decode = forward_ctx.is_decode

        # 收集阶段
        if is_decode:
            # Generate phase
            sample_to_vectors = self._gather_applicable_vectors_generate(hidden_states, forward_ctx)
            # 解决冲突
            resolved_mapping = self._resolve_conflicts(sample_to_vectors)
            
            # 应用阶段：对每个样本顺序应用其对应的向量
            for sample_idx, vectors in resolved_mapping.items():
                if sample_idx < hidden_states.shape[0]:
                    # 顺序应用每个向量
                    for vector_instance in vectors:
                        algo = vector_instance.algorithm
                        # 临时设置算法的active状态
                        algo.set_active_tensor(0)  # Use index 0 for internal storage
                        
                        # 获取算法参数并应用变换
                        params = algo._get_params()
                        if algo._is_valid(params):
                            hidden_states[sample_idx] = algo._transform(hidden_states[sample_idx], params)
                            
                        if self.debug:
                            print(f"[MultiVector] Applied vector {vector_instance.vector_idx} to sample {sample_idx}")
        else:
            # Prefill phase
            position_to_vectors = self._gather_applicable_vectors_prefill(hidden_states, forward_ctx)
            # 解决冲突
            resolved_mapping = self._resolve_conflicts(position_to_vectors)
            
            # 应用阶段：对每个位置顺序应用其对应的向量
            for pos, vectors in resolved_mapping.items():
                if pos < hidden_states.shape[0]:
                    # 顺序应用每个向量
                    for vector_instance in vectors:
                        algo = vector_instance.algorithm
                        # 临时设置算法的active状态
                        algo.set_active_tensor(0)  # Use index 0 for internal storage
                        
                        # 获取算法参数并应用变换
                        params = algo._get_params()
                        if algo._is_valid(params):
                            hidden_states[pos] = algo._transform(hidden_states[pos], params)
                            
                        if self.debug:
                            print(f"[MultiVector] Applied vector {vector_instance.vector_idx} to position {pos}")

        return hidden_states

    # 实现算法模板要求的抽象方法（在多向量模式下不直接使用）
    def _get_params(self) -> Any:
        """多向量模式下不使用此方法"""
        return None

    def _is_valid(self, params: Any) -> bool:
        """多向量模式下不使用此方法"""
        return False

    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """多向量模式下不使用此方法"""
        return hidden_state

    # 以下是为了符合BaseSteerVectorAlgorithm接口的方法
    def set_steer_vector(self, index: int, **kwargs) -> None:
        """这个方法在多向量模式下不直接使用"""
        pass

    def reset_steer_vector(self, index: int) -> None:
        """重置所有向量"""
        self.vector_algorithms.clear()

    def set_active_tensor(self, index: int) -> None:
        """这个方法在多向量模式下不直接使用"""
        pass 