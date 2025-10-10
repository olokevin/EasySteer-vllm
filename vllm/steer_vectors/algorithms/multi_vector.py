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

    def _apply_global_vectors(self, hidden_states: torch.Tensor, vectors_in_order: List[Tuple[int, AlgorithmTemplate]], phase: str) -> torch.Tensor:
        """
        当所有向量都在某阶段（prefill/generate）对全部 token 生效（触发器为 [-1] 且无位置触发）时，
        走快速路径，避免构建按位置/样本的映射。

        Args:
            hidden_states: 需要变换的 hidden states（prefill: [T, H]，generate: [B, H]）。
            vectors_in_order: 需要应用的向量（保持插入顺序）。
            phase: 'prefill' 或 'generate'，仅用于日志。
        """
        if not vectors_in_order:
            return hidden_states

        if self.conflict_resolution == "error" and len(vectors_in_order) > 1:
            raise ValueError(
                f"Multiple global vectors conflict in {phase} phase: "
                f"vector indices {[vid for vid, _ in vectors_in_order]}"
            )

        # priority: 仅使用第一个； sequential: 顺序全部应用
        to_apply: List[Tuple[int, AlgorithmTemplate]] = (
            [vectors_in_order[0]] if self.conflict_resolution == "priority" else vectors_in_order
        )

        for vec_idx, algo in to_apply:
            algo.set_active_tensor(0)
            params = algo._get_params()
            if not algo._is_valid(params):
                continue
            original_dtype = hidden_states.dtype
            hidden_states = algo._transform(hidden_states, params).to(original_dtype)
            if self.debug:
                print(f"[MultiVector] FastPath({phase}) applied vector {vec_idx} to ALL")

        return hidden_states

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
        if "prefill_exclude_tokens" in kwargs:
            algo.set_prefill_exclude_tokens(kwargs["prefill_exclude_tokens"])
        if "prefill_exclude_positions" in kwargs:
            algo.set_prefill_exclude_positions(kwargs["prefill_exclude_positions"])
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

            # 处理 exclude tokens (优先级高于 include)
            if algo.prefill_exclude_tokens is not None and positions_for_this_vector:
                if current_tokens is not None and current_tokens.numel() > 0 and current_tokens.dim() == 1:
                    # 过滤掉需要 exclude 的位置
                    filtered_positions = []
                    for pos in positions_for_this_vector:
                        if pos < current_tokens.shape[0]:
                            token_id = current_tokens[pos].item()
                            if token_id not in algo.prefill_exclude_tokens:
                                filtered_positions.append(pos)
                        else:
                            filtered_positions.append(pos)
                    positions_for_this_vector = filtered_positions

            # 处理 exclude positions (优先级高于 include)
            if algo.prefill_exclude_positions is not None and positions_for_this_vector:
                # 解析要排除的位置（支持负数索引）
                if seq_start_loc is not None:
                    excluded_positions_set = set(algo._resolve_positions_per_sample(algo.prefill_exclude_positions, seq_start_loc))
                else:
                    excluded_positions_set = set(algo._resolve_positions(algo.prefill_exclude_positions, total_tokens))
                
                # 过滤掉要排除的位置
                positions_for_this_vector = [pos for pos in positions_for_this_vector if pos not in excluded_positions_set]

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
            position_to_vectors: 每个位置/样本的向量列表 (会被就地修改)
            
        Returns:
            解决冲突后的映射
        """
        # In 'sequential' mode, no conflict resolution is necessary; all vectors are applied in order.
        # We can return early to avoid an unnecessary loop.
        if self.conflict_resolution == "sequential":
            if self.debug:
                for pos, vectors in position_to_vectors.items():
                    if len(vectors) > 1:
                        vector_indices = [v.vector_idx for v in vectors]
                        print(f"[MultiVector] Conflict at position {pos}: "
                              f"vectors {vector_indices}, applying all in sequence (sequential mode)")
            return position_to_vectors

        # For 'error' or 'priority' modes, we iterate to find and handle conflicts.
        if self.conflict_resolution in ["error", "priority"]:
            for pos, vectors in position_to_vectors.items():
                if len(vectors) > 1:
                    # A conflict exists.
                    if self.conflict_resolution == "error":
                        vector_indices = [v.vector_idx for v in vectors]
                        raise ValueError(
                            f"Multiple vectors conflict at position/sample {pos}: "
                            f"vector indices {vector_indices}. "
                            f"Set conflict_resolution='priority' to use the first vector, "
                            f"or 'sequential' to apply all vectors in sequence."
                        )
                    elif self.conflict_resolution == "priority":
                        # Use the first vector.
                        position_to_vectors[pos] = [vectors[0]]
                        if self.debug:
                            vector_indices = [v.vector_idx for v in vectors]
                            print(f"[MultiVector] Conflict at position {pos}: "
                                  f"vectors {vector_indices}, using vector {vectors[0].vector_idx} (priority mode)")
            return position_to_vectors
        
        # If we reach here, the strategy is unknown.
        raise ValueError(f"Unknown conflict resolution strategy: {self.conflict_resolution}")

    def _apply_vectorized(self, hidden_states: torch.Tensor, resolved_mapping: Dict[int, List[VectorInstance]], target_name: str):
        """
        Vectorized application of transformations.
        
        Args:
            hidden_states: The tensor to modify.
            resolved_mapping: A map from index (sample_idx or pos) to a list of vector instances to apply.
            target_name: A string ('sample' or 'position') for logging.
        """
        if not resolved_mapping:
            return

        # Find the maximum number of vectors to apply sequentially to any single item.
        max_depth = max((len(v) for v in resolved_mapping.values()), default=0)

        # Apply vectors layer by layer to handle sequential application.
        for depth in range(max_depth):
            # Group indices by vector_idx for the current application depth.
            vector_to_indices: Dict[int, List[int]] = {}
            for idx, vectors in resolved_mapping.items():
                if len(vectors) > depth:
                    instance = vectors[depth]
                    vector_idx = instance.vector_idx
                    if vector_idx not in vector_to_indices:
                        vector_to_indices[vector_idx] = []
                    vector_to_indices[vector_idx].append(idx)
            
            # Apply transformations for all vectors at the current depth.
            for vector_idx, indices in vector_to_indices.items():
                if not indices:
                    continue

                algo = self.vector_algorithms.get(vector_idx)
                if not algo:
                    continue
                    
                # Activate the algorithm and get its parameters.
                algo.set_active_tensor(0)  # Use index 0 for internal storage.
                params = algo._get_params()
                if not algo._is_valid(params):
                    continue
                
                # Perform the vectorized transformation.
                valid_indices = sorted([i for i in indices if i < hidden_states.shape[0]])
                if not valid_indices:
                    continue
                
                indices_tensor = torch.tensor(valid_indices, device=hidden_states.device, dtype=torch.long)
                
                selected_states = hidden_states.index_select(0, indices_tensor)
                original_dtype = selected_states.dtype
                
                transformed_states = algo._transform(selected_states, params).to(original_dtype)
                
                hidden_states.index_copy_(0, indices_tensor, transformed_states)

                if self.debug:
                    print(f"[MultiVector] Applied vector {vector_idx} to {target_name}s: {valid_indices} at depth {depth}")


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

        # 收集、解决冲突、然后应用
        if is_decode:
            # Generate phase
            # Fast path: 所有向量均对全部样本生效（generate_trigger_tokens == {-1} 且无其他触发）
            global_only_vectors: List[Tuple[int, AlgorithmTemplate]] = []
            mixed_case = False
            for vid, algo in self.vector_algorithms.items():
                if algo.generate_trigger_tokens is None and algo.prefill_trigger_positions is None:
                    continue
                # 仅当 generate 触发器为 {-1} 且无其他 generate 触发条件时视为全局
                is_global_only = (
                    algo.generate_trigger_tokens is not None and
                    (-1 in algo.generate_trigger_tokens) and
                    len(algo.generate_trigger_tokens) == 1
                )
                if is_global_only:
                    global_only_vectors.append((vid, algo))
                elif algo.generate_trigger_tokens is not None:
                    mixed_case = True

            if global_only_vectors and not mixed_case:
                # 所有向量都是全局触发，直接快速路径
                hidden_states = self._apply_global_vectors(hidden_states, global_only_vectors, "generate")
                return hidden_states

            sample_to_vectors = self._gather_applicable_vectors_generate(hidden_states, forward_ctx)
            resolved_mapping = self._resolve_conflicts(sample_to_vectors)
            self._apply_vectorized(hidden_states, resolved_mapping, "sample")
        else:
            # Prefill phase
            # Fast path: 所有向量均对全部 token 生效（prefill_trigger_tokens == {-1} 且无位置触发且无exclude）
            global_only_vectors: List[Tuple[int, AlgorithmTemplate]] = []
            mixed_case = False
            for vid, algo in self.vector_algorithms.items():
                if not algo._has_prefill_triggers():
                    continue
                is_global_only = (
                    algo.prefill_trigger_tokens is not None and
                    (-1 in algo.prefill_trigger_tokens) and
                    len(algo.prefill_trigger_tokens) == 1 and
                    algo.prefill_trigger_positions is None and
                    algo.prefill_exclude_tokens is None and  # 如果有exclude tokens，不能走全局路径
                    algo.prefill_exclude_positions is None  # 如果有exclude positions，不能走全局路径
                )
                if is_global_only:
                    global_only_vectors.append((vid, algo))
                else:
                    mixed_case = True

            if global_only_vectors and not mixed_case:
                # 所有向量都是全局触发，直接快速路径
                hidden_states = self._apply_global_vectors(hidden_states, global_only_vectors, "prefill")
                return hidden_states

            position_to_vectors = self._gather_applicable_vectors_prefill(hidden_states, forward_ctx)
            resolved_mapping = self._resolve_conflicts(position_to_vectors)
            self._apply_vectorized(hidden_states, resolved_mapping, "position")

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