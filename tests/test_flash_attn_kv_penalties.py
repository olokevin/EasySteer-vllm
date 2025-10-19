import types
from contextlib import nullcontext
from unittest.mock import patch

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.flash_attn import (FlashAttentionImpl,
                                                FlashAttentionMetadata)


class DummyCompressor:
    last_call_kwargs = None

    def __init__(self):
        self.calls = []

    def update_kv(self, key_states, query_states, value_states, **kwargs):
        DummyCompressor.last_call_kwargs = kwargs
        self.calls.append(kwargs)
        return key_states, value_states


def _dummy_flash_attn(*args, **kwargs):
    return None


def _dummy_get_seq_args(*args, **kwargs):
    seq_lens = torch.tensor([1], dtype=torch.int32)
    return seq_lens, None, None


def test_flash_attn_passes_kv_penalties(monkeypatch):
    impl = FlashAttentionImpl(num_heads=1,
                              head_size=32,
                              scale=1.0,
                              num_kv_heads=1,
                              alibi_slopes=None,
                              sliding_window=None,
                              kv_cache_dtype="auto",
                              logits_soft_cap=None,
                              attn_type=AttentionType.DECODER)
    monkeypatch.setattr(
        "vllm.attention.backends.flash_attn.VLLM_V1_R_KV_BUDGET", 2,
        raising=False)
    monkeypatch.setattr(
        "vllm.attention.backends.flash_attn.VLLM_V1_R_KV_BUFFER", 1,
        raising=False)
    dummy_compressor = DummyCompressor()
    impl.kvcompressor = dummy_compressor
    impl.kv_method = "rkv"

    query = torch.zeros((1, 1, 32), dtype=torch.float16)
    kv_cache = torch.zeros((2, 1, 16, 1, 32), dtype=torch.float16)

    context_len = 6
    occupied = torch.arange(context_len, dtype=torch.int32)
    remove_map = {(0, 3): 0.75}
    completed_ranges = [(0, 3)]
    similarity = [[0.5]]

    decode_meta = FlashAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.arange(context_len, dtype=torch.int32),
        multi_modal_placeholder_index_maps=None,
        enable_kv_scales_calculation=True,
        seq_lens=None,
        seq_lens_tensor=None,
        max_prefill_seq_len=0,
        max_decode_seq_len=context_len,
        context_lens_tensor=torch.tensor([context_len], dtype=torch.int32),
        context_lens=[context_len],
        block_tables=None,
        use_cuda_graph=False,
        max_query_len=1,
        max_decode_query_len=1,
        query_start_loc=None,
        seq_start_loc=None,
        encoder_seq_lens=None,
        encoder_seq_lens_tensor=None,
        encoder_seq_start_loc=None,
        max_encoder_seq_len=None,
        num_encoder_tokens=None,
        cross_slot_mapping=None,
        cross_block_tables=None,
        occupied_slot_mapping=occupied,
        total_num_kv_cache_tokens=context_len,
        num_dropped_tokens_list=[0],
        remove_punct_ranges_list=[remove_map],
        completed_punct_ranges_list=[completed_ranges],
        sentence_similarity_list=[similarity],
    )

    class _MetadataWrapper:
        def __init__(self, decode_meta_inner):
            self._decode_meta = decode_meta_inner
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.num_decode_tokens = decode_meta_inner.num_decode_tokens
            self.slot_mapping = decode_meta_inner.slot_mapping
            self.multi_modal_placeholder_index_maps = None
            self.enable_kv_scales_calculation = True

        @property
        def prefill_metadata(self):
            return None

        @property
        def decode_metadata(self):
            return self._decode_meta

    attn_metadata = _MetadataWrapper(decode_meta)

    layer = types.SimpleNamespace(
        layer_idx=0,
        _k_scale=torch.tensor(1.0, dtype=torch.float32),
        _v_scale=torch.tensor(1.0, dtype=torch.float32),
        _q_scale=torch.tensor(1.0, dtype=torch.float32),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )

    output = torch.zeros_like(query)
    acquire_counter = {"count": 0}

    assert ((impl.kv_method == "rkv2_slow" and impl._kvcompressor_cls is not None)
            or impl.kvcompressor is not None), "Compressor not configured"
    assert decode_meta.context_lens == [context_len]
    assert decode_meta.occupied_slot_mapping.numel() == context_len

    original_acquire = impl._acquire_sequence_compressor

    def intercept_acquire(self, indices):
        acquire_counter["count"] += 1
        return original_acquire(indices)

    impl._acquire_sequence_compressor = intercept_acquire.__get__(
        impl, FlashAttentionImpl)

    ops_patch = (patch.object(torch.ops._C_cache_ops, "reshape_and_cache_flash",
                              lambda *args, **kwargs: None)
                 if hasattr(torch.ops, "_C_cache_ops")
                 and hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash")
                 else nullcontext())

    with patch("vllm.attention.backends.flash_attn.flash_attn_with_kvcache",
               _dummy_flash_attn), \
         patch("vllm.attention.backends.flash_attn.flash_attn_varlen_func",
               _dummy_flash_attn), \
         patch("vllm.attention.backends.flash_attn.get_seq_len_block_table_args",
               _dummy_get_seq_args), \
         ops_patch:
        DummyCompressor.last_call_kwargs = None
        impl.forward(layer=layer,
                     query=query,
                     key=None,
                     value=None,
                     kv_cache=kv_cache,
                     attn_metadata=attn_metadata,
                     output=output,
                     output_scale=None)

    assert acquire_counter["count"] > 0, "internal compressor branch not hit"
    assert DummyCompressor.last_call_kwargs is not None, \
        "update_kv should be invoked"
    kwargs = DummyCompressor.last_call_kwargs
    assert kwargs.get("remove_punct_ranges") == remove_map
    assert kwargs.get("completed_punct_ranges") == completed_ranges
    assert kwargs.get("sentence_similarity") == similarity
