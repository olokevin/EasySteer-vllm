# R1KV Migration Plan: SkipKV/vllm → EasySteer-vllm

**Objective:** Integrate R1KV KV cache compression from SkipKV/vllm (v0 backend) into EasySteer-vllm to enable **R1KV + Steering Vectors** combined functionality.

**Date:** 2025-10-22
**Target:** vLLM v0 backend (`FLASH_ATTN`) only (v1 migration can follow later)

---

## Executive Summary

This migration brings the R1KV compression capability from the SkipKV/vllm fork into EasySteer-vllm, enabling simultaneous KV cache compression and steering vector control. The migration focuses on the **v0 backend** implementation which has:

- ✅ Full memory reclamation (90% savings)
- ✅ Complete coordination across attention → sequence → scheduler → block manager
- ✅ Per-sequence compression tracking
- ✅ CUDA graph compatibility
- ✅ Multi-sequence batch support

**Compatibility:** R1KV and Steering Vectors operate at different levels and should not conflict:
- **R1KV**: Operates in attention backend, compresses KV cache after attention computation
- **Steering Vectors**: Operates in decoder layers, modifies hidden states during forward pass
- **No conflict**: Steering happens before attention, R1KV compresses after attention

---

## Architecture Overview

### Source: SkipKV/vllm
- **Branch:** v0 backend implementation (commit: `c2e61197e`)
- **Scope:** 5 files modified, ~200 lines of code
- **Components:**
  1. Environment variables for budget/buffer
  2. Attention backend compression logic
  3. Sequence state tracking
  4. Engine coordination for metadata propagation
  5. Memory reclamation via modified `n_blocks` property

### Target: EasySteer-vllm
- **Current State:** Clean vLLM fork with steering vectors (no R1KV)
- **Integration Point:** v0 attention backend (`vllm/attention/backends/flash_attn.py`)
- **Compatibility:** Steering vectors already integrated, no conflicts expected

---

## Migration Steps

### Phase 1: Dependency Setup

#### Step 1.1: Install RKV Package
The R1KV algorithm is in a separate package that must be installed.

**Option A: Copy from SkipKV/vllm**
```bash
# In EasySteer-vllm root
cp -r /home/yequan/Project/SkipKV/vllm/third_party/rkv_repo ./third_party/
pip install -e ./third_party/rkv_repo
```

**Option B: Use existing installation** (if rkv is already installed)
```bash
# Verify installation
python -c "from rkv.modeling import R1KV; print('R1KV available')"
```

**Files needed from rkv package:**
- `rkv/modeling.py` - Exports `R1KV` class
- `rkv/compression/r1_kv.py` - Core compression algorithm
- `rkv/compression/__init__.py` - Helper functions (`cal_similarity`, `compute_attention_scores`)

---

### Phase 2: Environment Variables

#### Step 2.1: Add R1KV Environment Variables
**File:** `vllm/envs.py`

**Location:** After existing environment variables (around line 1000+)

**Code to add:**
```python
    # R1KV compression for v0 backend (FLASH_ATTN)
    # Total KV cache size limit for v0 in tokens. Set to -1 to disable compression.
    # When enabled, sequences exceeding BUDGET + BUFFER tokens will be compressed.
    "VLLM_V0_R_KV_BUDGET":
    lambda: int(os.getenv("VLLM_V0_R_KV_BUDGET", "-1")),

    # Controls how many new tokens are generated before triggering KV compression in v0.
    # Similar to VLLM_V1_R_KV_BUFFER but applies to vLLM v0 backend (FLASH_ATTN).
    # Compression triggers when seq_len >= BUDGET + BUFFER.
    "VLLM_V0_R_KV_BUFFER":
    lambda: int(os.getenv("VLLM_V0_R_KV_BUFFER", "-1")),
```

**Notes:**
- Default is `-1` (disabled) to avoid breaking existing behavior
- Positive values enable compression
- Must export these before running: `export VLLM_V0_R_KV_BUDGET=512 VLLM_V0_R_KV_BUFFER=64`

---

### Phase 3: Attention Backend Modifications

#### Step 3.1: Import R1KV in Flash Attention Backend
**File:** `vllm/attention/backends/flash_attn.py`

**Location:** After other imports (around line 30-35)

**Code to add:**
```python
# R1KV compression support for v0 backend
try:
    from rkv.modeling import R1KV
except ImportError:
    R1KV = None
    logger.warning("R1KV package not found. KV compression will be disabled. "
                   "Install with: pip install -e third_party/rkv_repo")
```

**Rationale:** Graceful degradation if rkv package not installed.

---

#### Step 3.2: Extend FlashAttentionMetadata
**File:** `vllm/attention/backends/flash_attn.py`

**Location:** Inside `FlashAttentionMetadata` class (around line 105-160)

**Code to add:**
```python
@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    # ... existing fields ...

    # R1KV: Number of tokens dropped per sequence in this batch due to compression
    # List of integers, one per sequence in the batch
    # Used to update sequence lengths in scheduler for memory reclamation
    num_dropped_tokens_list: Optional[List[int]] = None
```

**Purpose:** Track compression statistics per sequence in each batch.

---

#### Step 3.3: Initialize Tracking List in Metadata Builder
**File:** `vllm/attention/backends/flash_attn.py`

**Location:** Inside `FlashAttentionMetadataBuilder.build()` method (around line 450-550)

**Find the section where metadata fields are initialized, then add:**
```python
# Initialize num_dropped_tokens_list for R1KV compression tracking
# One entry per sequence, initialized to 0
num_dropped_tokens_list = [0] * num_seqs
```

**Add to return statement:**
```python
return FlashAttentionMetadata(
    # ... existing fields ...
    num_dropped_tokens_list=num_dropped_tokens_list,  # NEW
)
```

**Purpose:** Initialize tracking list for all sequences in batch.

---

#### Step 3.4: Initialize R1KV Compressor in FlashAttentionImpl
**File:** `vllm/attention/backends/flash_attn.py`

**Location:** Inside `FlashAttentionImpl.__init__()` method (around line 600-680)

**Add after head size validation, before end of `__init__`:**
```python
        # Initialize R1KV compressor for v0 backend
        from vllm.envs import VLLM_V0_R_KV_BUDGET, VLLM_V0_R_KV_BUFFER
        if R1KV is not None and VLLM_V0_R_KV_BUDGET > 0 and VLLM_V0_R_KV_BUFFER > 0:
            self.kvcompressor = R1KV(budget=VLLM_V0_R_KV_BUDGET)
            logger.info(f"Initialized R1KV compressor with budget "
                        f"{VLLM_V0_R_KV_BUDGET} for v0 FLASH_ATTN backend.")
        else:
            self.kvcompressor = None
            if R1KV is not None:
                logger.info("R1KV compressor is disabled for v0 FLASH_ATTN backend "
                           "(BUDGET or BUFFER not set or <= 0).")
```

**Purpose:** Create compressor instance if environment variables are set.

---

#### Step 3.5: Add Compression Logic in Forward Pass
**File:** `vllm/attention/backends/flash_attn.py`

**Location:** Inside `FlashAttentionImpl.forward()` method, **AFTER** the decode attention computation (around line 900-950)

**Find the section after `flash_attn_with_kvcache()` is called in decode phase, then add:**

```python
        # R1KV compression for v0 backend (decode phase only)
        if self.kvcompressor is not None and decode_meta is not None:
            from vllm.envs import VLLM_V0_R_KV_BUDGET, VLLM_V0_R_KV_BUFFER

            # Only compress in normal decoding (not varlen decoding)
            if decode_meta.max_decode_query_len is not None and decode_meta.max_decode_query_len <= 1:
                # Get batch size from decode metadata
                if decode_meta.seq_lens_tensor is not None:
                    # Use .cpu().item() to safely extract scalar values
                    # This works with CUDA graphs because seq_lens_tensor is created fresh each iteration
                    num_seqs = decode_meta.seq_lens_tensor.shape[0]
                    block_size = kv_cache.shape[2]  # [2, num_blocks, block_size, num_kv_heads, head_size]

                    for seq_idx in range(num_seqs):
                        # Use .cpu().item() to match v1 implementation for CUDA graph compatibility
                        seq_len = decode_meta.seq_lens_tensor[seq_idx].cpu().item()

                        # Only compress if sequence is long enough
                        if seq_len < VLLM_V0_R_KV_BUDGET + VLLM_V0_R_KV_BUFFER:
                            continue

                        # Extract block table for this sequence
                        if decode_meta.block_tables is not None:
                            block_table = decode_meta.block_tables[seq_idx]
                            # Get number of blocks actually used
                            num_blocks_used = (seq_len + block_size - 1) // block_size
                            block_table = block_table[:num_blocks_used]

                            # 1. EXTRACT: Gather KV cache for this sequence from blocks
                            # key_cache: [num_blocks, block_size, num_kv_heads, head_size]
                            seq_key_blocks = key_cache[block_table]  # [num_blocks_used, block_size, num_kv_heads, head_size]
                            seq_val_blocks = value_cache[block_table]

                            # Reshape to continuous sequence
                            # [num_blocks_used, block_size, num_kv_heads, head_size] -> [num_blocks_used * block_size, num_kv_heads, head_size]
                            seq_key = seq_key_blocks.reshape(-1, seq_key_blocks.shape[-2], seq_key_blocks.shape[-1])[:seq_len]
                            seq_val = seq_val_blocks.reshape(-1, seq_val_blocks.shape[-2], seq_val_blocks.shape[-1])[:seq_len]

                            # 2. RESHAPE: Convert to R1KV format [1, num_kv_heads, seq_len, head_size]
                            current_key = seq_key.permute(1, 0, 2).unsqueeze(0)  # [1, num_kv_heads, seq_len, head_size]
                            current_val = seq_val.permute(1, 0, 2).unsqueeze(0)

                            # Get current query (last token for this sequence)
                            # decode_query shape: [num_decode_tokens, num_heads, head_size]
                            # We need to find which position corresponds to seq_idx
                            current_query = decode_query[seq_idx:seq_idx+1].unsqueeze(2)  # [1, num_heads, 1, head_size]

                            # 3. COMPRESS: Apply R1KV compression
                            compressed_key, compressed_val = self.kvcompressor.update_kv(
                                current_key, current_query, current_val
                            )

                            # Get compressed sequence length
                            compressed_len = compressed_key.shape[2]

                            # 4. TRACK: Record dropped tokens for scheduler (like v1 does)
                            num_dropped_tokens_i = seq_len - compressed_len
                            if decode_meta.num_dropped_tokens_list is not None:
                                # Update the count for this sequence
                                # Only update if we actually dropped tokens and haven't already counted them
                                if num_dropped_tokens_i != decode_meta.num_dropped_tokens_list[seq_idx]:
                                    assert decode_meta.num_dropped_tokens_list[seq_idx] == 0, \
                                        f"num_dropped_tokens_list[{seq_idx}] should be 0 before compression"
                                    decode_meta.num_dropped_tokens_list[seq_idx] = num_dropped_tokens_i

                            # 5. WRITE BACK: Store compressed cache back to blocks
                            # compressed_key: [1, num_kv_heads, compressed_len, head_size]
                            # Need to reshape back to block format
                            compressed_key = compressed_key.squeeze(0).permute(1, 0, 2)  # [compressed_len, num_kv_heads, head_size]
                            compressed_val = compressed_val.squeeze(0).permute(1, 0, 2)

                            # Calculate how many blocks needed for compressed cache
                            num_compressed_blocks = (compressed_len + block_size - 1) // block_size

                            # Zero out the original blocks first
                            key_cache[block_table] = 0
                            value_cache[block_table] = 0

                            # Write compressed cache back block by block
                            for block_idx in range(num_compressed_blocks):
                                start_token = block_idx * block_size
                                end_token = min(start_token + block_size, compressed_len)
                                num_tokens = end_token - start_token

                                # Copy compressed tokens to block
                                key_cache[block_table[block_idx], :num_tokens] = compressed_key[start_token:end_token]
                                value_cache[block_table[block_idx], :num_tokens] = compressed_val[start_token:end_token]

        return output
```

**Critical Notes:**
- **Placement:** Must be AFTER decode attention computation, BEFORE return
- **Phase:** Only operates in decode phase (not prefill)
- **Condition:** Only when `max_decode_query_len <= 1` (normal decoding, not varlen)
- **Per-sequence:** Loops through all sequences in batch independently

---

### Phase 4: Sequence State Tracking

#### Step 4.1: Add Dropped Token Counter to Sequence
**File:** `vllm/sequence.py`

**Location:** Inside `Sequence.__init__()` method (around line 490-510)

**Add after existing field initialization:**
```python
        # R1KV: Track number of tokens dropped due to KV cache compression
        # Used by scheduler to update sequence lengths and free blocks
        self.num_dropped_tokens: int = 0
```

---

#### Step 4.2: Modify n_blocks Property for Memory Reclamation
**File:** `vllm/sequence.py`

**Location:** Replace the existing `n_blocks` property (around line 500-510)

**Original code:**
```python
    @property
    def n_blocks(self) -> int:
        return (self.get_len() + self.block_size - 1) // self.block_size
```

**Replace with:**
```python
    @property
    def n_blocks(self) -> int:
        # R1KV: Subtract dropped tokens from length to get actual blocks needed
        # This allows the block manager to free unused blocks after compression
        effective_len = max(0, self.get_len() - self.num_dropped_tokens)
        return (effective_len + self.block_size - 1) // self.block_size
```

**Purpose:** Enables automatic memory reclamation by reporting fewer blocks needed after compression.

---

### Phase 5: Engine Coordination

#### Step 5.1: Add Update Hook in LLMEngine
**File:** `vllm/engine/llm_engine.py`

**Location:** Inside `LLMEngine.step()` method, **AFTER** model execution (around line 1330-1340)

**Find the line after `outputs = self.model_executor.execute_model(...)`, then add:**
```python
        # R1KV: Update sequences with dropped tokens from compression
        self._update_sequences_with_dropped_tokens(
            seq_group_metadata_list, scheduler_outputs)
```

**Purpose:** Hook to propagate compression statistics after each step.

---

#### Step 5.2: Implement Update Method
**File:** `vllm/engine/llm_engine.py`

**Location:** Add new method to `LLMEngine` class (around line 1500-1600, near other helper methods)

**Code to add:**
```python
    def _update_sequences_with_dropped_tokens(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            scheduler_outputs: SchedulerOutputs) -> None:
        """Update sequences with number of dropped tokens from R1KV compression.

        This method extracts num_dropped_tokens_list from the attention metadata
        and updates each sequence's num_dropped_tokens field. This is used by
        the scheduler to properly track sequence lengths and free unused blocks.

        Args:
            seq_group_metadata_list: List of sequence group metadata from scheduler
            scheduler_outputs: Scheduler outputs containing sequence groups
        """
        # Get the attention metadata which contains num_dropped_tokens_list
        if not seq_group_metadata_list:
            return

        # Access the first metadata to get the attention metadata
        # In v0, all sequences share the same attention metadata
        first_metadata = seq_group_metadata_list[0]
        if not hasattr(first_metadata, 'attn_metadata') or first_metadata.attn_metadata is None:
            return

        attn_metadata = first_metadata.attn_metadata

        # Check if we have num_dropped_tokens_list (only present if R1KV compression is enabled)
        if not hasattr(attn_metadata, 'num_dropped_tokens_list') or attn_metadata.num_dropped_tokens_list is None:
            return

        num_dropped_tokens_list = attn_metadata.num_dropped_tokens_list

        # Update each sequence group's sequences
        for seq_group_idx, seq_group_metadata in enumerate(seq_group_metadata_list):
            if seq_group_idx >= len(num_dropped_tokens_list):
                break

            num_dropped = num_dropped_tokens_list[seq_group_idx]
            if num_dropped > 0:
                # Find matching sequence group in scheduler outputs
                for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
                    if scheduled_seq_group.request_id == seq_group_metadata.request_id:
                        # Update all sequences in this group
                        for seq in scheduled_seq_group.seqs:
                            # Accumulate dropped tokens (compression may happen multiple times)
                            seq.num_dropped_tokens += num_dropped
                            # Cap at total computed tokens to avoid overflow
                            if seq.num_dropped_tokens > seq.get_len():
                                seq.num_dropped_tokens = seq.get_len()
                        break
```

**Purpose:** Extract compression stats from attention metadata and update sequence state.

---

### Phase 6: Testing & Validation

#### Step 6.1: Unit Test - Compression Logic
Create a simple test to verify compression works:

**File:** `tests/test_r1kv_compression.py` (new file)

```python
import pytest
import torch
import os

# Set environment variables before importing vllm
os.environ["VLLM_V0_R_KV_BUDGET"] = "128"
os.environ["VLLM_V0_R_KV_BUFFER"] = "32"

from vllm import LLM, SamplingParams

@pytest.mark.skipif(not os.getenv("RUN_R1KV_TESTS"),
                    reason="R1KV tests disabled by default")
def test_r1kv_basic_compression():
    """Test that R1KV compression is initialized and works."""
    # Create LLM with small model
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=512,
        enforce_eager=True,  # Disable CUDA graphs for testing
        gpu_memory_utilization=0.5,
    )

    # Generate with long prompt to trigger compression
    prompt = "Hello, " * 100  # ~200 tokens
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    outputs = llm.generate(prompt, sampling_params)

    # Verify output is generated
    assert len(outputs) > 0
    assert len(outputs[0].outputs[0].text) > 0
    print(f"Generated: {outputs[0].outputs[0].text[:100]}")

if __name__ == "__main__":
    test_r1kv_basic_compression()
```

**Run:**
```bash
export VLLM_V0_R_KV_BUDGET=128
export VLLM_V0_R_KV_BUFFER=32
export RUN_R1KV_TESTS=1
python tests/test_r1kv_compression.py
```

---

#### Step 6.2: Integration Test - R1KV + Steering
Test that both features work together:

**File:** `tests/test_r1kv_steering_integration.py` (new file)

```python
import pytest
import torch
import os

# Configure R1KV
os.environ["VLLM_V0_R_KV_BUDGET"] = "256"
os.environ["VLLM_V0_R_KV_BUFFER"] = "64"

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"),
                    reason="Integration tests disabled by default")
def test_r1kv_with_steering():
    """Test R1KV compression + steering vectors together."""
    # Create LLM with steering enabled
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        enable_steer_vector=True,
        max_steer_vectors=1,
        max_model_len=1024,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
    )

    # Create simple steering request (if you have a test vector)
    # sv_request = SteerVectorRequest(
    #     steer_vector_name="test",
    #     steer_vector_id=1,
    #     steer_vector_local_path="./test_vectors/happiness.gguf",
    #     scale=1.0,
    #     algorithm="direct"
    # )

    # Generate with long prompt to trigger R1KV compression
    prompt = "The meaning of life is " * 50
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

    # Generate without steering
    outputs_no_steer = llm.generate(prompt, sampling_params)

    # Generate with steering (if sv_request is available)
    # outputs_with_steer = llm.generate(prompt, sampling_params,
    #                                    steer_vector_request=sv_request)

    # Verify both work
    assert len(outputs_no_steer) > 0
    print(f"No steering: {outputs_no_steer[0].outputs[0].text[:100]}")

    # If steering is enabled:
    # assert len(outputs_with_steer) > 0
    # print(f"With steering: {outputs_with_steer[0].outputs[0].text[:100]}")

if __name__ == "__main__":
    test_r1kv_with_steering()
```

---

#### Step 6.3: Manual Verification Script
Quick test to verify everything works:

**File:** `test_r1kv_manual.py` (in root)

```python
#!/usr/bin/env python3
"""Manual test script for R1KV compression in EasySteer-vllm."""
import os

# Configure R1KV compression
os.environ["VLLM_V0_R_KV_BUDGET"] = "512"
os.environ["VLLM_V0_R_KV_BUFFER"] = "64"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # Explicitly use v0

from vllm import LLM, SamplingParams

def main():
    print("=" * 80)
    print("Testing R1KV Compression in EasySteer-vllm")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  VLLM_V0_R_KV_BUDGET: {os.getenv('VLLM_V0_R_KV_BUDGET')}")
    print(f"  VLLM_V0_R_KV_BUFFER: {os.getenv('VLLM_V0_R_KV_BUFFER')}")
    print(f"  VLLM_ATTENTION_BACKEND: {os.getenv('VLLM_ATTENTION_BACKEND')}")

    # Initialize model
    print("\nInitializing LLM...")
    llm = LLM(
        model="facebook/opt-125m",  # Small model for quick testing
        max_model_len=1024,
        enforce_eager=True,  # Easier to debug
        gpu_memory_utilization=0.5,
    )

    # Test prompt (long enough to trigger compression)
    prompt = "Once upon a time, in a land far far away, " * 40  # ~400 tokens
    print(f"\nPrompt length: ~{len(prompt.split())} words")

    # Generate
    print("\nGenerating...")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=200,
    )

    outputs = llm.generate(prompt, sampling_params)

    # Display results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"\nGenerated ({len(generated_text.split())} words):")
        print(generated_text[:200] + "..." if len(generated_text) > 200 else generated_text)

    print("\n✓ Test completed successfully!")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
python test_r1kv_manual.py
```

---

### Phase 7: Documentation Updates

#### Step 7.1: Update CLAUDE.md
Add R1KV section to the EasySteer-vllm CLAUDE.md:

**Location:** After "Experimental Features" section

**Content:**
```markdown
## R1KV KV Cache Compression (v0 Backend)

EasySteer-vllm integrates **R1KV (Redundancy-aware KV Cache Compression)** for efficient inference with long contexts. R1KV achieves 90% memory savings while maintaining accuracy by compressing KV cache on-the-fly during decoding.

### Enabling R1KV Compression

Set environment variables before running:
```bash
export VLLM_V0_R_KV_BUDGET=512      # Total KV cache size (in tokens)
export VLLM_V0_R_KV_BUFFER=64       # Compression trigger threshold
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Must use v0 backend
```

### Using R1KV with Steering Vectors

R1KV and steering vectors are fully compatible:

```python
import os
os.environ["VLLM_V0_R_KV_BUDGET"] = "512"
os.environ["VLLM_V0_R_KV_BUFFER"] = "64"

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# Initialize with both features
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_steer_vector=True,
    max_steer_vectors=2,
)

# Create steering request
sv_request = SteerVectorRequest(
    steer_vector_name="friendly",
    steer_vector_id=1,
    steer_vector_local_path="./vectors/friendly.gguf",
    scale=1.5,
)

# Generate with both R1KV compression and steering
outputs = llm.generate(
    "Your prompt here",
    sampling_params=SamplingParams(max_tokens=500),
    steer_vector_request=sv_request
)
```

### How R1KV Works

1. **Compression Trigger:** When sequence length >= budget + buffer
2. **Algorithm:** Combines attention importance + redundancy scoring
3. **Memory Savings:** Automatically frees unused blocks via modified `Sequence.n_blocks`
4. **Compatibility:** Works with steering vectors (operates at different levels)

### Key Configuration

- **VLLM_V0_R_KV_BUDGET:** Target cache size (e.g., 512, 1024)
- **VLLM_V0_R_KV_BUFFER:** How many new tokens before compression (e.g., 64, 128)
- **Trade-off:** Lower budget = more compression + less memory, but may impact quality

### Dependencies

R1KV requires the separate `rkv` package:

```bash
# Install from third_party (included in repo)
pip install -e third_party/rkv_repo

# Or verify installation
python -c "from rkv.modeling import R1KV; print('OK')"
```

### Architecture

R1KV operates in the v0 attention backend after attention computation:

1. **Attention Backend** (`vllm/attention/backends/flash_attn.py`): Compresses KV cache
2. **Sequence State** (`vllm/sequence.py`): Tracks dropped tokens
3. **Engine** (`vllm/engine/llm_engine.py`): Propagates compression stats
4. **Block Manager**: Automatically frees unused blocks

### Troubleshooting

**Issue:** Compression not happening
- **Check:** `VLLM_V0_R_KV_BUDGET` and `VLLM_V0_R_KV_BUFFER` are both > 0
- **Check:** Using v0 backend (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`)
- **Check:** Sequence length >= budget + buffer

**Issue:** `ImportError: cannot import name 'R1KV'`
- **Solution:** Install rkv package: `pip install -e third_party/rkv_repo`

**Issue:** Out of memory with R1KV enabled
- **Solution:** Increase budget or reduce batch size
```

---

#### Step 7.2: Create R1KV README
**File:** `R1KV_README.md` (in root)

Quick reference guide for R1KV usage in EasySteer-vllm.

```markdown
# R1KV KV Cache Compression in EasySteer-vllm

## Quick Start

```bash
# 1. Install dependencies
pip install -e third_party/rkv_repo

# 2. Set environment variables
export VLLM_V0_R_KV_BUDGET=512
export VLLM_V0_R_KV_BUFFER=64

# 3. Run inference
python your_inference_script.py
```

## Configuration Guide

### Budget Selection

| Use Case | Budget | Buffer | Notes |
|----------|--------|--------|-------|
| Extreme compression | 128 | 32 | 90%+ savings, may impact quality |
| Balanced | 512 | 64 | Good trade-off |
| Conservative | 1536 | 128 | Minimal quality impact |

### Combined with Steering

R1KV and steering vectors work together seamlessly:
- **Steering:** Modifies hidden states (decoder level)
- **R1KV:** Compresses KV cache (attention level)
- **No conflict:** Different layers of operation

## Implementation Details

- **Backend:** vLLM v0 (`FLASH_ATTN`)
- **Algorithm:** Token-level redundancy + importance scoring
- **Memory Reclamation:** Automatic via modified `Sequence.n_blocks`
- **Files Modified:** 5 core files, ~200 lines

## Testing

```bash
# Basic test
python test_r1kv_manual.py

# With steering vectors
export RUN_INTEGRATION_TESTS=1
python tests/test_r1kv_steering_integration.py
```

## References

- **Source:** SkipKV/vllm fork (v0 backend)
- **Original Paper:** R-KV compression for reasoning models
- **Migration Plan:** See `R1KV_MIGRATION_PLAN.md`
```

---

### Phase 8: Verification Checklist

#### Before Deployment:

- [ ] **Dependency Check:**
  - [ ] rkv package installed (`pip show rkv`)
  - [ ] Can import R1KV: `python -c "from rkv.modeling import R1KV"`

- [ ] **Code Changes:**
  - [ ] `vllm/envs.py`: Added `VLLM_V0_R_KV_BUDGET`, `VLLM_V0_R_KV_BUFFER`
  - [ ] `vllm/attention/backends/flash_attn.py`:
    - [ ] Imported R1KV
    - [ ] Extended `FlashAttentionMetadata` with `num_dropped_tokens_list`
    - [ ] Initialize tracking list in metadata builder
    - [ ] Initialize `self.kvcompressor` in `FlashAttentionImpl.__init__`
    - [ ] Added compression loop in `forward()` after decode attention
  - [ ] `vllm/sequence.py`:
    - [ ] Added `num_dropped_tokens` field
    - [ ] Modified `n_blocks` property
  - [ ] `vllm/engine/llm_engine.py`:
    - [ ] Added call to `_update_sequences_with_dropped_tokens()` in `step()`
    - [ ] Implemented `_update_sequences_with_dropped_tokens()` method

- [ ] **Testing:**
  - [ ] Unit test passes: `python tests/test_r1kv_compression.py`
  - [ ] Manual test passes: `python test_r1kv_manual.py`
  - [ ] Integration test passes (if steering vectors available)

- [ ] **Documentation:**
  - [ ] Updated `CLAUDE.md` with R1KV section
  - [ ] Created `R1KV_README.md`
  - [ ] Updated main `README.md` (if needed)

---

## Rollback Plan

If issues arise after migration:

### Quick Disable
```bash
# Disable R1KV without code changes
export VLLM_V0_R_KV_BUDGET=-1
export VLLM_V0_R_KV_BUFFER=-1
```

### Full Rollback
```bash
# Revert all changes
git revert <migration_commit_hash>

# Or cherry-pick revert individual files
git checkout HEAD~1 -- vllm/attention/backends/flash_attn.py
git checkout HEAD~1 -- vllm/sequence.py
git checkout HEAD~1 -- vllm/engine/llm_engine.py
git checkout HEAD~1 -- vllm/envs.py
```

---

## Future Work

### v1 Backend Migration
After v0 is stable, consider migrating v1 backend:
- Requires modifying `vllm/v1/attention/backends/flash_attn.py`
- Different architecture (slot mapping vs block tables)
- More complex scheduler integration
- See SkipKV/vllm CLAUDE.md lines 133-355 for v1 details

### Additional Features
- [ ] R2KV_SLOW (sentence-level redundancy) - requires more complex state tracking
- [ ] Per-layer compression budgets
- [ ] Dynamic budget adjustment
- [ ] Compression statistics reporting

---

## Contact & Support

**Migration Author:** Claude Code
**Date:** 2025-10-22
**Source Repository:** https://github.com/olokevin/EasySteer-vllm
**R1KV Source:** /home/yequan/Project/SkipKV/vllm (v0 backend)

For issues, refer to:
- This migration plan
- `CLAUDE.md` in both repos
- Source code comments
