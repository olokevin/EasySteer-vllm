# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EasySteer-vllm** is a fork of vLLM that adds **Steering Vector** support for controllable text generation AND **R1KV KV Cache Compression** for memory-efficient inference.

**Key Features:**
- **Steering Vectors:** Fine-grained control over model behavior via hidden state intervention
  - Multiple algorithms: Direct, LoReFT, Linear Transformation, LM-Steer, Multi-Vector
  - Per-layer and per-token control
  - Prefill and generation phase steering
  - Multi-vector composition with conflict resolution
- **R1KV Compression:** 90% memory savings via redundancy-aware KV cache compression
  - Token-level redundancy detection via cosine similarity
  - Importance scoring via attention weights
  - Automatic memory reclamation
  - **Compatible with steering vectors** (operates at different levels)
- Compatible with vLLM's distributed inference and quantization features

**Repository:** https://github.com/olokevin/EasySteer-vllm

## Installation and Setup

### Build from Source

```bash
# Install dependencies
pip install -r requirements/common.txt
pip install -r requirements/cuda.txt  # for CUDA GPUs

# Build and install vLLM with steering vector support
pip install -e .
```

### Requirements
- Python 3.9-3.12
- PyTorch 2.7.0
- CUDA 12.8 (for GPU support)
- CMake >= 3.26.1

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/basic_correctness/

# Run with specific markers
pytest -m "not distributed"

# Note: Some tests require distributed GPU setup
# The test suite may have import errors in some environments - this is expected
```

## Architecture Overview

### Core Components

#### 1. Steering Vector System (`vllm/steer_vectors/`)

**Main Modules:**
- `models.py` - SteerVectorModel and SteerVectorModelManager
- `layers.py` - DecoderLayerWithSteerVector wrapper for injecting interventions
- `request.py` - SteerVectorRequest and VectorConfig for per-request configuration
- `worker_manager.py` - Manages steering vectors across distributed workers

**Algorithm Framework** (`vllm/steer_vectors/algorithms/`):
- `base.py` - BaseSteerVectorAlgorithm interface
- `template.py` - AlgorithmTemplate with common functionality
- `factory.py` - Registry and factory for algorithm creation
- **Implementations:**
  - `direct.py` - Direct vector addition (default, simplest)
  - `loreft.py` - LoReFT (Low-Rank Efficient Fine-Tuning) intervention
  - `linear.py` - Linear transformation of hidden states
  - `lm_steer.py` - LM-Steer algorithm
  - `multi_vector.py` - Composition of multiple vectors

#### 2. Integration Points

**Model Loading:**
- Steering vectors wrap decoder layers automatically when `SteerVectorConfig` is provided
- Supported layer types listed in `_decoder_layer_class_names` (vllm/steer_vectors/models.py:42-62)
- Includes Llama, Qwen2, Mistral, Gemma, Phi, DeepSeek, and many more

**Request Processing:**
- Each inference request can specify its own steering vector configuration
- Supports both single-vector and multi-vector modes
- Configuration includes trigger tokens, positions, and exclusions

**Configuration:**
```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# Initialize LLM with steering vector support
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_steer_vector=True,
    max_steer_vectors=4
)

# Create steering vector request
sv_request = SteerVectorRequest(
    steer_vector_name="happiness",
    steer_vector_id=1,
    steer_vector_local_path="path/to/vector.gguf",
    scale=2.0,
    target_layers=[10, 11, 12],  # Apply to specific layers
    prefill_trigger_positions=[-1],  # Apply to last token in prefill
    algorithm="direct"
)
```

### Key Design Patterns

#### Layer Wrapping System
The steering vector system uses a **wrapper pattern** to inject interventions:
1. Original decoder layers are wrapped with `DecoderLayerWithSteerVector`
2. Wrapper intercepts forward pass and applies steering algorithms
3. Algorithms can modify hidden states at decoder or MLP sublayer level

#### Algorithm Plugin System
New algorithms can be registered using the decorator pattern:
```python
from vllm.steer_vectors.algorithms import register_algorithm, AlgorithmTemplate

@register_algorithm("my_algorithm")
class MyAlgorithm(AlgorithmTemplate):
    def apply_intervention(self, hidden_states):
        # Custom intervention logic
        return modified_hidden_states
```

#### Multi-Vector Composition
Multiple vectors can be applied with conflict resolution:
- `"error"` - Raise error on conflicts
- `"priority"` - Use first vector, ignore others
- `"sequential"` - Apply all vectors in sequence (effects stack)

### File Loading Support

Steering vectors can be loaded from multiple formats:
- **GGUF files** - Standard format for control vectors (`.gguf`)
- **PyTorch files** - Direct tensor files (`.pt`)
- **ReFT directories** - LoReFT checkpoints

## Development Workflow

### Adding a New Steering Algorithm

1. **Create algorithm class** in `vllm/steer_vectors/algorithms/`:
```python
from .template import AlgorithmTemplate
from .factory import register_algorithm

@register_algorithm("my_algo")
class MyAlgorithm(AlgorithmTemplate):
    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs):
        # Load vector data from file
        return {"layer_payloads": {...}}

    def apply_intervention(self, hidden_states: torch.Tensor):
        # Apply your intervention logic
        return hidden_states + self.active_vector
```

2. **Register in `__init__.py`**:
```python
from .my_algo import MyAlgorithm
__all__ = [..., "MyAlgorithm"]
```

3. **Test the algorithm**:
```python
sv_request = SteerVectorRequest(
    algorithm="my_algo",
    ...
)
```

### Code Formatting and Linting

```bash
# Format code
./format.sh

# Run type checking
./tools/mypy.sh

# Run shell script linting
./tools/shellcheck.sh
```

### Understanding the Request Flow

1. **Request Creation:** User creates `SteerVectorRequest` with vector path and config
2. **Model Loading:** `SteerVectorModel.from_local_checkpoint()` loads vector file
3. **Layer Injection:** Decoder layers are wrapped with `DecoderLayerWithSteerVector`
4. **Algorithm Setup:** Vector algorithm is instantiated and configured per layer
5. **Inference:** During forward pass, algorithm applies intervention based on triggers
6. **Token Filtering:** Trigger tokens/positions determine which tokens receive intervention

### Trigger System

**Prefill Phase Triggers:**
- `prefill_trigger_tokens: [token_id, ...]` - Apply to specific tokens
- `prefill_trigger_tokens: [-1]` - Apply to ALL prefill tokens
- `prefill_trigger_positions: [0, -1]` - Apply to first and last positions
- `prefill_exclude_tokens/positions` - Exclude specific tokens (higher priority)

**Generation Phase Triggers:**
- `generate_trigger_tokens: [-1]` - Apply to ALL generated tokens
- `generate_trigger_tokens: [token_id, ...]` - Apply to specific tokens

### Multi-GPU and Distributed Inference

Steering vectors are automatically synchronized across distributed workers:
- Vector models are broadcast to all workers during initialization
- Each worker maintains its own copy of the steering vector state
- Compatible with tensor parallelism and pipeline parallelism

## Common Development Tasks

### Testing a New Algorithm on a Model

```bash
# Start Python interpreter
python

# Import and test
from vllm import LLM
from vllm.steer_vectors.request import SteerVectorRequest

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_steer_vector=True,
    max_steer_vectors=1,
    enforce_eager=True  # Disable CUDA graphs for debugging
)

sv_req = SteerVectorRequest(
    steer_vector_name="test",
    steer_vector_id=1,
    steer_vector_local_path="./vectors/happiness.gguf",
    algorithm="direct",
    scale=1.5,
    debug=True  # Enable debug output
)

outputs = llm.generate(
    "Hello, how are you?",
    sampling_params=SamplingParams(max_tokens=50),
    steer_vector_request=sv_req
)
```

### Debugging Layer Wrapping

Check which layers are wrapped:
```python
# After model initialization
for name, module in llm.llm_engine.model_executor.driver_worker.model_runner.model.named_modules():
    if "DecoderLayerWithSteerVector" in str(type(module)):
        print(f"Wrapped: {name}")
```

### Creating a Steering Vector File

Steering vectors are typically created through contrastive training or activation engineering:

```python
import torch
import gguf

# Example: Create a simple steering vector
vector = torch.randn(4096)  # Hidden size of model

# Save as PyTorch file
torch.save(vector, "my_vector.pt")

# Or save as GGUF (requires gguf library)
writer = gguf.GGUFWriter("my_vector.gguf", "steervector")
writer.add_tensor("vector.0", vector.numpy())  # layer 0
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

## Key Configuration Classes

**SteerVectorConfig** (vllm/config.py:3089):
- `max_steer_vectors: int` - Maximum number of concurrent steering vectors
- `adapter_dtype: torch.dtype` - Data type for adapter weights (default: float16)

**SteerVectorRequest** (vllm/steer_vectors/request.py:42):
- Single-vector mode for simple cases
- Multi-vector mode with `vector_configs` list
- Per-vector trigger configuration

## Important Notes

- Steering vectors are applied **after** the main computation in decoder layers
- Algorithm choice affects both memory usage and intervention strength
- `debug=True` in SteerVectorRequest enables detailed logging during forward pass
- Layer indices are 0-based (layer 0 is the first transformer layer)
- Negative position indices are relative to sequence end (-1 = last token)
- When using multi-vector mode, ensure vectors target different positions or use appropriate conflict resolution

## Related vLLM Documentation

Since this is a vLLM fork, standard vLLM documentation applies for:
- Model loading and quantization
- Distributed inference setup
- API server deployment
- See: https://docs.vllm.ai/

## R1KV KV Cache Compression (v0 Backend)

EasySteer-vllm now integrates **R1KV (Redundancy-aware KV Cache Compression)** for memory-efficient long-context inference. R1KV achieves up to 90% memory savings while maintaining accuracy by compressing KV cache on-the-fly during decoding.

### Quick Start

```bash
# Enable R1KV compression
export VLLM_V0_R_KV_BUDGET=512      # Total KV cache size (in tokens)
export VLLM_V0_R_KV_BUFFER=64       # Compression trigger threshold
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Must use v0 backend

# Run inference (compression happens automatically)
python your_script.py
```

### Using R1KV with Steering Vectors

R1KV and steering vectors are **fully compatible** and work together seamlessly:

```python
import os
# Configure R1KV BEFORE importing vllm
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

# Generate with BOTH R1KV compression AND steering
# Compression triggers automatically when seq_len >= 576 tokens (512+64)
outputs = llm.generate(
    "Your long prompt here",
    sampling_params=SamplingParams(max_tokens=500),
    steer_vector_request=sv_request
)
```

### How R1KV Works

1. **Compression Trigger:** Activates when `sequence_length >= budget + buffer`
2. **Algorithm:**
   - Computes **importance scores** via attention weights (how relevant each token is)
   - Computes **redundancy scores** via cosine similarity (how similar each token is to others)
   - Combines scores: `final_score = λ·importance - (1-λ)·redundancy`
   - Keeps top-k tokens with highest scores
3. **Memory Reclamation:** Automatically frees unused blocks via modified `Sequence.n_blocks` property
4. **Compatibility:** Works with steering vectors (operates at attention backend level, steering operates at decoder layer level)

### Configuration

**Environment Variables:**

- `VLLM_V0_R_KV_BUDGET` (int, default: -1)
  - Total KV cache size limit in tokens
  - Set to positive value to enable compression
  - Example values:
    - `128` - Extreme compression (90%+ savings, may impact quality)
    - `512` - Balanced (recommended for most use cases)
    - `1536` - Conservative (minimal quality impact)

- `VLLM_V0_R_KV_BUFFER` (int, default: -1)
  - Number of new tokens generated before triggering compression
  - Compression triggers when `seq_len >= BUDGET + BUFFER`
  - Example values:
    - `32` - Compress frequently (more aggressive)
    - `64` - Balanced (recommended)
    - `128` - Compress less often (more conservative)

**Disabling Compression:**
```bash
# Set either variable to -1
export VLLM_V0_R_KV_BUDGET=-1
# or
export VLLM_V0_R_KV_BUFFER=-1
```

### Architecture

R1KV is integrated into the v0 attention backend through 4 coordinated components:

1. **Attention Backend** (`vllm/attention/backends/flash_attn.py`):
   - Imports R1KV compressor
   - Extends metadata with `num_dropped_tokens_list`
   - Compresses KV cache after decode attention
   - Tracks dropped tokens per sequence

2. **Sequence State** (`vllm/sequence.py`):
   - Tracks `num_dropped_tokens` per sequence
   - Modifies `n_blocks` property to report compressed size

3. **Engine Coordination** (`vllm/engine/llm_engine.py`):
   - Propagates compression stats from attention → sequences
   - Updates sequence state after each step

4. **Block Manager** (automatic):
   - Queries `Sequence.n_blocks` for allocation
   - Automatically frees unused blocks

**Data Flow:**
```
Decode Attention
    ↓
R1KV Compression (if seq_len >= budget + buffer)
    ↓
Update num_dropped_tokens_list in metadata
    ↓
LLMEngine._update_sequences_with_dropped_tokens()
    ↓
seq.num_dropped_tokens += dropped
    ↓
seq.n_blocks returns compressed size
    ↓
Block Manager frees unused blocks → Memory Saved!
```

### Testing

Run the comprehensive test suite:

```bash
# Activate environment
conda activate easysteer

# Run all R1KV tests
python test_r1kv.py
```

The test suite verifies:
1. Basic R1KV compression works
2. R1KV and Steering Vectors coexist
3. Compression can be enabled/disabled

### Troubleshooting

**Issue:** Compression not happening
- **Check:** `VLLM_V0_R_KV_BUDGET` and `VLLM_V0_R_KV_BUFFER` are both > 0
- **Check:** Using v0 backend (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`)
- **Check:** Sequence length >= budget + buffer

**Issue:** `ImportError: cannot import name 'R1KV'`
- **Solution:** R1KV package already installed in `third_party/rkv_repo`
- **Verify:** `python -c "from rkv.compression import R1KV; print('OK')"`

**Issue:** Out of memory with R1KV enabled
- **Solution:** Increase budget or reduce batch size
- **Try:** Start with `BUDGET=1536` and decrease gradually

**Issue:** Quality degradation
- **Solution:** Increase budget (less compression)
- **Recommended:** Start with `BUDGET=512` for balanced performance

### Key Implementation Files

- `vllm/envs.py`: Environment variable definitions (lines 961-968)
- `vllm/attention/backends/flash_attn.py`: Core compression logic
  - Line 35: Import R1KV
  - Lines 163-166: Metadata extension
  - Lines 540-542: Tracking list initialization
  - Lines 681-690: Compressor initialization
  - Lines 953-1040: Compression execution loop
- `vllm/sequence.py`: State tracking (lines 511, 515-518)
- `vllm/engine/llm_engine.py`: Engine coordination (lines 1373-1374, 1580-1629)

### Why R1KV + Steering Vectors Work Together

- **Steering Vectors:** Modify hidden states at **decoder layer level** (before attention)
- **R1KV:** Compresses KV cache at **attention backend level** (after attention)
- **No Conflict:** Different layers of the stack, independent operation
- **Combined Benefit:** Memory-efficient controllable text generation

### Performance Characteristics

**Memory Savings:**
- Budget 128: ~90% savings (extreme)
- Budget 512: ~70% savings (balanced)
- Budget 1536: ~50% savings (conservative)

**Speed:**
- Compression overhead: ~2-5% per token
- Overall throughput improvement: Up to 2-3× (due to memory savings enabling larger batches)

**Quality:**
- Budget >= 512: Minimal impact on most tasks
- Budget < 512: May impact long-context reasoning

## Experimental Features

The repository includes experimental integration with:
- Vision-language models (VLM support)
- Additional steering algorithms (check recent commits for updates)

Check git history for latest features:
```bash
git log --oneline --all --grep="feat:\|steer" -20
```
