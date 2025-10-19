import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 1000

# Enable the v0 FlashAttention backend + R-KV compression
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_V1_R_KV_BUDGET", "1536")
os.environ.setdefault("VLLM_V1_R_KV_BUFFER", "128")
os.environ.setdefault("VLLM_RKV_METHOD", "rkv2_slow")
# os.environ.setdefault("VLLM_RKV_METHOD_CONFIG", '{"mix_lambda": 0.07, "retain_ratio": 0.1}')
os.environ.setdefault("EASYSTEER_ENABLE_KV_SHARING", "0")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_steer_request() -> SteerVectorRequest:
    """Construct a simple steer-vector request using the existing SEAL data."""
    steer_root = os.path.join(os.path.dirname(__file__), "seal", "vectors")
    baseline_vec = os.path.join(steer_root, "baseline.vec")

    return SteerVectorRequest(
        steer_vector_name="baseline_control",
        steer_vector_id=1,
        steer_vector_local_path=baseline_vec,
        scale=1.0,
        algorithm="direct",
        target_layers=None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts: List[str] = [
        "You are given a math problem. Problem: If f(x) = (3x-2)/(x-2), "
        "what is the value of f(-2)+f(-1)+f(0)? Express your answer as a common fraction. "
        "Solve step by step with a brief explanation, then provide the final answer "
        "in the format: Final answer: \\boxed{}"
    ]

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    
    llm = LLM(
        model=MODEL_NAME,
        ### RKV
        enable_prefix_caching=False,
        ### SEAL
        enable_steer_vector=True,
        enforce_eager=True,
        tensor_parallel_size=1
    )

    steer_request = build_steer_request()

    outputs = llm.generate(
        prompts,
        sampling_params,
        steer_vector_request=steer_request,
    )

    for output in outputs:
        generated = output.outputs[0]
        print(f"Prompt: {output.prompt!r}\nGenerated Text:\n{generated.text}\n")


if __name__ == "__main__":
    main()
