import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

from metrics_tracker import GenerationMetricsTracker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

TEMPERATURE = 0
TOP_P = 0.95
MAX_TOKENS = 8192

### v0 backend
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

# os.environ["VLLM_V0_R_KV_BUDGET"] = "-1"
# os.environ["VLLM_V0_R_KV_BUFFER"] = "-1"

os.environ["VLLM_V0_R_KV_BUDGET"] = "1536"
os.environ["VLLM_V0_R_KV_BUFFER"] = "128"

### v1 backend
# os.environ["VLLM_USE_V1"] = "1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"

# # os.environ["VLLM_V1_R_KV_BUDGET"] = "-1"
# # os.environ["VLLM_V1_R_KV_BUFFER"] = "-1"

# os.environ["VLLM_V1_R_KV_BUDGET"] = "768"
# os.environ["VLLM_V1_R_KV_BUFFER"] = "128"

# os.environ.setdefault("VLLM_RKV_METHOD_CONFIG", '{"mix_lambda": 0.07, "retain_ratio": 0.1}')
os.environ.setdefault("EASYSTEER_ENABLE_KV_SHARING", "0")


os.environ["VLLM_RKV_METHOD"] = "rkv"
# os.environ["VLLM_RKV_METHOD"] = "rkv2_slow"

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

    ### single prompt
    # examples = ['You are given a math problem. Problem: If $f(x) = \frac{3x-2}{x-2}$, what is the value of $f(-2)+f(-1)+f(0)$? Express your answer as a common fraction. Solve step by step with a brief explanation, then provide the final answer in the format: Final answer: \boxed{}',]
    # answers = ['\\boxed{\\frac{14}{3}}',]
    
    import json
    problems = []
    answers = []

    # file_path = "/home/yequan/Project/R-KV/HuggingFace/data/math.jsonl"
    # with open(file_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         item = json.loads(line)
    #         problems.append(item["problem"])
    #         answers.append(item["answer"])

    file_path = "/home/yequan/Project/R-KV/HuggingFace/data/gsm8k.jsonl"     
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            problems.append(item["question"])
            answers.append(item["answer"])

    examples = ["Please reason step by step, and put your final answer within \\boxed{}.\nUser: " + prompt + "\nAssistant: <think>" for prompt in problems]

    ### Setup LLM and Sampling Parameters
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
        tensor_parallel_size=1,
        disable_log_stats=False,
        gpu_memory_utilization=0.5
    )

    tracker = GenerationMetricsTracker(llm)
    tracker.start()

    ### Prepare SEAL steering vector request
    
    # Define the suffix for newline tokens in the tokenizer
    target_suffix = "ĊĊ"  # "\n\n" is tokenized as "ĊĊ"

    # Get complete tokenizer vocabulary
    vocab = tokenizer.get_vocab()

    # Find all tokens and their IDs that end with the target suffix
    # These are the newline tokens we'll apply steering to
    matching_tokens_ids = [
        token_id
        for token, token_id in vocab.items()
        if isinstance(token, str) and token.endswith(target_suffix)
    ]

    # Configure steering vector request for SEAL control
    steer_request = SteerVectorRequest(
        # Name and ID for the steering vector
        steer_vector_name="complex_control",
        steer_vector_id=4,
        
        # Configure the three steering vectors (execution, reflection, transition)
        vector_configs=[
            # Execution vector (positive scale to promote execution-like text)
            VectorConfig(
                path="seal/execution_avg_vector.gguf",
                scale=0.5,                            # Positive scale promotes this behavior
                target_layers=[20],                   # Apply at layer 20
                generate_trigger_tokens=matching_tokens_ids,  # Apply to newline tokens
                algorithm="direct",                   # Direct application
                normalize=False                       # Do not normalize vectors
            ),
            
            # Reflection vector (negative scale to suppress reflection)
            VectorConfig(
                path="seal/reflection_avg_vector.gguf",
                scale=-0.5,                           # Negative scale suppresses this behavior
                target_layers=[20],
                generate_trigger_tokens=matching_tokens_ids,
                algorithm="direct",
                normalize=False
            ),
            
            # Transition vector (negative scale to suppress transitions)
            VectorConfig(
                path="seal/transition_avg_vector.gguf",
                scale=-0.5,                           # Negative scale suppresses this behavior
                target_layers=[20],
                generate_trigger_tokens=matching_tokens_ids,
                algorithm="direct", 
                normalize=False
            ),
        ],
        
        # Additional parameters
        debug=False,                        # Don't output debug info
        conflict_resolution="sequential"    # Apply vectors in sequence
    )

    example_answers = llm.generate(
        examples[0],
        sampling_params,
        steer_vector_request=steer_request,
    )

    llm.llm_engine.do_log_stats()

    summary = tracker.stop()
    if summary:
        print(summary)
    
    ### SEAL setting
    # example_answers = llm.generate(
    #   examples[0], 
    #   SamplingParams(
    #       temperature=0,
    #       max_tokens=8192,
    #       skip_special_tokens=False,
    #   ), 
    #   steer_vector_request=steer_request
    # )

    # Print generated results
    for output in example_answers:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated Text: {generated_text!r}\n")
    
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    outputs = [output.outputs[0].text for output in example_answers]
    extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
    results = []
    for i, llm_output in enumerate(outputs):
        gold = parse(f"${answers[i]}$", extraction_config=extraction_target)
        answer = parse(llm_output, extraction_config=extraction_target)
        result = verify(gold, answer)
        results.append(result)
    accuracy = sum(results) / len(results)
    print(accuracy)

    length = 0
    for i in range(len(outputs)):
        length += len(tokenizer.tokenize(outputs[i], add_special_tokens=True))
    print("Length: ", length/len(outputs))


if __name__ == "__main__":
    main()
