import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 1000
R1KV_BUDGET = 1536
R1KV_BUFFER = 128

# Prompts to generate completions for
# prompts = [
#     "Hello, my name is",
#     "How are you",
#     "Good morning",
#     "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
# ]

# prompts = ["The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 100 + 
#     f"The pass key is 71432. Remember it. 71432 is the pass key. " + "The grass is green. The sky is blue. "
#     "The sun is yellow. Here we go. There and back again. " * 100 + "What is the pass key?"
#     ]

# prompts = ['You are given a math problem. Problem: Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$ You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer. Provide the final answer in the format: Final answer:  \boxed{}']
prompts = ['You are given a math problem. Problem: If $f(x) = \frac{3x-2}{x-2}$, what is the value of $f(-2)+f(-1)+f(0)$? Express your answer as a common fraction. Solve step by step with a brief explanation, then provide the final answer in the format: Final answer: \boxed{}']

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
eos_token_id = tokenizer.eos_token_id


def build_seal_request(tok: AutoTokenizer) -> SteerVectorRequest:
    """Construct the SEAL multi-vector steering request."""
    target_suffix = "ĊĊ"  # tokenized newline suffix
    vocab = tok.get_vocab()
    matching_tokens_ids = [
        token_id for token, token_id in vocab.items()
        if isinstance(token, str) and token.endswith(target_suffix)
    ]
    if not matching_tokens_ids:
        # Fallback: apply to all tokens if suffix not found.
        matching_tokens_ids = [-1]

    vector_dir = Path(__file__).resolve().parent / "steer_examples"
    vectors = {
        "execution": 0.5,
        "reflection": -0.5,
        "transition": -0.5,
    }
    vector_configs = [
        VectorConfig(
            path=str(vector_dir / f"{name}_avg_vector.gguf"),
            scale=scale,
            target_layers=[20],
            generate_trigger_tokens=matching_tokens_ids,
            algorithm="direct",
            normalize=False,
        ) for name, scale in vectors.items()
    ]

    return SteerVectorRequest(
        steer_vector_name="seal",
        steer_vector_id=1,
        conflict_resolution="sequential",
        vector_configs=vector_configs,
    )


SEAL_REQUEST = build_seal_request(tokenizer)

# Set up sampling parameters
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    stop_token_ids=[eos_token_id],
)

def configure_r1kv(enable: bool) -> None:
    if enable:
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_V1_R_KV_BUDGET"] = str(R1KV_BUDGET)
        os.environ["VLLM_V1_R_KV_BUFFER"] = str(R1KV_BUFFER)
    else:
        os.environ["VLLM_V1_R_KV_BUDGET"] = "-1"
        os.environ["VLLM_V1_R_KV_BUFFER"] = "-1"
        os.environ.pop("VLLM_USE_V1", None)


def run_case(name: str, *, enable_r1kv: bool, enable_steering: bool) -> None:
    print(f"\n=== {name} ===")
    configure_r1kv(enable_r1kv)

    llm_kwargs = dict(
        model=MODEL_NAME,
        enforce_eager=False,
        enable_prefix_caching=False,
    )
    if enable_steering:
        llm_kwargs["enable_steer_vector"] = True

    llm = LLM(**llm_kwargs)

    generate_kwargs = {}
    if enable_steering:
        generate_kwargs["steer_vector_request"] = SEAL_REQUEST

    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}\nGenerated Text: {generated_text!r}\n")


cases = [
    ("full-kv (no steering)", False, False),
    ("R1KV-only", True, False),
    ("steering-only", False, True),
    ("R1KV + steering", True, True),
]

for case_name, use_r1kv, use_steering in cases:
    run_case(case_name, enable_r1kv=use_r1kv, enable_steering=use_steering)

# Reset R1KV env vars after the tests.
configure_r1kv(False)
