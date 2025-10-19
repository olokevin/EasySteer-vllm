from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 1000

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

### v0 backend
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

### v1 backend
# os.environ["VLLM_USE_V1"] = "1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"

os.environ["VLLM_V1_R_KV_BUDGET"] = "1536"
os.environ["VLLM_V1_R_KV_BUFFER"] = "128"
# os.environ["VLLM_RKV_METHOD"] = "rkv"
os.environ["VLLM_RKV_METHOD"] = "rkv2_slow"
# os.environ["VLLM_RKV_METHOD_CONFIG"] = '{"mix_lambda": 0.07, "retain_ratio": 0.1}'

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

# Set up sampling parameters
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    stop_token_ids=[eos_token_id],
)

# Initialize the LLM
llm = LLM(
    model=MODEL_NAME,
    # enforce_eager=False,
    enable_prefix_caching=False,
    ### following baseline in EasySteer
    enforce_eager=True,
    enable_steer_vector=True,
    tensor_parallel_size=1
)

# Generate outputs
outputs = llm.generate(prompts, sampling_params)

# Print generated results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated Text: {generated_text!r}\n")
