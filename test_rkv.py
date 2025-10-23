import torch
import random
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from metrics_tracker import GenerationMetricsTracker

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Configuration
# MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TEMPERATURE = 0
TOP_P = 0.95
MAX_TOKENS = 8192

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

### v0 backend
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

os.environ["VLLM_V0_R_KV_BUDGET"] = "-1"
os.environ["VLLM_V0_R_KV_BUFFER"] = "-1"

# os.environ["VLLM_V0_R_KV_BUDGET"] = "1536"
# os.environ["VLLM_V0_R_KV_BUFFER"] = "128"

### v1 backend
# os.environ["VLLM_USE_V1"] = "1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"

# # os.environ["VLLM_V1_R_KV_BUDGET"] = "-1"
# # os.environ["VLLM_V1_R_KV_BUFFER"] = "-1"

# os.environ["VLLM_V1_R_KV_BUDGET"] = "768"
# os.environ["VLLM_V1_R_KV_BUFFER"] = "128"

# os.environ["VLLM_RKV_METHOD"] = "rkv"
# # os.environ["VLLM_RKV_METHOD"] = "rkv2_slow"
# # os.environ["VLLM_RKV_METHOD_CONFIG"] = '{"mix_lambda": 0.07, "retain_ratio": 0.1}'


### Single Prompts
# examples = ['You are given a math problem. Problem: If $f(x) = \frac{3x-2}{x-2}$, what is the value of $f(-2)+f(-1)+f(0)$? Express your answer as a common fraction. Solve step by step with a brief explanation, then provide the final answer in the format: Final answer: \boxed{}',]
# answers = ['\\boxed{\\frac{14}{3}}',]

### math datasets
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

### Seal setup
# sampling_params = SamplingParams(
#     temperature=0,
#     max_tokens=1000,
#     skip_special_tokens=False,
# )

# Initialize the LLM
llm = LLM(
    model=MODEL_NAME,
    enforce_eager=True,
    enable_prefix_caching=False,
    tensor_parallel_size=1,
    disable_log_stats=False,
    gpu_memory_utilization=0.5
)

tracker = GenerationMetricsTracker(llm)
tracker.start()

# Generate outputs
example_answers = llm.generate(examples[0], sampling_params)

# llm.llm_engine.do_log_stats()

summary = tracker.stop()
if summary:
    print(summary)

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
