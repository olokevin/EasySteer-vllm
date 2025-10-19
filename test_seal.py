# Import necessary libraries
import os
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig
from transformers import AutoTokenizer

# Set environment variables
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# Initialize LLM with steering vector capability
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    enable_steer_vector=True,
    enforce_eager=True,
    tensor_parallel_size=1
)

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
sv_request = SteerVectorRequest(
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

import json
file_path = "/home/yequan/Project/R-KV/HuggingFace/data/math.jsonl"

problems = []
answers = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        problems.append(item["problem"])
        answers.append(item["answer"])

# 看看前两个
print("Problems:", problems[:2])
print("Answers:", answers[:2])


examples = ["Please reason step by step, and put your final answer within \\boxed{}.\nUser: " + prompt + "\nAssistant: <think>" for prompt in problems]

# Generate response with SEAL steering
example_answers = llm.generate(
    examples[0], 
    SamplingParams(
        temperature=0,
        max_tokens=8192,
        skip_special_tokens=False,
    ), 
    steer_vector_request=sv_request
)

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