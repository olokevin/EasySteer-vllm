# source ../R-KV/.venv/bin/activate
# source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1

export VLLM_V1_R_KV_BUDGET=1536        # max tokens to keep
export VLLM_V1_R_KV_BUFFER=128         # tokens to accumulate before compressing

# export VLLM_V1_R_KV_BUDGET=-1        # max tokens to keep
# export VLLM_V1_R_KV_BUFFER=-1        # tokens to accumulate before compressing

GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

time="exp_$(date +%Y%m%d_%H%M%S)"

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# MODEL_PATH=Qwen/Qwen3-0.6B

DATASET_PATH=${DATASET_PATH:-../data/math.jsonl}
# DATASET_PATH=${DATASET_PATH:-../data/aime24.jsonl}
# DATASET_PATH=${DATASET_PATH:-../data/gsm8k.jsonl}

OUTPUT_PATH=./outputs/math-rkv-vllm-DeepSeek-R1-Distill-Llama-8B-$(date +%Y%m%d_%H%M%S)/output.jsonl
# OUTPUT_PATH=./outputs/math-fullkv-vllm-DeepSeek-R1-Distill-Llama-8B-$(date +%Y%m%d_%H%M%S)/output.jsonl

# OUTPUT_PATH=./outputs/aime-rkv-vllm-DeepSeek-R1-Distill-Llama-8B-$(date +%Y%m%d_%H%M%S)/output.jsonl
# OUTPUT_PATH=./outputs/aime-fullkv-vllm-DeepSeek-R1-Distill-Llama-8B-$(date +%Y%m%d_%H%M%S)/output.jsonl



python run_math_vllm.py \
    --dataset_path "$DATASET_PATH" \
    --save_path "$OUTPUT_PATH" \
    --model_path "$MODEL_PATH" \
    --max_length -1 \
    --eval_batch_size 1 \
    --temperature 0.6 \
    --top_p 0.95 \
    --attention_backend "$VLLM_ATTENTION_BACKEND" \
    --kv_budget "$VLLM_V1_R_KV_BUDGET" \
    --window_size "$VLLM_V1_R_KV_BUFFER" \
    --gpu_memory_utilization "$GPU_MEMORY_UTIL" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS"

# nohup bash ./run_math_vllm.sh > nohup_rkv.log 2>&1 &
# nohup bash ./run_math_vllm.sh > nohup_fullkv.log 2>&1 &