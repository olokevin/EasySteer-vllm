export CUDA_VISIBLE_DEVICES=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1
export VLLM_V1_R_KV_BUDGET=1536 # max tokens to keep
export VLLM_V1_R_KV_BUFFER=128 # tokens to accumulate before compressing
python test.py