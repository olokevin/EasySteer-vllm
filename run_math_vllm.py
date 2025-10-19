import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_):
        return iterable
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

# dataset2max_length = {
#     "gsm8k": 8192,
#     "aime24": 16384,
#     "math": 8192,
# }

dataset2max_length = {
    "gsm8k": 32768,
    "aime24": 32768,
    "math": 16384,
}

prompt_template = (
    "You are given a math problem.\n\nProblem: {question}\n\n"
    " You need to solve the problem step by step. First, you need to provide the"
    " chain-of-thought, then provide the final answer.\n\n Provide the final"
    " answer in the format: Final answer:  \\boxed{{}}"
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_dataset(
    dataset_path: str, dataset_name: str, tokenizer: AutoTokenizer
) -> Tuple[List[str], List[Dict], List[int]]:
    prompts: List[str] = []
    samples: List[Dict] = []
    prefill_lengths: List[int] = []

    question_key = dataset2key[dataset_name][0]

    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            example = json.loads(line)
            question = example[question_key]
            example["question"] = question

            prompt = prompt_template.format(**example)
            example["prompt"] = prompt
            example["index"] = idx

            prompts.append(prompt)
            samples.append(example)
            prefill_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=True)))

    return prompts, samples, prefill_lengths


def configure_environment(args) -> None:
    if args.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    if args.kv_budget is not None:
        os.environ["VLLM_V1_R_KV_BUDGET"] = str(args.kv_budget)
    if args.window_size is not None:
        os.environ["VLLM_V1_R_KV_BUFFER"] = str(args.window_size)
    if args.kv_method:
        os.environ["VLLM_RKV_METHOD"] = args.kv_method
    if args.kv_method_config:
        os.environ["VLLM_RKV_METHOD_CONFIG"] = args.kv_method_config


def generate_with_vllm(args):
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts, samples, prefill_lengths = prepare_dataset(
        args.dataset_path, args.dataset_name, tokenizer
    )

    stop_token_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_length,
        n=1,
        stop_token_ids=stop_token_ids or None,
        seed=args.seed,
    )

    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    llm = LLM(**{k: v for k, v in llm_kwargs.items() if v is not None})

    with open(args.save_path, "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(prompts), args.eval_batch_size)):
            batch_prompts = prompts[start : start + args.eval_batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)

            for local_idx, output in enumerate(outputs):
                sample_idx = start + local_idx
                generated = output.outputs[0]
                generated_text = generated.text
                output_tokens = len(generated.token_ids)
                prefill_tokens = prefill_lengths[sample_idx]
                total_tokens = prefill_tokens + output_tokens

                sample = samples[sample_idx]
                sample["prompt"] = batch_prompts[local_idx]
                sample["output"] = generated_text
                sample["prefill_tokens"] = prefill_tokens
                sample["output_tokens"] = output_tokens
                sample["total_tokens"] = total_tokens
                sample["sample_idx"] = sample_idx

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--attention_backend", type=str, default=None)
    parser.add_argument("--kv_budget", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--kv_method", type=str, default=None,
                        help="Compression method to use (e.g., rkv, rkv2_slow).")
    parser.add_argument(
        "--kv_method_config",
        type=str,
        default=None,
        help="JSON string with extra kwargs for the KV compression method.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    args.dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    if args.dataset_name not in dataset2key:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    if args.max_length == -1:
        args.max_length = dataset2max_length[args.dataset_name]

    configure_environment(args)
    generate_with_vllm(args)
