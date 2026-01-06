#!/usr/bin/env python3
"""
Fast local evaluation for GMTW-Ro using vLLM
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams
from rombench.gmtw_ro.worlds.base import Instance


def run_vllm_batch(
    instances_file: str,
    output_file: str,
    model_path: str,
    language: str = "ro",
    max_instances: int = None,
    max_tokens: int = 512,
    batch_size: int = 32,
):
    """
    Run batch evaluation using vLLM (much faster than transformers).
    """
    # Load instances
    instances = []
    with open(instances_file, "r", encoding="utf-8") as f:
        for line in f:
            instances.append(Instance.from_dict(json.loads(line)))

    if max_instances:
        instances = instances[:max_instances]

    print(f"Loaded {len(instances)} instances")
    print(f"Model: {model_path}")
    print(f"Batch size: {batch_size}")

    # Load model with vLLM
    print("\nLoading model (this takes a minute)...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
    )

    # Prepare all prompts
    prompts = []
    for inst in instances:
        prompt = inst.prompt_ro if language == "ro" else inst.prompt_en
        # Apply chat template
        chat_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompts.append(chat_prompt)

    # Process in batches
    outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    start_time = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_instances = instances[batch_start:batch_end]

        print(f"[Batch {batch_idx + 1}/{total_batches}] Processing {len(batch_prompts)} prompts...", end=" ", flush=True)

        batch_time = time.time()
        results = llm.generate(batch_prompts, sampling_params)
        batch_elapsed = time.time() - batch_time

        for inst, result in zip(batch_instances, results):
            text = result.outputs[0].text.strip()
            outputs.append({
                "instance_id": inst.instance_id,
                "output": text,
                "model": model_path,
                "language": language,
            })

        print(f"✓ ({batch_elapsed:.1f}s, {batch_elapsed/len(batch_prompts):.2f}s/prompt)")

    total_time = time.time() - start_time
    print(f"\n✓ Processed {len(outputs)} instances in {total_time:.1f}s ({total_time/len(outputs):.2f}s/prompt avg)")

    # Save outputs
    with open(output_file, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"✓ Saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GMTW-Ro with vLLM (fast)")
    parser.add_argument("instances", help="JSONL file with GMTW instances")
    parser.add_argument("--model-path", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", default="vllm_outputs.jsonl")
    parser.add_argument("--language", choices=["ro", "en"], default="ro")
    parser.add_argument("--max", type=int, help="Max instances to process")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    run_vllm_batch(
        args.instances,
        args.output,
        args.model_path,
        args.language,
        args.max,
        args.max_tokens,
        args.batch_size,
    )
