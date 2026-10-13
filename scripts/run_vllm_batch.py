#!/usr/bin/env python3
"""
Fast local evaluation for GMTW-Ro using vLLM
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rombench.gmtw_ro.worlds.base import Instance


def turn_marker_id(tokenizer):
    """The token id the chat template actually ends an assistant turn with.

    Some released checkpoints ship a `generation_config.json` whose
    `eos_token_id` is not the token their own template emits -- the RoGemma2
    releases carry `eos_token_id: 1` (`<eos>`, from pretraining) while their
    template ends a turn with `<end_of_turn>` (107). vLLM stops on the former,
    the model emits the latter, and generation runs past the answer into
    degenerate repetition until max_tokens, cut off mid-JSON. Scored as-is that
    reads as a model unable to close a brace.

    Returns the LAST SPECIAL token of the rendered template, not the last
    token: Gemma renders "...<end_of_turn>\\n", and stopping on that trailing
    newline would truncate every answer at its first line break -- worse than
    the bug being fixed.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    special = set(tokenizer.all_special_ids) | {
        tokenizer.convert_tokens_to_ids(t)
        for t in getattr(tokenizer, "additional_special_tokens", [])}
    marker = [i for i in ids if i in special]
    return marker[-1] if marker else None


def run_vllm_batch(
    instances_file: str,
    output_file: str,
    model_path: str,
    language: str = "ro",
    max_instances: int = None,
    max_tokens: int = 4096,
    batch_size: int = 32,
    align_stop_token: bool = False,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
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

    # Load tokenizer for chat template (auto-detects correct format)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Chat template detected: {tokenizer.chat_template[:80] if tokenizer.chat_template else 'None'}...")

    # Load model with vLLM
    print("\nLoading model (this takes a minute)...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=enforce_eager,
    )

    stop_ids = None
    if align_stop_token:
        marker = turn_marker_id(tokenizer)
        if marker is None:
            print("  stop token: no special token in the rendered template, "
                  "leaving the model's own eos_token_id in place")
        else:
            stop_ids = [marker]
            print(f"  stop token: also stopping on template turn marker id {marker}")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop_token_ids=stop_ids,
    )

    # Prepare all prompts using the model's native chat template
    prompts = []
    for inst in instances:
        prompt = inst.prompt_ro if language == "ro" else (inst.prompt_de if language == "de" else inst.prompt_en)
        # Apply chat template from tokenizer (auto-detects Llama/Gemma/Mistral/etc.)
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
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
    parser.add_argument("--language", choices=["ro", "de", "en"], default="ro")
    parser.add_argument("--max", type=int, help="Max instances to process")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--align-stop-token", action="store_true",
        help="also stop on the token the model's own chat template ends a turn "
             "with, instead of trusting generation_config.json alone. Off by "
             "default so existing runs are unchanged; needed for checkpoints "
             "whose shipped eos_token_id their template never emits")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="fraction of the card vLLM may reserve (default 0.9, unchanged). "
             "A 9B model in bf16 needs ~18.5GB, so on a 143GB H200 the default "
             "reserves ~129GB it will not use -- lower it on a shared machine")
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="skip torch.compile and run the model eagerly. Slower, but the "
             "compile path needs Triton to build a CUDA helper with gcc, which "
             "fails without the Python development headers (Python.h). Use "
             "this when python3-dev is not installed and cannot be")

    args = parser.parse_args()

    run_vllm_batch(
        args.instances,
        args.output,
        args.model_path,
        args.language,
        args.max,
        args.max_tokens,
        args.batch_size,
        args.align_stop_token,
        args.gpu_memory_utilization,
        args.enforce_eager,
    )
