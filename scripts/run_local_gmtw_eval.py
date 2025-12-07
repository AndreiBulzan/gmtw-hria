#!/usr/bin/env python3
"""
Run GMTW-Ro evaluation using a LOCAL Gemma model.
"""

import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rombench.gmtw_ro.worlds.base import Instance


def load_model(model_path):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_tokens=512):
    """
    Generate an answer using chat template (required for Gemma).
    """

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def run_local(
    instances_file,
    output_file,
    model_path,
    language="ro",
    max_instances=None,
    delay=0.0
):
    instances = []
    with open(instances_file, "r", encoding="utf-8") as f:
        for line in f:
            instances.append(Instance.from_dict(json.loads(line)))

    if max_instances:
        instances = instances[:max_instances]

    print(f"Loaded {len(instances)} instances")

    model, tokenizer = load_model(model_path)

    outputs = []

    for i, inst in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] {inst.instance_id}...", end=" ")

        prompt = inst.prompt_ro if language == "ro" else inst.prompt_en

        start = time.time()
        result = generate_response(model, tokenizer, prompt)
        latency = time.time() - start

        print(f"({latency:.1f}s)")

        outputs.append({
            "instance_id": inst.instance_id,
            "output": result,
            "model": model_path,
            "language": language,
            "latency": latency
        })

        time.sleep(delay)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n Saved outputs to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instances", help="JSONL file with GMTW instances")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="local_outputs.jsonl")
    parser.add_argument("--language", choices=["ro", "en"], default="ro")
    parser.add_argument("--max", type=int)
    parser.add_argument("--delay", type=float, default=0.0)

    args = parser.parse_args()

    run_local(
        args.instances,
        args.output,
        args.model_path,
        args.language,
        args.max,
        args.delay
    )
