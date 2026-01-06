#!/usr/bin/env python3
"""
Local evaluation for GMTW-Ro using OpenLLM-Ro/RoLlama3.1-8b-Instruct
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True
    )

    # Try Flash Attention 2 for speed, fall back to default
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        print("✓ Using Flash Attention 2")
    except Exception as e:
        print(f"⚠ Flash Attention not available ({e}), using default")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    # Compile for faster inference (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("✓ Using torch.compile")
    except Exception:
        pass

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_tokens=4096):
    """
    Apply the chat template for RoLlama3.1-Instruct and generate response.
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
            do_sample=False
        )

    gen = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()

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
    print(f"Using model: {model_path}")

    model, tokenizer = load_model(model_path)
    outputs = []

    # Loop
    for i, inst in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] {inst.instance_id}...", end=" ")

        prompt = inst.prompt_ro if language == "ro" else inst.prompt_en

        start = time.time()
        result = generate_response(model, tokenizer, prompt)
        latency = time.time() - start

        print(f"✓ ({latency:.1f}s)")

        outputs.append({
            "instance_id": inst.instance_id,
            "output": result,
            "model": model_path,
            "language": language,
            "latency": latency
        })

        time.sleep(delay)

    with open(output_file, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(outputs)} outputs to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GMTW-Ro with a local RoLlama model")
    parser.add_argument("instances", help="JSONL file with GMTW instances")
    parser.add_argument("--model-path", required=True, help="HuggingFace model ID or local folder")
    parser.add_argument("--output", default="local_llama_outputs.jsonl")
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
