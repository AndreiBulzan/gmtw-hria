#!/usr/bin/env python3
"""
Run GMTW-Ro evaluation using Cerebras API

Cerebras offers extremely fast inference (~2000+ tokens/s).
Get a free API key at: https://cloud.cerebras.ai/

Available models:
  - llama3.1-8b (~2200 tokens/s)
  - llama-3.3-70b (~2100 tokens/s)
  - qwen-3-32b (~2600 tokens/s)
"""

import json
import sys
import os
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro.worlds.base import Instance

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    print("Error: 'cerebras-cloud-sdk' library not found.")
    print("Install it with: pip install cerebras-cloud-sdk")
    sys.exit(1)


def call_cerebras(client: Cerebras, prompt: str, model: str = "llama3.1-8b") -> str:
    """
    Call Cerebras API with a prompt

    Args:
        client: Cerebras client instance
        prompt: The prompt to send
        model: Model name (default: llama3.1-8b)

    Returns:
        Model response text
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.3,
            max_tokens=2048,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""


def run_batch(
    instances_file: str,
    output_file: str,
    api_key: str,
    model: str = "llama3.1-8b",
    language: str = "ro",
    max_instances: int = None,
    delay: float = 0.0,
):
    """
    Run batch evaluation using Cerebras

    Args:
        instances_file: JSONL file with instances
        output_file: Output file for model responses
        api_key: Cerebras API key
        model: Cerebras model name
        language: 'ro' for Romanian, 'en' for English
        max_instances: Max instances to process (None = all)
        delay: Delay in seconds between API calls (default: 0.0 - Cerebras is fast!)
    """
    # Initialize client
    client = Cerebras(api_key=api_key)

    # Load instances
    instances = []
    with open(instances_file, 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(Instance.from_dict(json.loads(line)))

    if max_instances:
        instances = instances[:max_instances]

    print(f"Loaded {len(instances)} instances")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Output: {output_file}")
    print(f"Delay: {delay}s\n")

    # Process each instance
    outputs = []
    failed = 0

    for i, instance in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] Processing {instance.instance_id}...", end=" ", flush=True)

        # Get the appropriate prompt
        prompt = instance.prompt_ro if language == "ro" else instance.prompt_en

        # Call Cerebras
        start_time = time.time()
        response = call_cerebras(client, prompt, model)
        latency = time.time() - start_time

        if response:
            print(f"✓ ({latency:.1f}s)")
            outputs.append({
                "instance_id": instance.instance_id,
                "output": response,
                "model": model,
                "language": language,
                "latency": latency,
            })
        else:
            print(f"✗ Failed")
            failed += 1

        # Optional delay between calls
        if delay > 0:
            time.sleep(delay)

    # Save outputs
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(outputs)} outputs to {output_file}")
    if failed > 0:
        print(f"✗ Failed: {failed} instances")

    # Calculate average latency
    if outputs:
        avg_latency = sum(o['latency'] for o in outputs) / len(outputs)
        print(f"Average latency: {avg_latency:.2f}s")

    print(f"\nNext step: Evaluate with:")
    print(f"  python scripts/evaluate_outputs.py data/gmtw_ro_v0.jsonl {output_file} --save-metrics data/metrics_{Path(output_file).stem}.jsonl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GMTW-Ro evaluation with Cerebras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  llama3.1-8b     - Llama 3.1 8B (~2200 tokens/s)
  llama-3.3-70b   - Llama 3.3 70B (~2100 tokens/s)
  qwen-3-32b      - Qwen 3 32B (~2600 tokens/s)

Examples:
  python run_cerebras_batch.py data/gmtw_ro_v0.jsonl --model llama3.1-8b
  python run_cerebras_batch.py data/gmtw_ro_v0.jsonl --model llama-3.3-70b --max 10
"""
    )
    parser.add_argument("instances", help="JSONL file with instances")
    parser.add_argument("--output", help="Output file (default: outputs_cerebras_{model}.jsonl)")
    parser.add_argument("--api-key", help="Cerebras API key (or set CEREBRAS_API_KEY env var)")
    parser.add_argument("--model", default="llama3.1-8b", help="Cerebras model (default: llama3.1-8b)")
    parser.add_argument("--language", choices=["ro", "en"], default="ro", help="Prompt language")
    parser.add_argument("--max", type=int, help="Max instances to process")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between calls in seconds (default: 0)")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: No API key provided")
        print("Either:")
        print("  1. Use --api-key YOUR_KEY")
        print("  2. Set CEREBRAS_API_KEY environment variable")
        print("\nGet a free key at: https://cloud.cerebras.ai/")
        sys.exit(1)

    # Set default output file
    output_file = args.output
    if not output_file:
        model_name = args.model.replace(".", "").replace("-", "_")
        output_file = f"data/outputs_cerebras_{model_name}.jsonl"

    run_batch(
        args.instances,
        output_file,
        api_key,
        args.model,
        args.language,
        args.max,
        args.delay,
    )
