#!/usr/bin/env python3
"""
Run GMTW-Ro evaluation using Groq API
"""

import json
import sys
import os
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro.worlds.base import Instance

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found.")
    print("Install it with: pip install requests")
    sys.exit(1)


def call_groq_with_retry(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile", max_retries: int = 5) -> str:
    """
    Call Groq API with a prompt, with retry logic and exponential backoff.

    Args:
        prompt: The prompt to send
        api_key: Groq API key
        model: Model name
        max_retries: Maximum number of retry attempts

    Returns:
        Model response text
    """
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)

            if response.ok:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content.strip()
                # Empty response - log what we got for debugging
                finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
                print(f"⚠ Empty (finish_reason={finish_reason})", end=" ")
                return ""

            # Handle rate limiting (429) or server errors (5xx) - retry
            if response.status_code == 429 or response.status_code >= 500:
                wait_time = (2 ** attempt) * 2
                print(f"⏳ Error {response.status_code}, waiting {wait_time}s...", end=" ", flush=True)
                time.sleep(wait_time)
                last_error = f"HTTP {response.status_code}"
                continue

            # Other 4xx errors - don't retry (model issue, not transient)
            error_detail = response.text[:50] if response.text else "No details"
            print(f"❌ Error {response.status_code}", end=" ")
            return ""

        except requests.exceptions.Timeout:
            wait_time = (2 ** attempt) * 2
            print(f"⏳ Timeout, waiting {wait_time}s...", end=" ", flush=True)
            time.sleep(wait_time)
            last_error = "Timeout"
            continue

        except Exception as e:
            wait_time = (2 ** attempt) * 2
            print(f"⏳ {type(e).__name__}: {str(e)[:30]}, waiting {wait_time}s...", end=" ", flush=True)
            time.sleep(wait_time)
            last_error = str(e)[:50]
            continue

    print(f"❌ {last_error}", end=" ")
    return ""


def run_batch(
    instances_file: str,
    output_file: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    language: str = "ro",
    max_instances: int = None,
    delay: float = 2.0,
):
    """
    Run batch evaluation using Groq

    Args:
        instances_file: JSONL file with instances
        output_file: Output file for model responses
        api_key: Groq API key
        model: Groq model name
        language: 'ro' for Romanian, 'en' for English
        max_instances: Max instances to process (None = all)
        delay: Delay in seconds between API calls (default: 2.0)
    """
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
    print(f"Output: {output_file}\n")

    # Process each instance
    outputs = []
    failed = []

    for i, instance in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] Processing {instance.instance_id}...", end=" ")

        # Get the appropriate prompt
        prompt = instance.prompt_ro if language == "ro" else instance.prompt_en

        # Call Groq with retry logic
        start_time = time.time()
        response = call_groq_with_retry(prompt, api_key, model)
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
            print(f"✗ Failed permanently")
            failed.append(instance.instance_id)

        # Delay to avoid rate limits
        time.sleep(delay)

    # Save outputs
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    print(f"\n✓ Saved {len(outputs)}/{len(instances)} outputs to {output_file}")

    if failed:
        print(f"\n⚠ {len(failed)} instances failed (model returned empty):")
        for fid in failed[:10]:
            print(f"  - {fid}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

        # Save failed list
        failed_file = output_file.replace('.jsonl', '_failed.txt')
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed))
        print(f"  Failed list saved to: {failed_file}")

    print(f"\nNext step: Evaluate with:")
    print(f"  python scripts/evaluate_outputs.py {instances_file} {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GMTW-Ro evaluation with Groq")
    parser.add_argument("instances", help="JSONL file with instances")
    parser.add_argument("--output", default="groq_outputs.jsonl", help="Output file")
    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq model")
    parser.add_argument("--language", choices=["ro", "en"], default="ro", help="Prompt language")
    parser.add_argument("--max", type=int, help="Max instances to process")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between API calls (default: 2.0)")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: No API key provided")
        print("Either:")
        print("  1. Use --api-key YOUR_KEY")
        print("  2. Set GROQ_API_KEY environment variable")
        print("\nOn Windows:")
        print("  set GROQ_API_KEY=your_key_here")
        print("On Linux/Mac:")
        print("  export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    run_batch(
        args.instances,
        args.output,
        api_key,
        args.model,
        args.language,
        args.max,
        args.delay,
    )
