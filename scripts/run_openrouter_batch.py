#!/usr/bin/env python3
"""
Run GMTW-Ro evaluation using OpenRouter API

OpenRouter provides access to 400+ models including free tiers.
See: https://openrouter.ai/models
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


# Popular free/cheap models on OpenRouter
SUGGESTED_MODELS = {
    "grok-4.1-fast": "x-ai/grok-4.1-fast:free",
    "gemma-3-4b": "google/gemma-3-4b-it:free",
    "gemma-3-12b": "google/gemma-3-12b-it:free",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
    "mistral-7b": "mistralai/mistral-7b-instruct:free",
    "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct:free",
    "deepseek-r1": "deepseek/deepseek-r1:free",
}


def call_openrouter(
    prompt: str,
    api_key: str,
    model: str = "x-ai/grok-4.1-fast:free",
    site_url: str = None,
    site_name: str = None,
) -> str:
    """
    Call OpenRouter API with a prompt

    Args:
        prompt: The prompt to send
        api_key: OpenRouter API key
        model: Model name (default: google/gemma-3-4b-it:free)
        site_url: Optional site URL for OpenRouter rankings
        site_name: Optional site name for OpenRouter rankings

    Returns:
        Model response text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Optional headers for OpenRouter leaderboards
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

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

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if not response.ok:
            error_detail = response.text
            print(f"\n  OpenRouter API error ({response.status_code}): {error_detail[:300]}")
            return ""

        data = response.json()

        # Check for error in response
        if "error" in data:
            print(f"\n  OpenRouter error: {data['error']}")
            return ""

        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        print(f"\n  Timeout after 120s")
        return ""
    except Exception as e:
        print(f"\n  Error: {e}")
        return ""


def run_batch(
    instances_file: str,
    output_file: str,
    api_key: str,
    model: str = "x-ai/grok-4.1-fast:free",
    language: str = "ro",
    max_instances: int = None,
    delay: float = 1.0,
    site_url: str = None,
    site_name: str = None,
):
    """
    Run batch evaluation using OpenRouter

    Args:
        instances_file: JSONL file with instances
        output_file: Output file for model responses
        api_key: OpenRouter API key
        model: OpenRouter model name
        language: 'ro' for Romanian, 'en' for English
        max_instances: Max instances to process (None = all)
        delay: Delay in seconds between API calls (default: 1.0)
        site_url: Optional site URL for OpenRouter rankings
        site_name: Optional site name for OpenRouter rankings
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
    print(f"Output: {output_file}")
    print(f"Delay: {delay}s between calls\n")

    # Process each instance
    outputs = []
    failed = 0

    for i, instance in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] {instance.instance_id}...", end=" ", flush=True)

        # Get the appropriate prompt
        prompt = instance.prompt_ro if language == "ro" else instance.prompt_en

        # Call OpenRouter
        start_time = time.time()
        response = call_openrouter(prompt, api_key, model, site_url, site_name)
        latency = time.time() - start_time

        if response:
            print(f"OK ({latency:.1f}s)")
            outputs.append({
                "instance_id": instance.instance_id,
                "output": response,
                "model": model,
                "language": language,
                "latency": latency,
            })
        else:
            print(f"FAILED")
            failed += 1

        # Delay to avoid rate limits
        if i < len(instances):
            time.sleep(delay)

    # Save outputs
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    print(f"\nDone! Saved {len(outputs)} outputs to {output_file}")
    if failed > 0:
        print(f"Failed: {failed} instances")

    print(f"\nNext step: Evaluate with:")
    print(f"  python scripts/evaluate_outputs.py {instances_file} {output_file}")


def list_models():
    """Print suggested models"""
    print("Suggested free models on OpenRouter:\n")
    for shortname, fullname in SUGGESTED_MODELS.items():
        print(f"  --model {fullname}")
        print(f"     (shortcut: --model {shortname})")
        print()
    print("Browse all models at: https://openrouter.ai/models")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GMTW-Ro evaluation with OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with free Gemma 3 4B
  python scripts/run_openrouter_batch.py data/instances.jsonl --output gemma_out.jsonl

  # Run with free Llama 3.1 8B
  python scripts/run_openrouter_batch.py data/instances.jsonl --model llama-3.1-8b --output llama_out.jsonl

  # Run English prompts
  python scripts/run_openrouter_batch.py data/instances.jsonl --language en --output en_out.jsonl

  # List suggested models
  python scripts/run_openrouter_batch.py --list-models
        """
    )
    parser.add_argument("instances", nargs="?", help="JSONL file with instances")
    parser.add_argument("--output", default="openrouter_outputs.jsonl", help="Output file")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", default="x-ai/grok-4.1-fast:free", help="OpenRouter model (default: grok-4.1-fast:free)")
    parser.add_argument("--language", choices=["ro", "en"], default="ro", help="Prompt language")
    parser.add_argument("--max", type=int, help="Max instances to process")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between API calls (default: 1.0)")
    parser.add_argument("--site-url", help="Optional site URL for OpenRouter rankings")
    parser.add_argument("--site-name", help="Optional site name for OpenRouter rankings")
    parser.add_argument("--list-models", action="store_true", help="List suggested models and exit")

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        list_models()
        sys.exit(0)

    # Check for instances file
    if not args.instances:
        parser.print_help()
        print("\nError: instances file required")
        sys.exit(1)

    # Resolve model shortcut
    model = args.model
    if model in SUGGESTED_MODELS:
        model = SUGGESTED_MODELS[model]
        print(f"Using model: {model}")

    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: No API key provided")
        print("Either:")
        print("  1. Use --api-key YOUR_KEY")
        print("  2. Set OPENROUTER_API_KEY environment variable")
        print("\nOn Windows:")
        print("  set OPENROUTER_API_KEY=your_key_here")
        print("On Linux/Mac:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        print("\nGet a free API key at: https://openrouter.ai/keys")
        sys.exit(1)

    run_batch(
        args.instances,
        args.output,
        api_key,
        model,
        args.language,
        args.max,
        args.delay,
        args.site_url,
        args.site_name,
    )
