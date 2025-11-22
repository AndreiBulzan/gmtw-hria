#!/usr/bin/env python3
"""
Evaluate model outputs for GMTW-Ro instances
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro import Instance, evaluate_instance


def evaluate_batch(instances_file: str, outputs_file: str, output_metrics: str = None):
    """
    Evaluate a batch of model outputs

    Args:
        instances_file: JSONL file with instances
        outputs_file: JSONL file with model outputs (format: {"instance_id": "...", "output": "..."})
        output_metrics: Optional file to save detailed metrics
    """

    # Load instances
    instances = {}
    with open(instances_file, 'r', encoding='utf-8') as f:
        for line in f:
            inst = Instance.from_dict(json.loads(line))
            instances[inst.instance_id] = inst

    print(f"Loaded {len(instances)} instances")

    # Load outputs
    outputs = {}
    with open(outputs_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            outputs[data['instance_id']] = data['output']

    print(f"Loaded {len(outputs)} outputs")

    # Evaluate
    results = []
    for inst_id, instance in instances.items():
        if inst_id not in outputs:
            print(f"⚠️  Missing output for {inst_id}")
            continue

        result = evaluate_instance(instance, outputs[inst_id])
        results.append(result)

        # Print individual result
        status = "✓" if result.U > 0.7 and result.R > 0.7 else "✗"
        print(f"{status} {inst_id}: U={result.U:.2f} R={result.R:.2f} G={result.G:.2f} F={result.F:.2f}")

    # Compute averages
    if results:
        avg_U = sum(r.U for r in results) / len(results)
        avg_R = sum(r.R for r in results) / len(results)
        avg_G = sum(r.G for r in results) / len(results)
        avg_F = sum(r.F for r in results) / len(results)

        print("\n" + "="*60)
        print(f"AVERAGE SCORES ({len(results)} instances)")
        print("="*60)
        print(f"  U (Understanding): {avg_U:.3f}")
        print(f"  R (Reasoning):     {avg_R:.3f}")
        print(f"  G (Generation):    {avg_G:.3f}")
        print(f"  F (Faithfulness):  {avg_F:.3f}")
        print("="*60)

        # Save detailed metrics if requested
        if output_metrics:
            with open(output_metrics, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
            print(f"\n✓ Detailed metrics saved to {output_metrics}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GMTW-Ro outputs")
    parser.add_argument("instances", help="JSONL file with instances")
    parser.add_argument("outputs", help="JSONL file with model outputs")
    parser.add_argument("--save-metrics", help="Save detailed metrics to file")

    args = parser.parse_args()

    evaluate_batch(args.instances, args.outputs, args.save_metrics)
