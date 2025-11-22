#!/usr/bin/env python3
"""
Compare multiple model outputs side-by-side
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro import Instance, evaluate_instance


def compare_models(instances_file: str, *output_files):
    """
    Compare multiple model outputs side-by-side

    Args:
        instances_file: JSONL with instances
        *output_files: Multiple JSONL files with model outputs
    """
    # Load instances
    instances = {}
    with open(instances_file, 'r', encoding='utf-8') as f:
        for line in f:
            inst = Instance.from_dict(json.loads(line))
            instances[inst.instance_id] = inst

    # Load all outputs
    all_outputs = {}
    model_names = {}

    for output_file in output_files:
        outputs = {}
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                outputs[data['instance_id']] = data['output']
                # Try to get model name from data
                if 'model' in data and output_file not in model_names:
                    model_names[output_file] = data['model']

        all_outputs[output_file] = outputs

        # Use filename as model name if not found
        if output_file not in model_names:
            model_names[output_file] = Path(output_file).stem

    # Evaluate all models
    results_by_model = {}

    for output_file, outputs in all_outputs.items():
        results = []
        for inst_id, instance in instances.items():
            if inst_id in outputs:
                result = evaluate_instance(instance, outputs[inst_id])
                results.append(result)

        results_by_model[output_file] = results

    # Compute and display comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    # Header
    print(f"\n{'Model':<30} {'U':>8} {'R':>8} {'G':>8} {'F':>8} {'Avg':>8}")
    print("-"*80)

    # Each model
    for output_file, results in results_by_model.items():
        if not results:
            continue

        model_name = model_names[output_file]
        avg_U = sum(r.U for r in results) / len(results)
        avg_R = sum(r.R for r in results) / len(results)
        avg_G = sum(r.G for r in results) / len(results)
        avg_F = sum(r.F for r in results) / len(results)
        avg_all = (avg_U + avg_R + avg_G + avg_F) / 4

        print(f"{model_name:<30} {avg_U:>8.3f} {avg_R:>8.3f} {avg_G:>8.3f} {avg_F:>8.3f} {avg_all:>8.3f}")

    print("="*80)

    # Per-instance comparison
    print("\nPer-Instance Breakdown:")
    print("-"*80)

    for inst_id in sorted(instances.keys()):
        instance = instances[inst_id]
        world_type = instance.world.world_type

        print(f"\n{inst_id} ({world_type})")

        for output_file in output_files:
            if inst_id not in all_outputs[output_file]:
                continue

            result = None
            for r in results_by_model[output_file]:
                if r.instance_id == inst_id:
                    result = r
                    break

            if result:
                model_name = model_names[output_file][:20]
                status = "✓" if result.U >= 0.7 and result.R >= 0.7 and result.F >= 0.7 else "✗"
                print(f"  {status} {model_name:<20} U={result.U:.2f} R={result.R:.2f} G={result.G:.2f} F={result.F:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare model outputs")
    parser.add_argument("instances", help="JSONL file with instances")
    parser.add_argument("outputs", nargs="+", help="JSONL files with model outputs")

    args = parser.parse_args()

    compare_models(args.instances, *args.outputs)
