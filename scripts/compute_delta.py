#!/usr/bin/env python3
"""
Compute Foreign Language Penalty (Δ) for GMTW-Ro

Compares model performance on Romanian vs English prompts to quantify
language-induced capability degradation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro import Instance, evaluate_instance


def compute_delta(instances_file: str, ro_outputs_file: str, en_outputs_file: str):
    """
    Compute Foreign Language Penalty (Δ)

    Args:
        instances_file: JSONL with instances
        ro_outputs_file: JSONL with Romanian outputs
        en_outputs_file: JSONL with English outputs
    """
    # Load instances
    instances = {}
    with open(instances_file, 'r', encoding='utf-8') as f:
        for line in f:
            inst = Instance.from_dict(json.loads(line))
            instances[inst.instance_id] = inst

    # Load Romanian outputs
    ro_outputs = {}
    with open(ro_outputs_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ro_outputs[data['instance_id']] = data['output']

    # Load English outputs
    en_outputs = {}
    with open(en_outputs_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            en_outputs[data['instance_id']] = data['output']

    print(f"Loaded {len(instances)} instances")
    print(f"Romanian outputs: {len(ro_outputs)}")
    print(f"English outputs: {len(en_outputs)}")

    # Find common instances (where we have both RO and EN outputs)
    common_instances = set(ro_outputs.keys()) & set(en_outputs.keys()) & set(instances.keys())

    if not common_instances:
        print("\n❌ No common instances found!")
        print("Make sure you have outputs for the same instances in both languages.")
        return

    print(f"Common instances: {len(common_instances)}\n")

    # Evaluate both
    ro_results = []
    en_results = []

    for inst_id in sorted(common_instances):
        instance = instances[inst_id]

        # Evaluate Romanian
        ro_result = evaluate_instance(instance, ro_outputs[inst_id])
        ro_results.append(ro_result)

        # Evaluate English
        en_result = evaluate_instance(instance, en_outputs[inst_id])
        en_results.append(en_result)

    # Compute averages
    avg_U_ro = sum(r.U for r in ro_results) / len(ro_results)
    avg_R_ro = sum(r.R for r in ro_results) / len(ro_results)
    avg_G_ro = sum(r.G for r in ro_results) / len(ro_results)
    avg_F_ro = sum(r.F for r in ro_results) / len(ro_results)

    avg_U_en = sum(r.U for r in en_results) / len(en_results)
    avg_R_en = sum(r.R for r in en_results) / len(en_results)
    avg_G_en = sum(r.G for r in en_results) / len(en_results)
    avg_F_en = sum(r.F for r in en_results) / len(en_results)

    # Compute Δ (Foreign Language Penalty)
    delta_U = avg_U_en - avg_U_ro
    delta_R = avg_R_en - avg_R_ro
    delta_F = avg_F_en - avg_F_ro

    # Display results
    print("="*80)
    print("FOREIGN LANGUAGE PENALTY (Δ) ANALYSIS")
    print("="*80)

    print(f"\n{'Metric':<20} {'English':>12} {'Romanian':>12} {'Δ (Penalty)':>15}")
    print("-"*80)
    print(f"{'U (Understanding)':<20} {avg_U_en:>12.3f} {avg_U_ro:>12.3f} {delta_U:>15.3f}")
    print(f"{'R (Reasoning)':<20} {avg_R_en:>12.3f} {avg_R_ro:>12.3f} {delta_R:>15.3f}")
    print(f"{'F (Faithfulness)':<20} {avg_F_en:>12.3f} {avg_F_ro:>12.3f} {delta_F:>15.3f}")
    print(f"{'G (Generation)':<20} {avg_G_en:>12.3f} {avg_G_ro:>12.3f} {'[N/A]':>15}")
    print("="*80)

    print("\nInterpretation:")
    print(f"  ΔU = {delta_U:.3f}  →  ", end="")
    if delta_U > 0.1:
        print("⚠️  Model struggles to understand Romanian constraints")
    elif delta_U > 0.05:
        print("Minor understanding degradation in Romanian")
    else:
        print("✓ Understanding is consistent across languages")

    print(f"  ΔR = {delta_R:.3f}  →  ", end="")
    if delta_R > 0.1:
        print("⚠️  Romanian processing significantly impacts reasoning ability")
    elif delta_R > 0.05:
        print("Minor reasoning degradation in Romanian")
    else:
        print("✓ Reasoning is consistent across languages")

    print(f"  ΔF = {delta_F:.3f}  →  ", end="")
    if delta_F > 0.1:
        print("⚠️  Model is less faithful in Romanian explanations")
    elif delta_F > 0.05:
        print("Minor faithfulness degradation in Romanian")
    else:
        print("✓ Faithfulness is consistent across languages")

    print("\n" + "="*80)
    print(f"Based on {len(common_instances)} instances")
    print("="*80)

    # Per-instance breakdown (optional)
    print("\nPer-Instance Breakdown:")
    print("-"*80)
    print(f"{'Instance':<20} {'U_Δ':>8} {'R_Δ':>8} {'F_Δ':>8}")
    print("-"*80)

    for inst_id, ro_res, en_res in zip(sorted(common_instances), ro_results, en_results):
        delta_u = en_res.U - ro_res.U
        delta_r = en_res.R - ro_res.R
        delta_f = en_res.F - ro_res.F

        # Highlight large penalties
        marker = " ⚠️" if abs(delta_u) > 0.2 or abs(delta_r) > 0.2 or abs(delta_f) > 0.2 else ""

        print(f"{inst_id:<20} {delta_u:>8.2f} {delta_r:>8.2f} {delta_f:>8.2f}{marker}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Foreign Language Penalty")
    parser.add_argument("instances", help="JSONL file with instances")
    parser.add_argument("ro_outputs", help="JSONL file with Romanian outputs")
    parser.add_argument("en_outputs", help="JSONL file with English outputs")

    args = parser.parse_args()

    compute_delta(args.instances, args.ro_outputs, args.en_outputs)
