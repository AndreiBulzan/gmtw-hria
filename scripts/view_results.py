#!/usr/bin/env python3
"""
View detailed GMTW-Ro evaluation results
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro import Instance, evaluate_instance


def view_results(instances_file: str, outputs_file: str, filter_type: str = None):
    """
    View detailed results with prompts, outputs, and failure analysis

    Args:
        instances_file: JSONL with instances
        outputs_file: JSONL with model outputs
        filter_type: Show only 'failed', 'passed', or None for all
    """
    # Load instances
    instances = {}
    with open(instances_file, 'r', encoding='utf-8') as f:
        for line in f:
            inst = Instance.from_dict(json.loads(line))
            instances[inst.instance_id] = inst

    # Load outputs
    outputs = {}
    with open(outputs_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            outputs[data['instance_id']] = data['output']

    # Evaluate and display
    results = []
    for inst_id, instance in instances.items():
        if inst_id not in outputs:
            continue

        result = evaluate_instance(instance, outputs[inst_id])
        results.append((instance, outputs[inst_id], result))

    # Filter if requested
    if filter_type == "failed":
        results = [(i, o, r) for i, o, r in results if r.U < 0.7 or r.R < 0.7 or r.F < 0.7]
    elif filter_type == "passed":
        results = [(i, o, r) for i, o, r in results if r.U >= 0.7 and r.R >= 0.7 and r.F >= 0.7]

    # Display each result
    for idx, (instance, output, result) in enumerate(results, 1):
        print("\n" + "="*80)
        print(f"RESULT {idx}/{len(results)}: {result.instance_id}")
        print("="*80)

        # Scores summary
        status = "✓ PASS" if result.U >= 0.7 and result.R >= 0.7 and result.F >= 0.7 else "✗ FAIL"
        print(f"\n{status}")
        print(f"  U={result.U:.2f}  R={result.R:.2f}  G={result.G:.2f}  F={result.F:.2f}")

        # Show prompt
        print("\n" + "-"*80)
        print("PROMPT:")
        print("-"*80)
        print(instance.prompt_ro)

        # Show model output
        print("\n" + "-"*80)
        print("MODEL OUTPUT:")
        print("-"*80)
        print(output)

        # Detailed failure analysis
        if result.U < 1.0:
            print("\n" + "─"*80)
            print("❌ UNDERSTANDING FAILURES:")
            print("─"*80)
            for c in result.U_details.get('constraints', []):
                if not c['satisfied']:
                    print(f"  ✗ {c['id']}: {c['description']}")

        if result.R < 1.0:
            print("\n" + "─"*80)
            print("❌ REASONING FAILURES:")
            print("─"*80)
            for g in result.R_details.get('goals', []):
                if not g['satisfied']:
                    print(f"  ✗ {g['id']}: {g['description']}")

        if result.F < 1.0:
            print("\n" + "─"*80)
            print("❌ FAITHFULNESS ISSUES:")
            print("─"*80)
            missing = result.F_details.get('missing', [])
            if missing:
                print(f"  Missing from explanation: {', '.join(missing)}")
                print(f"  → These entities are in the JSON plan but not mentioned in the text")
            else:
                print(f"  (All planned entities were mentioned)")

            # Show what was checked
            total = result.F_details.get('total_count', 0)
            mentioned = result.F_details.get('mentioned_count', 0)
            if total > 0:
                print(f"  Coverage: {mentioned}/{total} entities mentioned")

        if result.G < 1.0:
            print("\n" + "─"*80)
            print("⚠️  GENERATION ISSUES:")
            print("─"*80)
            g_details = result.G_details
            if g_details.get('G_gram', 1.0) < 1.0:
                print(f"  Grammar issues detected")
            if g_details.get('G_dia', 1.0) < 1.0:
                print(f"  Diacritic coverage: {g_details.get('dia_coverage', 0):.1%}")
            if g_details.get('cs_rate', 0) > 0:
                print(f"  Code-switching rate: {g_details.get('cs_rate', 0):.1%}")

        # Parse info
        if result.repaired:
            print("\n⚠️  JSON was auto-repaired")

        # Navigation
        if idx < len(results):
            print("\n" + "="*80)
            response = input("Press Enter for next, 'q' to quit, 's' to save current: ")
            if response.lower() == 'q':
                break
            elif response.lower() == 's':
                save_single_result(instance, output, result)


def save_single_result(instance, output, result):
    """Save a single result to a file for inspection"""
    filename = f"debug_{result.instance_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Instance: {result.instance_id}\n")
        f.write(f"Scores: U={result.U:.2f} R={result.R:.2f} G={result.G:.2f} F={result.F:.2f}\n\n")
        f.write("="*80 + "\n")
        f.write("PROMPT:\n")
        f.write("="*80 + "\n")
        f.write(instance.prompt_ro + "\n\n")
        f.write("="*80 + "\n")
        f.write("MODEL OUTPUT:\n")
        f.write("="*80 + "\n")
        f.write(output + "\n\n")
        f.write("="*80 + "\n")
        f.write("EVALUATION DETAILS:\n")
        f.write("="*80 + "\n")
        f.write(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print(f"✓ Saved to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View detailed GMTW-Ro results")
    parser.add_argument("instances", help="JSONL file with instances")
    parser.add_argument("outputs", help="JSONL file with model outputs")
    parser.add_argument(
        "--filter",
        choices=["failed", "passed"],
        help="Show only failed or passed instances"
    )

    args = parser.parse_args()

    view_results(args.instances, args.outputs, args.filter)
