#!/usr/bin/env python3
"""
Evaluate model outputs for GMTW-Ro instances

Optional enhancements:
  --use-languagetool  Include LanguageTool grammar checking in G score
                      (requires: pip install language-tool-python)
  --use-stanza        Use Stanza for Romanian lemmatization in F score
                      (requires: pip install stanza)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro import Instance, evaluate_instance


def evaluate_batch(
    instances_file: str,
    outputs_file: str,
    output_metrics: str = None,
    use_languagetool: bool = False,
    use_stanza: bool = False,
):
    """
    Evaluate a batch of model outputs

    Args:
        instances_file: JSONL file with instances
        outputs_file: JSONL file with model outputs (format: {"instance_id": "...", "output": "..."})
        output_metrics: Optional file to save detailed metrics
        use_languagetool: If True, include LanguageTool grammar checking in G score
        use_stanza: If True, use Stanza for Romanian lemmatization in F score
    """

    # Print mode info
    modes = []
    if use_languagetool:
        modes.append("LanguageTool (G_grammar)")
    if use_stanza:
        modes.append("Stanza (F lemmatization)")
    if modes:
        print(f"Enhanced mode: {', '.join(modes)}")
    else:
        print("Standard mode (fast, no optional dependencies)")

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
            print(f" Missing output for {inst_id}")
            continue

        result = evaluate_instance(
            instance,
            outputs[inst_id],
            use_languagetool=use_languagetool,
            use_stanza=use_stanza,
        )
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

    parser = argparse.ArgumentParser(
        description="Evaluate GMTW-Ro outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optional enhancements (require additional dependencies):

  --use-languagetool  Include grammar checking in G score
                      Install: pip install language-tool-python
                      Effect: Adds G_grammar component (25% of G score)

  --use-stanza        Use Stanza for Romanian lemmatization in F score
                      Install: pip install stanza
                      Effect: More accurate entity matching in explanations

Examples:
  # Standard evaluation (fast)
  python evaluate_outputs.py data/instances.jsonl data/outputs.jsonl

  # With grammar checking
  python evaluate_outputs.py data/instances.jsonl data/outputs.jsonl --use-languagetool

  # Full enhanced mode
  python evaluate_outputs.py data/instances.jsonl data/outputs.jsonl --use-languagetool --use-stanza
"""
    )
    parser.add_argument("instances", help="JSONL file with instances")
    parser.add_argument("outputs", help="JSONL file with model outputs")
    parser.add_argument("--save-metrics", help="Save detailed metrics to file")
    parser.add_argument(
        "--use-languagetool",
        action="store_true",
        help="Include LanguageTool grammar checking in G score (slower)"
    )
    parser.add_argument(
        "--use-stanza",
        action="store_true",
        help="Use Stanza for Romanian lemmatization in F score (slower)"
    )

    args = parser.parse_args()

    evaluate_batch(
        args.instances,
        args.outputs,
        args.save_metrics,
        use_languagetool=args.use_languagetool,
        use_stanza=args.use_stanza,
    )
