#!/usr/bin/env python3
"""
Evaluate model outputs for GMTW instances (multi-language)

Supports Romanian, English, and German evaluation with automatic
language detection. New languages can be added via the registry.

Optional enhancements:
  --use-languagetool  Include LanguageTool grammar checking in G score
                      (requires: pip install language-tool-python)
  --use-stanza        Use Stanza for Romanian lemmatization in F score
                      (requires: pip install stanza)
  --language          Force language (ro/en/de), otherwise auto-detect
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro import Instance
from rombench.registry import (
    create_evaluator as registry_create_evaluator,
    get_supported_languages,
    get_language_name,
)


def detect_language(outputs_file: str) -> str:
    """
    Detect which language was used by checking first output.

    Returns language code (e.g., 'ro', 'en', 'de')
    """
    with open(outputs_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            data = json.loads(first_line)
            return data.get('language', 'ro')  # Default to Romanian
    return 'ro'


def create_evaluator(language: str, use_languagetool: bool, use_stanza: bool):
    """Create appropriate evaluator based on language using registry."""
    kwargs = {'use_languagetool': use_languagetool}
    if language == 'ro':
        kwargs['use_stanza'] = use_stanza
    return registry_create_evaluator(language, **kwargs)


def evaluate_batch(
    instances_file: str,
    outputs_file: str,
    output_metrics: str = None,
    use_languagetool: bool = False,
    use_stanza: bool = False,
    force_language: str = None,
):
    """
    Evaluate a batch of model outputs

    Args:
        instances_file: JSONL file with instances
        outputs_file: JSONL file with model outputs (format: {"instance_id": "...", "output": "..."})
        output_metrics: Optional file to save detailed metrics
        use_languagetool: If True, include LanguageTool grammar checking in G score
        use_stanza: If True, use Stanza for Romanian lemmatization in F score (Romanian only)
        force_language: Force language (ro/en), otherwise auto-detect
    """
    print('here')
    # Detect or use forced language
    if force_language:
        language = force_language
    else:
        language = detect_language(outputs_file)
    
    lang_name = get_language_name(language)
    print(f"Detected language: {lang_name} ({language})")
    print(f"Supported languages: {', '.join(get_supported_languages())}")

    # Print mode info
    modes = []
    if use_languagetool:
        modes.append("LanguageTool (G_grammar)")
    if use_stanza and language == 'ro':
        modes.append("Stanza (F lemmatization)")
    if modes:
        print(f"Enhanced mode: {', '.join(modes)}")
    else:
        print("Standard mode (fast, no optional dependencies)")

    # Create language-specific evaluator
    evaluator = create_evaluator(language, use_languagetool, use_stanza)

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
    missing_count = 0
    for inst_id, instance in instances.items():
        if inst_id not in outputs:
            print(f"✗ {inst_id}: MISSING (model refused/failed) → U=0.00 G=0.00 F=0.00")
            missing_count += 1
            # Create a zero-score result for missing outputs
            zero_result = {
                'U': 0.0, 'R': 0.0, 'G': 0.0, 'F': 0.0,
                'U_details': {'U_constraints': 0.0, 'U_format': 0.0},
                'instance_id': inst_id,
                'missing': True,
                'language': language,
            }
            # Wrap in simple object for attribute access
            class ZeroResult:
                def __init__(self, d):
                    self.__dict__.update(d)
                def to_dict(self):
                    return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            results.append(ZeroResult(zero_result))
            continue

        result = evaluator.evaluate_output(instance, outputs[inst_id])
        results.append(result)

        # Print individual result
        status = "✓" if result.U > 0.7 and result.R > 0.7 else "✗"
        print(f"{status} {inst_id}: U={result.U:.2f} R={result.R:.2f} G={result.G:.2f} F={result.F:.2f}")

    print('here')
    # Compute averages
    if results:
        avg_U = sum(r.U for r in results) / len(results)
        avg_G = sum(r.G for r in results) / len(results)
        avg_F = sum(r.F for r in results) / len(results)

        # Also compute U sub-components for detailed analysis
        avg_U_constraints = sum(r.U_details.get("U_constraints", r.U) for r in results) / len(results)
        avg_U_format = sum(r.U_details.get("U_format", 1.0) for r in results) / len(results)

        # Compute final score (weighted average)
        # U=50% (main discriminator), G=25%, F=25%
        # R is deprecated (integrated into U)
        final_score = (0.50 * avg_U + 0.25 * avg_G + 0.25 * avg_F)

        print("\n" + "="*60)
        print(f"AVERAGE SCORES ({len(results)} instances, {lang_name})")
        print("="*60)
        if missing_count > 0:
            print(f"  MISSING OUTPUTS:      {missing_count} (scored as 0)")
        print(f"  U (Understanding):      {avg_U:.3f}")
        print(f"    - U_constraints:      {avg_U_constraints:.3f}  (85% of U)")
        print(f"    - U_format:           {avg_U_format:.3f}  (15% of U)")
        print(f"  G (Generation):         {avg_G:.3f}")
        print(f"  F (Faithfulness):       {avg_F:.3f}")
        print("="*60)
        print(f"  FINAL SCORE:            {final_score:.1%}")
        print(f" 50% x U + 25% x G + 25% x F)")
        print("="*60)

        # If there were missing outputs, also show score excluding them
        if missing_count > 0:
            answered_results = [r for r in results if not getattr(r, 'missing', False)]
            if answered_results:
                ans_U = sum(r.U for r in answered_results) / len(answered_results)
                ans_G = sum(r.G for r in answered_results) / len(answered_results)
                ans_F = sum(r.F for r in answered_results) / len(answered_results)
                ans_score = (0.50 * ans_U + 0.25 * ans_G + 0.25 * ans_F)
                print(f"\n  (If ignoring {missing_count} missing: {ans_score:.1%} on {len(answered_results)} answered)")

        # Save detailed metrics if requested
        if output_metrics:
            with open(output_metrics, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
            print(f"\nDetailed metrics saved to {output_metrics}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate GMTW outputs (Romanian or English)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Optional enhancements (require additional dependencies):

            --use-languagetool  Include grammar checking in G score
                                Install: pip install language-tool-python
                                Effect: Adds G_grammar component

            --use-stanza        Use Stanza for Romanian lemmatization in F score
                                Install: pip install stanza
                                Effect: More accurate entity matching

            --language          Force language (ro/en), otherwise auto-detect from outputs

            Examples:
            # Standard evaluation (fast)
            python evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs.jsonl

            # With grammar checking
            python evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs.jsonl --use-languagetool

            # Force English evaluation
            python evaluate_outputs.py data/gmtw_ro_v0.jsonl data/outputs_70b_en.jsonl --language en
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
    parser.add_argument(
        "--language",
        choices=get_supported_languages(),
        help="Force language (ro/en/de), otherwise auto-detect",
    )

    args = parser.parse_args()
    print('here')
    evaluate_batch(
        args.instances,
        args.outputs,
        args.save_metrics,
        use_languagetool=args.use_languagetool,
        use_stanza=args.use_stanza,
        force_language=args.language,
    )