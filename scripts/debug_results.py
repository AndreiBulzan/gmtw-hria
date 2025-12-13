#!/usr/bin/env python3
"""
Debug GMTW-Ro results - shows actual prompts used and detailed failure analysis
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro import Instance, evaluate_instance


def detect_language(outputs_file: str) -> str:
    """Detect which language was used by checking first output"""
    with open(outputs_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            data = json.loads(first_line)
            return data.get('language', 'ro')  # Default to Romanian
    return 'ro'


def debug_results(
    instances_file: str,
    outputs_file: str,
    filter_type: str = None,
    use_languagetool: bool = False,
    use_stanza: bool = False,
):
    """
    Show detailed debugging info for each result

    Args:
        instances_file: JSONL with instances
        outputs_file: JSONL with model outputs
        filter_type: Show only 'failed', 'passed', or None for all
        use_languagetool: Use LanguageTool for grammar checking
        use_stanza: Use Stanza for Romanian lemmatization
    """
    # Detect language
    language = detect_language(outputs_file)
    lang_name = "ROMANIAN" if language == "ro" else "ENGLISH"

    print(f"\n{'='*80}")
    print(f"Detected prompt language: {lang_name}")
    print(f"{'='*80}\n")

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

        result = evaluate_instance(
            instance,
            outputs[inst_id],
            use_languagetool=use_languagetool,
            use_stanza=use_stanza,
        )
        results.append((instance, outputs[inst_id], result))

    # Filter if requested
    if filter_type == "failed":
        results = [(i, o, r) for i, o, r in results if r.U < 0.7 or r.R < 0.7 or r.F < 0.7]
    elif filter_type == "passed":
        results = [(i, o, r) for i, o, r in results if r.U >= 0.7 and r.R >= 0.7 and r.F >= 0.7]

    # Display each result
    for idx, (instance, output, result) in enumerate(results, 1):
        print("\n" + "="*80)
        print(f"INSTANCE {idx}/{len(results)}: {result.instance_id}")
        print("="*80)

        world = instance.world
        status = "✓ PASS" if result.U >= 0.7 and result.R >= 0.7 and result.F >= 0.7 else "✗ FAIL"

        print(f"\n{status}")
        print(f"  U={result.U:.2f}  R={result.R:.2f}  G={result.G:.2f}  F={result.F:.2f}")

        # Show world info
        print(f"\nWorld Type: {world.world_type}")
        print(f"Difficulty: {world.meta.get('difficulty', 'N/A')}")

        if world.world_type == "travel":
            print(f"City: {world.payload['city']}")
            print(f"Days: {world.payload['num_days']}")
            print(f"\nAttractions available:")
            for attr in world.payload['attractions']:
                indoor = "Indoor" if attr['indoor'] else "Outdoor"
                family = "Family-friendly" if attr['family_friendly'] else "Not for kids"
                print(f"  • {attr['name']} ({attr['type']}, {indoor}, {family})")

        elif world.world_type == "schedule":
            print(f"Days: {', '.join(world.payload['days_ro'])}")
            print(f"\nAppointments to schedule:")
            for apt in world.payload['appointments']:
                print(f"  • {apt['name_ro']} (Priority: {apt['priority']})")

        elif world.world_type == "fact":
            print(f"\nFacts in database:")
            for key, value in world.payload['facts'].items():
                print(f"  • {key}: {value}")

        # Show constraints
        print(f"\nConstraints to follow ({len(world.constraints)}):")
        for c in world.constraints:
            if c.type.value == "instruction":
                desc = c.description_en if language == "en" else c.description_ro
                print(f"  → {desc}")

        # Show THE ACTUAL PROMPT USED
        print("\n" + "-"*80)
        print(f"ACTUAL PROMPT ({lang_name}):")
        print("-"*80)
        prompt_used = instance.prompt_en if language == "en" else instance.prompt_ro
        print(prompt_used)

        # Show model output
        print("\n" + "-"*80)
        print("MODEL OUTPUT:")
        print("-"*80)
        print(output)

        # Detailed failure analysis
        if result.U < 1.0:
            print("\n" + "─"*80)
            print("❌ CONSTRAINT VIOLATIONS:")
            print("─"*80)
            for c in result.U_details.get('constraints', []):
                if not c['satisfied']:
                    print(f"  ✗ {c['id']}")
                    # Try to find constraint in world, or use description from result dict
                    matching = [con for con in world.constraints if con.id == c['id']]
                    if matching:
                        actual_desc = matching[0].description_en if language == "en" else matching[0].description_ro
                    else:
                        # Synthetic constraint (e.g., C_FORMAT_JSON_AT_END) - use description from dict
                        actual_desc = c.get('description', 'Format violation')
                    print(f"    Required: {actual_desc}")
                    print(f"    Status: VIOLATED")

        if result.R < 1.0:
            print("\n" + "─"*80)
            print("❌ LOGIC/REASONING FAILURES:")
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
                print(f"  Missing from explanation:")
                # Look up entity names from the world
                for eid in missing:
                    if eid in world.canonical_entities:
                        ent = world.canonical_entities[eid]
                        print(f"    • {eid}: {ent.name}")
                    else:
                        print(f"    • {eid}: (unknown entity)")
                print(f"  → These entities are in the JSON plan but not mentioned in the text")

            total = result.F_details.get('total_count', 0)
            mentioned = result.F_details.get('mentioned_count', 0)
            if total > 0:
                print(f"  Coverage: {mentioned}/{total} entities mentioned")

        if result.G < 0.95:  # Show G issues if notably imperfect
            print("\n" + "─"*80)
            print("⚠️  GENERATION QUALITY NOTES:")
            print("─"*80)
            g = result.G_details

            # Show component scores
            score_parts = f"G_dia={g.get('G_dia', 0):.2f}, G_cs={g.get('G_cs', 0):.2f}, G_len={g.get('G_len', 0):.2f}"
            if g.get('grammar_available') and g.get('G_grammar') is not None:
                score_parts += f", G_grammar={g.get('G_grammar', 0):.2f}"
            print(f"  G={result.G:.2f} ({score_parts})")

            # Diacritic issues
            dia = g.get('diacritic_details', {})
            if dia.get('missing', 0) > 0:
                examples = dia.get('examples', [])
                print(f"  Missing diacritics: {dia.get('missing', 0)} words")
                if examples:
                    print(f"    Words: {', '.join(examples)}")

            # Code-switching issues
            cs = g.get('codeswitch_details', {})
            if cs.get('english_count', 0) > 0:
                print(f"  English words detected: {cs.get('english_count', 0)} ({cs.get('english_rate', 0):.1%})")
                examples = cs.get('examples', [])
                if examples:
                    print(f"    Examples: {', '.join(examples[:5])}")

            # Grammar issues (only if LanguageTool was used)
            if g.get('grammar_available') and g.get('grammar_details'):
                gd = g['grammar_details']
                error_count = gd.get('error_count', 0)
                if error_count > 0:
                    print(f"  Grammar/spelling errors: {error_count}")
                    errors = gd.get('errors', [])
                    for err in errors[:5]:  # Show max 5 errors
                        print(f"    • [{err.get('issue_type', '?')}] {err.get('message', '')}")
                        if err.get('suggestions'):
                            print(f"      Suggestions: {err['suggestions'][:3]}")

            # Flags
            if g.get('is_likely_english'):
                print(f"  ⚠️  Text appears to be in English, not Romanian")
            if g.get('is_too_short'):
                print(f"  ⚠️  Text is too short ({g.get('n_words', 0)} words)")

        # Navigation
        if idx < len(results):
            print("\n" + "="*80)
            response = input("Press Enter for next, 'q' to quit: ")
            if response.lower() == 'q':
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug GMTW-Ro results")
    parser.add_argument("instances", help="JSONL file with instances")
    parser.add_argument("outputs", help="JSONL file with model outputs")
    parser.add_argument(
        "--filter",
        choices=["failed", "passed"],
        help="Show only failed or passed instances"
    )
    parser.add_argument(
        "--use-languagetool",
        action="store_true",
        help="Use LanguageTool for grammar checking"
    )
    parser.add_argument(
        "--use-stanza",
        action="store_true",
        help="Use Stanza for Romanian lemmatization (better entity matching)"
    )

    args = parser.parse_args()

    debug_results(
        args.instances,
        args.outputs,
        args.filter,
        use_languagetool=args.use_languagetool,
        use_stanza=args.use_stanza,
    )
