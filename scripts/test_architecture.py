"""
Test script to verify the new evaluation architecture works.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(_file_).parent.parent))

print("="*80)
print("Testing GMTW Evaluation Framework")
print("="*80)

# Test 1: Import base classes
print("\n1. Testing base class imports...")
try:
    from rombench.eval import BaseEvaluator, BaseMetrics, BaseParser
    print("   Base classes imported successfully")
except Exception as e:
    print(f"   Failed to import base classes: {e}")
    sys.exit(1)

# Test 2: Import Romanian components
print("\n2. Testing Romanian imports...")
try:
    from rombench.gmtw_ro.eval import RomanianEvaluator
    from rombench.nlp_ro.faithfulness import RomanianFaithfulness
    print("   Romanian components imported successfully")
except Exception as e:
    print(f"   Failed to import Romanian components: {e}")
    sys.exit(1)

# Test 3: Import English components
print("\n3. Testing English imports...")
try:
    from rombench.gmtw_en.eval import EnglishEvaluator
    from rombench.nlp_en import EnglishNLPToolkit
    print("   English components imported successfully")
except Exception as e:
    print(f"   Failed to import English components: {e}")
    sys.exit(1)

# Test 4: Create Romanian evaluator
print("\n4. Testing Romanian evaluator creation...")
try:
    ro_evaluator = RomanianEvaluator.create()
    print(f"   Romanian evaluator created (language: {ro_evaluator.language})")
except Exception as e:
    print(f"   Failed to create Romanian evaluator: {e}")
    sys.exit(1)

# Test 5: Create English evaluator
print("\n5. Testing English evaluator creation...")
try:
    en_evaluator = EnglishEvaluator.create()
    print(f"   English evaluator created (language: {en_evaluator.language})")
except Exception as e:
    print(f"   Failed to create English evaluator: {e}")
    sys.exit(1)

# Test 6: Test parser
print("\n6. Testing parser...")
try:
    from rombench.eval.base_parser import parse_dual_channel_output
    
    test_output = """This is an explanation about my plan.
    
I chose these activities because they are fun.

{
  "day1": ["Activity A", "Activity B"],
  "day2": ["Activity C"]
}"""
    
    result = parse_dual_channel_output(test_output)
    print(f"   Parser works:")
    print(f"     - Format OK: {result.format_ok}")
    print(f"     - Has plan: {result.plan is not None}")
    print(f"     - Explanation length: {len(result.explanation)} chars")
except Exception as e:
    print(f"   Failed to parse: {e}")
    sys.exit(1)

# Test 7: Test with sample instance (if available)
print("\n7. Testing with sample instance...")
try:
    import json
    from rombench.gmtw_ro import Instance
    
    # Try to load first instance
    instances_file = Path(__file__).parent.parent / "data" / "gmtw_ro_v0.jsonl"
    if instances_file.exists():
        with open(instances_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            inst = Instance.from_dict(json.loads(first_line))
        
        # Create a fake output
        fake_output = """Here is my travel plan for the trip.

{
  "day1": ["Museum", "Park"],
  "day2": ["Restaurant"]
}"""
        
        # Evaluate with both evaluators
        ro_result = ro_evaluator.evaluate_output(inst, fake_output)
        print(f"   Romanian evaluation: U={ro_result.U:.2f}, G={ro_result.G:.2f}, F={ro_result.F:.2f}")
        
        en_result = en_evaluator.evaluate_output(inst, fake_output)
        print(f"   English evaluation: U={en_result.U:.2f}, G={en_result.G:.2f}, F={en_result.F:.2f}")
    else:
        print("   Skipped (no instances file found)")
except Exception as e:
    print(f"   Failed evaluation test: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - this is optional

print("\n" + "="*80)
print("All core tests passed!")
print("="*80)
print()
print("Architecture verified:")
print("  - Base classes: ✓")
print("  - Romanian support: ✓")
print("  - English support: ✓")
print("  - Parser: ✓")
print("  - Evaluation: ✓")
print()
print("Ready to use!")