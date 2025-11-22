#!/usr/bin/env python3
"""Quick peek at model outputs"""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python peek_output.py <outputs.jsonl> [line_number]")
    sys.exit(1)

filename = sys.argv[1]
line_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

with open(filename, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if i == line_num:
            data = json.loads(line)
            print(f"Instance: {data['instance_id']}")
            print(f"Model: {data.get('model', 'unknown')}")
            print(f"Language: {data.get('language', 'unknown')}")
            print("\n" + "="*80)
            print("OUTPUT:")
            print("="*80)
            print(data['output'])
            break
    else:
        print(f"Line {line_num} not found in file")
