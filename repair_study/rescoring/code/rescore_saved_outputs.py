#!/usr/bin/env python3
"""Apply the unchanged proposal scorer to one file of saved model responses.

One checkpoint, language and training seed per input file. The frozen cohort
must be covered completely. No inference, training or network calls are made.
"""
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from rescore import ROOT, Instance, ci, dataset_audit, score

WORLD_FILES = {'standard': 'v0_DE_test.jsonl', 'adversarial': 'hard.jsonl'}
METRICS = ('T', 'raw_ok', 'normalized_ok', 'recovered', 'schema', 'joint')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    provenance = json.loads((ROOT / 'PROVENANCE.json').read_text(encoding='utf-8'))
    for name, info in provenance['source_files'].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != info['sha256']:
            raise ValueError(f'Bundled source differs from the proposal: {name}')
    for package, required in provenance['dependencies'].items():
        actual = importlib.metadata.version(package)
        if actual != required:
            raise ValueError(f'{package}=={required} is required for this reproduction; found {actual}')


def load_cohort(split):
    raw = [json.loads(line) for line in
           (ROOT / 'inputs' / WORLD_FILES[split]).read_text(encoding='utf-8').splitlines()
           if line.strip()]
    instances = {r['instance_id']: Instance.from_dict(r) for r in raw}
    if len(instances) != len(raw):
        raise ValueError('Duplicate IDs in the task definitions')
    with (ROOT / 'cohorts/cohort_exclusions.csv').open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == split]
    manifest = {r['instance_id']: r for r in rows}
    if len(manifest) != len(rows) or set(manifest) != set(instances):
        raise ValueError('The frozen cohort manifest does not match the task definitions')
    # Check against the existing audit, but never choose a new cohort from outputs.
    for audited in dataset_audit(raw):
        r = manifest[audited['instance_id']]
        reasons = [s.strip() for s in r['exclusion_reasons'].split(';') if s.strip()]
        if (r['eligible'] != str(audited['eligible']) or
                r['world_hash'] != audited['world_hash'] or
                (r['duplicate_of'] or None) != audited['duplicate_of'] or
                reasons != audited['exclusion_reasons']):
            raise ValueError(f"Audit differs from frozen manifest: {r['instance_id']}")
    retained = {r['instance_id'] for r in rows if r['eligible'] == 'True'}
    for label, expected in [('retained', retained), ('excluded', set(instances) - retained)]:
        listed = (ROOT / 'cohorts' / f'{split}_{label}_ids.txt').read_text(encoding='utf-8').splitlines()
        if len(listed) != len(set(listed)) or set(listed) != expected:
            raise ValueError(f'{split} {label} ID list does not match the manifest')
    expected_n = 438 if split == 'standard' else 240
    if len(retained) != expected_n:
        raise ValueError(f'Expected {expected_n} retained tasks, found {len(retained)}')
    return instances, retained


def load_outputs(path, instances, retained, id_field='instance_id', output_field='output'):
    outputs = {}
    metadata = collections.defaultdict(set)
    with path.open(encoding='utf-8-sig') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL record on line {line_number}: {exc.msg}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Line {line_number} must be a JSON object')
            ident = row.get(id_field)
            if not isinstance(ident, str) or ident not in instances:
                raise ValueError(f'Unknown task ID on line {line_number}: {ident!r}; check the split')
            if ident in outputs:
                raise ValueError(f'Duplicate task ID: {ident}; use one checkpoint/language/seed per file')
            if output_field not in row or not isinstance(row[output_field], str):
                raise ValueError(f'{ident}: {output_field!r} must contain the raw response string; use "" for a saved blank response')
            outputs[ident] = {'output': row[output_field], 'source_line': line_number}
            for key in ('model', 'model_id', 'checkpoint', 'language', 'lang', 'seed', 'training_seed'):
                if key in row:
                    metadata[key].add(json.dumps(row[key], sort_keys=True, ensure_ascii=False))
    missing = sorted(retained - set(outputs))
    if missing:
        raise ValueError(f'Missing {len(missing)} retained task IDs, including {missing[:5]}; no smaller cohort will be scored')
    mixed = [key for key, values in metadata.items() if len(values) > 1]
    if mixed:
        raise ValueError('Mixed run metadata in one file: ' + ', '.join(mixed))
    return outputs, {key: json.loads(next(iter(values))) for key, values in metadata.items()}


def interval(values):
    result = ci(values)
    return {key: result[key] for key in ('mean', 'low', 'high')}


def summarize(chosen):
    result = {'n': len(chosen)}
    for metric in METRICS:
        result[metric] = interval([r[metric] for r in chosen])
    factual = [r['QA'] for r in chosen if r['QA'] is not None]
    result['QA_n'] = len(factual)
    result['QA'] = interval(factual)
    result['families'] = {}
    for family in ('travel', 'schedule', 'fact', 'recipe'):
        selected = [r for r in chosen if r['family'] == family]
        result['families'][family] = {
            'n': len(selected),
            'T': float(np.mean([r['T'] for r in selected])),
            'joint': float(np.mean([r['joint'] for r in selected])),
        }
    result['parse_stage_counts'] = dict(sorted(collections.Counter(r['parse_stage'] for r in chosen).items()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=tuple(WORLD_FILES), required=True)
    parser.add_argument('--outputs', type=Path, required=True, help='JSONL file for one checkpoint, language and seed')
    parser.add_argument('--out-dir', type=Path, required=True, help='New directory for results; existing directories are not overwritten')
    parser.add_argument('--id-field', default='instance_id', help='Top-level task-ID field')
    parser.add_argument('--output-field', default='output', help='Top-level raw-response field')
    parser.add_argument('--label', help='Optional run label, e.g. RoLlama31_repaired_seed0_ro')
    args = parser.parse_args()
    try:
        if args.out_dir.exists():
            raise ValueError('Output directory already exists; choose a new directory')
        verify_sources()
        instances, retained = load_cohort(args.split)
        source_digest = sha256(args.outputs)
        outputs, metadata = load_outputs(args.outputs, instances, retained, args.id_field, args.output_field)
        if sha256(args.outputs) != source_digest:
            raise ValueError('Input file changed while reading')
        records = [dict(score(instances[ident], data['output']), eligible=ident in retained,
                        source_line=data['source_line']) for ident, data in sorted(outputs.items())]
        chosen = [r for r in records if r['eligible']]
        summary = summarize(chosen)
        summary.update(
            split=args.split, label=args.label or args.outputs.stem,
            input_file=str(args.outputs.resolve()), input_sha256=source_digest,
            input_id_field=args.id_field, input_output_field=args.output_field,
            input_records=len(records), excluded_records_present=len(records) - len(chosen),
            input_run_metadata=metadata,
            scorer_sha256=sha256(ROOT / 'code/rescore.py'),
            cohort_manifest_sha256=sha256(ROOT / 'cohorts/cohort_exclusions.csv'),
            world_file_sha256=sha256(ROOT / 'inputs' / WORLD_FILES[args.split]),
            versions={p: importlib.metadata.version(p) for p in ('numpy', 'json-repair')},
            python=sys.version,
            bootstrap={'resamples': 5000, 'seed': 20240617, 'interval': 'pointwise percentile 95%',
                       'scope': 'tasks within this single saved run; no between-training-seed uncertainty'},
            scale='All summary means and bounds use 0-1, not percentages.',
        )
        args.out_dir.mkdir(parents=True, exist_ok=False)
        with (args.out_dir / 'per_instance.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        (args.out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"Scored {len(chosen)} retained {args.split} tasks; T={summary['T']['mean']:.6f}, J={summary['joint']['mean']:.6f}")
        print(f'Results: {args.out_dir}')
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
