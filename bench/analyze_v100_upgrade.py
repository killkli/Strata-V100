#!/usr/bin/env python3
"""Summarize paired, fresh-prompt HTTP measurements without assuming identical answers."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--baseline', type=Path, required=True)
ap.add_argument('--candidate', type=Path, nargs='+', required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()


def load(path):
    data = json.loads(path.read_text())
    rows = data['benchmarks']
    assert rows, f'no measurements: {path}'
    indexed = {}
    for row in rows:
        assert row['usage']['completion_tokens'] == 256, 'output work differs'
        assert row['timings']['predicted_n'] == 256, 'engine output work differs'
        assert row['answer_sha256'] == hashlib.sha256(row['content'].encode()).hexdigest(), 'answer digest differs'
        assert row['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0, 'prompt was reused'
        assert row['timings']['cache_n'] == 0, 'engine reports prompt reuse'
        key = (row['lines'], row['repeat'])
        assert key not in indexed, 'duplicate case'
        indexed[key] = row
    return data, indexed


reference, baseline = load(args.baseline)
summary = {'baseline': reference['label'], 'output_tokens': 256, 'prompt_reuse': 0, 'runs': []}
for path in [args.baseline, *args.candidate]:
    data, rows = load(path)
    assert rows.keys() == baseline.keys(), 'case numbers differ between runs'
    run = {'label': data['label'], 'file': path.name, 'contexts': []}
    for size in sorted({key[0] for key in rows}):
        keys = sorted(key for key in rows if key[0] == size)
        for key in keys:
            assert rows[key]['usage']['prompt_tokens'] == baseline[key]['usage']['prompt_tokens']
        values = {'prompt_tok_s': [rows[k]['timings']['prompt_per_second'] for k in keys],
                  'decode_tok_s': [rows[k]['timings']['predicted_per_second'] for k in keys],
                  'wall_seconds': [rows[k]['wall_seconds'] for k in keys]}
        ref = {'prompt_tok_s': [baseline[k]['timings']['prompt_per_second'] for k in keys],
               'decode_tok_s': [baseline[k]['timings']['predicted_per_second'] for k in keys],
               'wall_seconds': [baseline[k]['wall_seconds'] for k in keys]}
        group = {'prompt_tokens': rows[keys[0]]['usage']['prompt_tokens'], 'cases': len(keys),
                 'median': {name: statistics.median(v) for name, v in values.items()},
                 'median_ratio_change_percent': {name: 100 * (statistics.median(v) / statistics.median(ref[name]) - 1)
                                                 for name, v in values.items()},
                 'paired_median_change_percent': {name: statistics.median(100 * (v / b - 1) for v, b in zip(vals, ref[name]))
                                                  for name, vals in values.items()},
                 'answer_hash_matches': sum(rows[k].get('answer_sha256') == baseline[k].get('answer_sha256') for k in keys),
                 'median_drafts': statistics.median(rows[k]['timings']['draft_n'] for k in keys),
                 'median_accepted_drafts': statistics.median(rows[k]['timings']['draft_n_accepted'] for k in keys)}
        run['contexts'].append(group)
    summary['runs'].append(run)
args.out.write_text(json.dumps(summary, indent=2) + '\n')
print('Saved ' + str(args.out))
