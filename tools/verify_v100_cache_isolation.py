#!/usr/bin/env python3
"""Check two streaming conversations through cache reuse and a history branch."""
import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
cfg = json.loads(args.config.read_text())
base = 'http://127.0.0.1:' + str(cfg.get('port', 8080))
markers = ['CACHE_ALDER_72914', 'CACHE_BIRCH_83526']
nonce = uuid.uuid4().hex
bases = []
for client in range(2):
    text = [f'Cache test {nonce}, client {client}.']
    text += [f'Record {i}: inventory quantity={1000+i}, status=stored.' for i in range(180)]
    text += [f'verification_code = {markers[client]}',
             'Reply with only the exact verification_code. Remember it for later questions.']
    bases.append([{'role': 'user', 'content': '\n'.join(text)}])

result = {'passed': False, 'rounds': []}
first_answers = [None, None]
for round_id in range(3):
    barrier = threading.Barrier(2)

    def run(client):
        messages = list(bases[client])
        if round_id:
            # Both follow-ups branch from the first answer. Round two must find a
            # historical prefix after the other conversation has used the engine.
            messages += [{'role': 'assistant', 'content': first_answers[client]},
                         {'role': 'user', 'content':
                          ('Repeat the verification_code, with no extra text.' if round_id == 1 else
                           'For this alternative follow-up, return only the same verification_code.')}]
        payload = {'model': cfg['model_name'], 'messages': messages, 'max_tokens': 64,
                   'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True},
                   'chat_template_kwargs': {'enable_thinking': False}}
        req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': 'Bearer ' + cfg['api_key']})
        barrier.wait(timeout=30)
        started = time.monotonic()
        content, usage, done = [], None, False
        with urllib.request.urlopen(req, timeout=600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data: '):
                    continue
                if line[6:] == '[DONE]':
                    done = True
                    continue
                frame = json.loads(line[6:])
                if frame.get('usage'):
                    usage = frame['usage']
                content += [c.get('delta', {}).get('content') or '' for c in frame.get('choices', [])]
        answer = ''.join(content)
        assert done and markers[client] in answer and markers[1-client] not in answer, answer
        assert usage, 'stream omitted usage'
        return {'client': client, 'content': answer, 'seconds': round(time.monotonic()-started, 3),
                'usage': usage, 'done': done}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(run, [0, 1]))
    if round_id == 0:
        first_answers = [r['content'] for r in replies]
    result['rounds'].append({'round': round_id, 'replies': replies})
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'cache isolation round {round_id}: both streams passed', flush=True)

assert any(r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) > 0
           for row in result['rounds'][1:] for r in row['replies']), 'no cache reuse was exercised'
result.update(passed=True, reuse_observed=True)
args.out.write_text(json.dumps(result, indent=2) + '\n')
print('cache isolation: PASS, including reuse and an alternative history branch', flush=True)
