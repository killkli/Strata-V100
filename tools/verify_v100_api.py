#!/usr/bin/env python3
"""Exercise the existing HTTP contract against the real resident model. Never print credentials."""
import argparse
import concurrent.futures
import json
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'strata-unsloth-ud-iq4_xs.json'))
ap.add_argument('--out', required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--mode', choices=['smoke', 'bench', 'all'], default='smoke')
args = ap.parse_args()
cfg = json.loads(Path(args.config).read_text())
base = 'http://127.0.0.1:' + str(cfg.get('port', 8080))
model = cfg['model_name']
result = {'label': args.label, 'model': model, 'checks': [], 'benchmarks': []}
output = Path(args.out)

def save():
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')

def call(path, payload=None, auth=True, stream=False):
    headers = {'Content-Type': 'application/json'}
    if auth:
        headers['Authorization'] = 'Bearer ' + cfg['api_key']
    data = None if payload is None else json.dumps(payload).encode()
    start = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=900) as r:
        if not stream:
            return json.load(r), time.monotonic() - start
        frames, done = [], False
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data: '):
                continue
            chunk = line[6:]
            if chunk == '[DONE]':
                done = True
            else:
                frames.append(json.loads(chunk))
        return {'frames': frames, 'done': done}, time.monotonic() - start

def chat(text, **extra):
    p = {'model': model, 'messages': [{'role': 'user', 'content': text}],
         'max_tokens': 128, 'temperature': 0, 'top_p': 0.95,
         'chat_template_kwargs': {'enable_thinking': False}}
    p.update(extra)
    return p

def record(name, response, elapsed):
    result['checks'].append({'name': name, 'passed': True, 'seconds': round(elapsed, 3), 'response': response})
    save()
    print(name + ': PASS', flush=True)

if args.mode in ('smoke', 'all'):
    h, dt = call('/health')
    assert h.get('status') in ('ok', 'ready'), h
    record('health', h, dt)
    models, dt = call('/v1/models')
    assert model in [m['id'] for m in models['data']], models
    record('model_alias', models, dt)
    try:
        call('/v1/models', auth=False)
        raise AssertionError('API authentication unexpectedly disabled')
    except urllib.error.HTTPError as e:
        assert e.code in (401, 403), e.code
        record('authentication_required', {'status': e.code}, 0)
    reply, dt = call('/v1/chat/completions', chat('Reply with the exact word STRATA_OK, without explanation.'))
    assert reply['object'] == 'chat.completion'
    assert reply['model'] == model
    assert reply['choices'][0]['message'].get('content'), reply
    assert reply['usage']['completion_tokens'] > 0
    record('openai_nonstream', reply, dt)
    stream, dt = call('/v1/chat/completions', chat('請用繁體中文簡短說明 PCIe 是什麼。', stream=True,
                       stream_options={'include_usage': True}), stream=True)
    assert stream['done'] and stream['frames'], stream
    assert any(f.get('usage') for f in stream['frames']), stream
    assert any(c.get('delta', {}).get('content') for f in stream['frames'] for c in f.get('choices', [])), stream
    record('openai_sse_and_usage', stream, dt)
    tool = {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get the current weather for a city.',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}
    reply, dt = call('/v1/chat/completions', chat('Call get_weather with city Taipei. Do not answer in prose.',
                       tools=[tool], tool_choice={'type': 'function', 'function': {'name': 'get_weather'}}))
    calls = reply['choices'][0]['message'].get('tool_calls', [])
    assert calls and calls[0]['function']['name'] == 'get_weather', reply
    assert json.loads(calls[0]['function']['arguments']).get('city'), reply
    record('openai_tool_call', reply, dt)
    followup = chat('What is the weather in Taipei?')
    followup['messages'] += [reply['choices'][0]['message'],
        {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '{"city":"Taipei","temperature_c":25,"condition":"sunny"}'}]
    followup['tools'] = [tool]
    followup['tool_choice'] = 'none'
    reply, dt = call('/v1/chat/completions', followup)
    assert reply['choices'][0]['message'].get('content'), reply
    record('openai_tool_result_roundtrip', reply, dt)
    reply, dt = call('/v1/messages', {'model': model, 'max_tokens': 128, 'temperature': 0,
                       'messages': [{'role': 'user', 'content': 'Reply briefly: what is 2+2?'}]})
    assert reply['type'] == 'message' and reply['role'] == 'assistant' and reply['content'], reply
    record('anthropic_messages', reply, dt)
    reply, dt = call('/v1/responses', {'model': model, 'input': 'Reply briefly with the result of 3+4.',
                       'max_output_tokens': 128, 'temperature': 0})
    assert reply['object'] == 'response' and reply['output'], reply
    record('responses_api', reply, dt)
    def concurrent_request(n):
        reply, dt = call('/v1/chat/completions', chat(f'Reply with only CONCURRENT_{n}.'))
        assert reply['choices'][0]['message'].get('content'), reply
        return {'request': n, 'seconds': round(dt, 3), 'reply': reply}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(concurrent_request, [1, 2]))
    record('two_simultaneous_clients', replies, 0)

    # Distinct sentinels catch cross-slot output reuse (a nonempty reply alone cannot).
    barrier = threading.Barrier(2)
    markers = ['SLOT_ALDER_72914', 'SLOT_BIRCH_83526']
    def isolated_stream(n):
        barrier.wait(timeout=30)
        reply, dt = call('/v1/chat/completions', chat(
            f'First write exactly {markers[n]}. Then list all prime numbers below 100.',
            stream=True, stream_options={'include_usage': True}, max_tokens=192), stream=True)
        content = ''.join(c.get('delta', {}).get('content') or ''
                          for f in reply['frames'] for c in f.get('choices', []))
        assert reply['done'] and markers[n] in content and markers[1-n] not in content, content
        return {'request': n, 'seconds': round(dt, 3), 'content': content, 'done': reply['done']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(isolated_stream, [0, 1]))
    record('two_streams_keep_distinct_output', replies, 0)

if args.mode in ('bench', 'all'):
    def compilers_running():
        names = {'ninja', 'nvcc', 'cicc', 'ptxas', 'cc1plus', 'cc1'}
        for p in Path('/proc').glob('[0-9]*/comm'):
            try:
                if p.read_text().strip() in names:
                    return True
            except OSError:
                pass
        return False
    if compilers_running():
        print('Waiting for compilation to finish before timing measurements', flush=True)
        while compilers_running():
            time.sleep(3)
    for size in (90, 360):
        for repeat in range(3):
            rng = random.Random(size * 100 + repeat)
            lines = [f'Case {size}-{repeat}: inspect these records.']
            for i in range(size):
                lines.append(f'row_{i}: key={rng.randrange(100000,999999)}, value={rng.randrange(100000,999999)}')
            lines.append('Write a detailed Python function to parse these rows and summarize key/value statistics. Include code and an explanation.')
            reply, dt = call('/v1/chat/completions', chat('\n'.join(lines), max_tokens=256))
            assert reply['choices'][0]['message'].get('content'), reply
            row = {'lines': size, 'repeat': repeat, 'wall_seconds': round(dt, 3),
                   'usage': reply['usage'], 'timings': reply.get('timings'),
                   'finish_reason': reply['choices'][0]['finish_reason']}
            result['benchmarks'].append(row)
            save()
            print('benchmark: ' + json.dumps(row), flush=True)
save()
print('Saved ' + str(output), flush=True)
