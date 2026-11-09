#!/usr/bin/env python3
"""Opt-in real-model checks for long-context recall and the existing vision API."""
import argparse
import base64
import io
import json
import random
import time
import urllib.request
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'strata-unsloth-ud-iq4_xs.json')
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
cfg = json.loads(args.config.read_text())
base = 'http://127.0.0.1:' + str(cfg.get('port', 8080))
results = []

def request(messages, max_tokens=128):
    payload = {'model': cfg['model_name'], 'messages': messages, 'max_tokens': max_tokens,
               'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
    r = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type':'application/json',
        'Authorization':'Bearer '+cfg['api_key']})
    start = time.monotonic()
    with urllib.request.urlopen(r, timeout=900) as response:
        reply = json.load(response)
    return reply, time.monotonic()-start

def record(name, reply, elapsed):
    results.append({'name':name, 'passed':True, 'seconds':round(elapsed,3), 'response':reply})
    args.out.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    print(name+': PASS '+str(reply.get('usage')), flush=True)

rng = random.Random(20261109)
lines = ['Find the verification_code in these records. At the end reply with only its exact value.']
for i in range(2700):
    if i == 1350:
        lines.append('verification_code = C7K9P2')
    lines.append(f'row_{i}: key={rng.randrange(100000,999999)}, value={rng.randrange(100000,999999)}')
lines.append('What is the verification_code? Reply with only the exact code.')
reply, elapsed = request([{'role':'user','content':'\n'.join(lines)}])
assert reply['usage']['prompt_tokens'] > 60000, reply['usage']
assert 'C7K9P2' in reply['choices'][0]['message'].get('content',''), reply
record('long_context_needle_over_60k', reply, elapsed)

image = Image.new('RGB', (224,224), (255,0,0))
buffer = io.BytesIO()
image.save(buffer, format='PNG')
url = 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
reply, elapsed = request([{'role':'user','content':[
    {'type':'text','text':'What is the dominant color in this image? Reply with one English color word.'},
    {'type':'image_url','image_url':{'url':url}}]}])
assert 'red' in reply['choices'][0]['message'].get('content','').lower(), reply
record('existing_vision_encoder_via_openai_api', reply, elapsed)
