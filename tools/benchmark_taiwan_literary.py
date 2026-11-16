#!/usr/bin/env python3
"""Measure real Taiwanese Traditional Chinese translations through the unchanged SSE API."""
import argparse
import concurrent.futures
import hashlib
import json
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = """你是一位臺灣文學小說譯者。請將使用者提供的英文經典小說完整翻譯，並改寫成臺灣讀者自然流暢的繁體中文小說正文。
使用臺灣常用詞彙、繁體字及中文標點，對話使用「」；避免簡體字與中國大陸慣用語。
忠實保留原作所有情節、人物關係、細節、敘事人稱、諷刺及情緒，調整生硬的英文句構，保持文學性；維持原作時代背景與文化，不改成本土故事。
人名採自然一致的中文譯名，保留原段落與對話次序。不得摘要、刪段、增添原文沒有的劇情。
保留原作的貨幣、計量單位與歷史背景，不得換算成本地貨幣或使用現代網路語；原文未註明幣別時，不要自行添加美元、鎂或任何貨幣符號。
只輸出翻譯改寫後的正文，不要標題、說明、英文原文、註解、識別碼或思考過程。"""
SIMPLIFIED = set('这们说为从过国时对还发见没让样开会给实学长间话书门风无关听声来当点气亲爱将动头边东万欢进')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'strata-unsloth-ud-iq4_xs.json')
    ap.add_argument('--passages', type=Path, default=ROOT/'bench/results/2026-11-09-taiwan-literary/passages.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--replicas', type=int, default=1)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    passages = json.loads(a.passages.read_text())
    jobs = [(book, replica) for replica in range(a.replicas) for book in passages]
    a.out.mkdir(parents=True, exist_ok=False)
    base = 'http://127.0.0.1:' + str(cfg.get('port', 8080))
    headers = {'Content-Type':'application/json', 'Authorization':'Bearer '+cfg['api_key']}
    def get(route):
        with urllib.request.urlopen(urllib.request.Request(base+route, headers=headers),timeout=10) as r:
            return json.load(r)
    before = get('/v1/status')
    assert before['loaded'] and not before['activity']['in_flight'], before['activity']
    samples, errors, records = [], [], []
    stop = threading.Event()
    def monitor():
        tick = 0
        while not stop.is_set():
            try:
                slots = get('/slots')
                status = get('/v1/status')
                sample = {'at_s':time.perf_counter(), 'active_slots':sum(bool(x['is_processing']) for x in slots),
                          'in_flight':status['activity']['in_flight']}
                if tick % 4 == 0:
                    sample['gpus_csv'] = subprocess.check_output(['nvidia-smi',
                        '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                samples.append(sample)
            except Exception as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc))
            tick += 1
            stop.wait(.5)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    clock = {}
    barrier = threading.Barrier(len(jobs)+1, action=lambda:clock.update(start=time.perf_counter()))
    def translate(book, replica):
        identifier = f"{book['id']}-r{replica:02d}"
        nonce = uuid.uuid4().hex
        payload = {'model':cfg['model_name'], 'messages':[
            {'role':'system','content':f'工作識別碼 {nonce}（僅供排程，不得輸出）。\n'+SYSTEM},
            {'role':'user','content':book['english']}],
            'max_tokens':3072,'temperature':.2,'top_p':.9,'seed':20261109,
            'stream':True,'stream_options':{'include_usage':True},
            'chat_template_kwargs':{'enable_thinking':False}}
        barrier.wait()
        started = time.perf_counter()
        content, reasoning = [], []
        first = last = None
        usage = timings = None
        finish = None
        done = False
        try:
            req = urllib.request.Request(base+'/v1/chat/completions', data=json.dumps(payload).encode(),headers=headers)
            with urllib.request.urlopen(req,timeout=2400) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith('data: '):
                        continue
                    item = line[6:]
                    if item == '[DONE]':
                        done = True
                        continue
                    frame = json.loads(item)
                    if frame.get('usage'):
                        usage = frame['usage']
                    if frame.get('timings'):
                        timings = frame['timings']
                    for choice in frame.get('choices',[]):
                        delta = choice.get('delta') or {}
                        if delta.get('content'):
                            now = time.perf_counter()
                            if first is None:first = now
                            last = now
                            content.append(delta['content'])
                        if delta.get('reasoning_content'):reasoning.append(delta['reasoning_content'])
                        if choice.get('finish_reason'):finish = choice['finish_reason']
            ended = time.perf_counter()
            text = ''.join(content)
            assert done and usage and first is not None, 'Missing SSE completion, usage or translated content'
            assert finish == 'stop', 'Incomplete translation: finish_reason='+str(finish)
            han = sum('\u4e00' <= c <= '\u9fff' for c in text)
            assert han > 200, 'Translation suspiciously short'
            row = {'id':identifier,'book':book['id'],'title_zh_TW':book['title_zh_TW'],'replica':replica,
                   'source_words':book['word_count'],'source_sha256':book['excerpt_sha256'],
                   'seconds':round(ended-started,4),'ttft_content_seconds':round(first-started,4),
                   'start_offset_seconds':round(started-clock['start'],4),
                   'end_offset_seconds':round(ended-clock['start'],4),
                   'usage':usage,'engine_timings':timings,'finish_reason':finish,
                   'translation_chars':len(text),'han_chars':han,
                   'visible_decode_tokens_per_second':round(usage['completion_tokens']/max(last-first,.001),3),
                   'reasoning_chars':len(''.join(reasoning)),
                   'simplified_character_hits':{c:text.count(c) for c in sorted(SIMPLIFIED) if c in text},
                   'historical_currency_warning':book['id']=='pride_and_prejudice' and any(x in text for x in ['鎂','美元','美金']),
                   'passed':True}
            (a.out/(identifier+'.zh-TW.txt')).write_text(text+'\n')
        except Exception as exc:
            row = {'id':identifier,'book':book['id'],'replica':replica,'passed':False,
                   'seconds':round(time.perf_counter()-started,4),'error':type(exc).__name__+': '+str(exc)}
        (a.out/(identifier+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        return row
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(translate,*j) for j in jobs]
            barrier.wait()
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                records.append(row)
                print(json.dumps(row,ensure_ascii=False),flush=True)
        wall = time.perf_counter()-clock['start']
    finally:
        stop.set()
        monitor_thread.join(timeout=15)
    good = [r for r in records if r['passed']]
    summary = {'label':a.label,'model':cfg['model_name'],'configured_slots':cfg['parallel'],
        'effective_slots':before['concurrency']['serving'],'max_context':before['cache_max_tokens'],
        'simultaneous_requests_sent':len(jobs),'replicas_per_book':a.replicas,
        'max_observed_active_slots':max((s['active_slots'] for s in samples),default=0),
        'wall_seconds':round(wall,4),'successful':len(good),'failed':len(records)-len(good),
        'completion_tokens':sum(r['usage']['completion_tokens'] for r in good),
        'aggregate_completion_tokens_per_second':round(sum(r['usage']['completion_tokens'] for r in good)/wall,3),
        'aggregate_han_chars_per_second':round(sum(r['han_chars'] for r in good)/wall,3),
        'ttft_content_median_seconds':round(statistics.median(r['ttft_content_seconds'] for r in good),3) if good else None,
        'ttft_content_max_seconds':max((r['ttft_content_seconds'] for r in good),default=None),
        'system_prompt':SYSTEM,'system_prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'generation_parameters':{'max_tokens':3072,'temperature':.2,'top_p':.9,'seed':20261109,'enable_thinking':False},
        'records':sorted(records,key=lambda x:x['id']), 'telemetry':samples,'telemetry_errors':errors,
        'limits':'One simultaneous burst, not a statistical comparison. Aggregate throughput includes prompt admission, queueing and generation; per-request TTFT is time to first translated content, not SSE headers. Input cache reuse and actual token counts are recorded.'}
    (a.out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('SUMMARY '+json.dumps({k:v for k,v in summary.items() if k not in ['records','telemetry','system_prompt']},ensure_ascii=False),flush=True)
    return int(len(good)!=len(jobs))

if __name__ == '__main__':
    raise SystemExit(main())
