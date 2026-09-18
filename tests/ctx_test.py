#!/usr/bin/env python3
"""ctx-retry / 超窗钳制 验证套件. VLLM_API_KEY 从环境读取; 不打印密钥.
用法: set -a; . onecat.env; set +a; python3 ctx_test.py"""
import json, os, re, sys, time, urllib.request, urllib.error

BASE = f"http://127.0.0.1:{os.environ.get('ONECAT_PORT', '2223')}"
KEY = os.environ["VLLM_API_KEY"]
MODEL = "qwen3.8-27b"
CTX = 262144
results = []

def call(path, payload, timeout=600, auth=True):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if auth:
        req.add_header("Authorization", f"Bearer {KEY}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

def ctx_prompt_tokens(body: bytes):
    m = re.search(rb"prompt (?:contains at least|has) (\d+) input tokens", body or b"")
    return int(m.group(1)) if m else None

def wordblob(n, start=0):
    return " ".join(f"w{i:06d}" for i in range(start, start + n))

def ok(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)

# 无 key 拒绝
st, b = call("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}, auth=False)
ok("T2 no-key rejected 401", st == 401, f"status={st}")

# 小请求回归
t0 = time.time()
st, b = call("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "只回答:OK"}], "max_tokens": 16})
u = json.loads(b).get("usage", {})
ok("T1 small chat regression", st == 200 and u.get("prompt_tokens", 0) > 0,
   f"status={st} pt={u.get('prompt_tokens')} ct={u.get('completion_tokens')} {time.time()-t0:.1f}s")

# 校准: 递增构造长 prompt, 用引擎真实 usage 计数 (引擎错误文本里的 token 数是合成值, 不可用)
n, P = 24000, None
while True:
    st, b = call("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": wordblob(n)}], "max_tokens": 1})
    if st == 200:
        P = json.loads(b)["usage"]["prompt_tokens"]
        if P > CTX - 123000:
            break
        n += 12000
    else:
        P = ctx_prompt_tokens(b)
        assert P, f"probe failed status={st} body={b[:200]!r}"
        break
print(f"calibration: words={n} prompt_tokens={P}", flush=True)

if P < CTX - 120000:
    print("FATAL: calibration short of window", flush=True); sys.exit(1)

BIG = wordblob(n)

# chat 非流式超窗 -> 弹预算重试 (引擎原生钳制)
t0 = time.time()
st, b = call("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": BIG}], "max_tokens": 131072})
j = json.loads(b) if st == 200 else {}
ok("T3 chat overflow clamp", st == 200 and j.get("usage", {}).get("prompt_tokens", 0) == P,
   f"status={st} pt={j.get('usage', {}).get('prompt_tokens')} fit={CTX - P - 16} {time.time()-t0:.1f}s")

# chat 流式超窗 -> 流开始前完成重试
t0 = time.time()
err = None
try:
    payload = {"model": MODEL, "messages": [{"role": "user", "content": BIG}], "max_tokens": 131072, "stream": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json"); req.add_header("Authorization", f"Bearer {KEY}")
    chunks = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        head = r.read(4096)
        for line in r:
            if line.startswith(b"data:"):
                chunks += 1
except urllib.error.HTTPError as e:
    err = f"HTTP {e.code}: {e.read()[:120]!r}"
    chunks, head = 0, b""
ok("T4 chat stream overflow clamp", err is None and chunks > 1 and head.lstrip().startswith(b"data:"),
   f"sse_data_lines={chunks} {err or f'{time.time()-t0:.1f}s'}")

# /v1/responses 超窗, 应用形态 (max_output_tokens=128000)
t0 = time.time()
err = None
try:
    st, b = call("/v1/responses", {"model": MODEL, "stream": False,
                                   "input": [{"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": BIG}]}],
                                   "max_output_tokens": 128000})
    j = json.loads(b) if st == 200 else {}
except Exception as e:
    err, st, j = str(e), 0, {}
ok("T5 responses overflow clamp", err is None and st == 200 and j.get("status") == "completed",
   f"status={st} resp_status={j.get('status')} pt={j.get('usage', {}).get('input_tokens')} {err or f'{time.time()-t0:.1f}s'}")

# prompt 本身超窗 -> 干净 400 + 中文指引
BIG2 = wordblob(n + 40000)
st, b = call("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": BIG2}], "max_tokens": 131072})
ok("T6 oversized prompt clean 400", st == 400 and b"maximum context length" in b,
   f"status={st} body_head={b[:80]!r}")

fails = [r for r in results if not r[1]]
print(f"\n=== {len(results) - len(fails)}/{len(results)} passed ===", flush=True)
sys.exit(1 if fails else 0)
