#!/usr/bin/env python3
"""onecat API gateway: clients -> 127.0.0.1:8000 (1Cat-vLLM)
- 并发限流: ONECAT_MAX_CONCURRENT (默认3) + 队列 ONECAT_MAX_QUEUE(10) + 排队超时 120s -> 429
- 认证: Authorization: Bearer <key>; Anthropic /v1/messages 也支持 x-api-key
- Anthropic Messages(v4.4): 原生请求体/协议头/状态码透传, JSON/SSE 共用并发限制
- SSE 流式透传; 完成/异常/客户端断开均释放并发名额
- 输出预算: 默认4096, 自动压到上限131072 (兼容 max_tokens/max_output_tokens/max_completion_tokens)
- 上下文预算自适应重试: vLLM 报"输入+输出超窗"时, 弹掉输出预算重发, 引擎原生默认
  max_tokens = max_model_len - prompt_tokens 即精确可放下的额度; 若不带预算仍超窗说明
  prompt 本身超限, 透传引擎 400
  (不可解析错误文本取 prompt token 数: 1Cat 引擎该值是合成值 ctx-max_tokens+1, 非真实分词数,
  按它钳制每轮只会缩 margin+1 个 token, 等于没缩)
- 角色规范化: responses 的顶层 instructions + developer/system 项合并为唯一一条开头 system;
  chat 的 developer/system 消息合并置于最前 (解决 "System message must be at the beginning")
- thinking 三档(v4): 默认开启 medium; low/medium/high(max) 映射为模板合法值 low/medium/xhigh;
  none/off/minimal 或显式 enable_thinking:false 关闭; 显式 chat_template_kwargs 优先;
  顶层 reasoning_effort/reasoning.effort/enable_thinking 统一收进 chat_template_kwargs
  (vLLM 会把顶层 reasoning_effort 直接送模板, high 会 400, 必须映射后弹掉)
- 图片透传(v4.3): 引擎为 Qwen3.5-VL 多模态模型; responses 的 input_image 部件(消息 content 内 /
  顶层散装 item / function_call_output 列表)翻译为 chat image_url 多部件 content 直达引擎;
  system 角色内的图丢弃, tool 角色内的图转挂为紧随其后的 user 消息(Qwen 模板 tool 只收文本);
  单请求图片上限 8 张、单图 data URI 上限 30MB, 超限丢弃并日志;
  >=30k 字符的巨型文本疑似 base64 时打形态告警(只记长度与 datauri 标记, 不记内容)
- responses 完整适配(v4.1): /v1/responses 请求翻译为 /v1/chat/completions 发引擎
  (instructions/system/developer 合并、function_call(_output) <-> tool_calls/role:tool、
  reasoning effort 映射)，非流式与流式响应再翻译回标准 Responses 协议
  （引擎原生 responses 对带工具调用长对话不可靠: 校验 400 / 500）
- 缓存可观测性+保温(v4.2): 统计 vLLM prefix-caching 命中 (cached_tokens/prompt_tokens),
  GET /cache/stats 实时指标; 非流式响应带 X-Cached-Tokens/X-Cache-Hit-Ratio 头;
  每分钟 CACHE 聚合日志; 活跃会话 system prompt 每 N 秒后台 max_tokens=1 保温重放
  (直连上游不占用户并发名额), 防止 KV 前缀块被 LRU 逐出
"""
import asyncio, hashlib, json, os, re, time, uuid
import aiohttp
from aiohttp import web
from multidict import CIMultiDict

UPSTREAM = os.environ.get("ONECAT_UPSTREAM", "http://127.0.0.1:8000")
MAX_CONCURRENT = int(os.environ.get("ONECAT_MAX_CONCURRENT", "3"))
MAX_QUEUE = int(os.environ.get("ONECAT_MAX_QUEUE", "10"))
QUEUE_TIMEOUT = float(os.environ.get("ONECAT_QUEUE_TIMEOUT", "120"))
MAX_MODEL_LEN = int(os.environ.get("ONECAT_MAX_MODEL_LEN", "32768"))
API_KEY = os.environ["VLLM_API_KEY"]
sem = asyncio.Semaphore(MAX_CONCURRENT)
waiting = 0

# ---- 缓存可观测性 + system prompt 保温 (v4.2) ----
# vLLM 已开 --enable-prefix-caching, 但网关此前纯透传从不统计 cached_tokens,
# 用户完全看不到命中情况; 且 KV 池紧张(每卡~4GB), 前缀块 LRU 易被逐出,
# 会话间隔一长就全量重算 prefill. 本版:
#   1) 解析每个响应 usage 的 prompt/cached tokens, 累计命中率 (流式/非流式/responses 全覆盖)
#   2) GET /cache/stats 实时指标; 非流式响应加 X-Cached-Tokens / X-Cache-Hit-Ratio 头
#   3) 每分钟一行 CACHE 聚合日志
#   4) 保温: 记录活跃会话 system prompt, 每 N 秒直连上游 max_tokens=1 重放,
#      钉住 KV 前缀块不被逐出 (绕开用户并发限流, 不占 sem 名额)
WARM_ENABLED = os.environ.get("ONECAT_WARM_ENABLED", "1") == "1"
WARM_INTERVAL = float(os.environ.get("ONECAT_WARM_INTERVAL", "30"))
WARM_IDLE = float(os.environ.get("ONECAT_WARM_IDLE", "600"))
WARM_MAX_CHARS = int(os.environ.get("ONECAT_WARM_MAX_CHARS", "131072"))
WARM_MODEL = os.environ.get("ONECAT_WARM_MODEL", "qwen3.8-27b")
CACHE_LOG_INTERVAL = float(os.environ.get("ONECAT_CACHE_LOG_INTERVAL", "60"))

STATS = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0}
WINDOW = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0}
WARM_STATS = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0}
WARM_PROMPTS: dict = {}
_last_window_log = time.monotonic()

# 引擎级 prefix cache 计数: 1Cat-vLLM 逐请求 usage.prompt_tokens_details 恒为 null,
# 真实命中只能从 /metrics 的 vllm:prefix_cache_queries_total / hits_total 差分得到
ENGINE_METRICS_INTERVAL = float(os.environ.get("ONECAT_ENGINE_METRICS_INTERVAL", "30"))
_ENGINE_METRIC_RE = re.compile(
    r'vllm:prefix_cache_(?:queries|hits)_total\{[^}]*\}\s+([\d.eE+]+)')
ENGINE_CACHE = {"queries_total": None, "hits_total": None,
                "session_queries": 0, "session_hits": 0}


def _hit_rate(d):
    return (f"{d['cached_tokens'] / d['prompt_tokens'] * 100:.1f}%"
            if d["prompt_tokens"] else "n/a")


def record_usage(u, warm=False):
    # 累计缓存命中; 兼容 chat(usage.prompt_tokens) 与 responses(usage.input_tokens) 两种格式
    # 返回 (prompt, cached) 供响应头使用
    if not isinstance(u, dict):
        return None
    if u.get("prompt_tokens") is not None:
        pt = int(u.get("prompt_tokens") or 0)
        ct = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    elif u.get("input_tokens") is not None:
        pt = int(u.get("input_tokens") or 0)
        ct = int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    else:
        return None
    tgt = WARM_STATS if warm else STATS
    tgt["requests"] += 1
    tgt["prompt_tokens"] += pt
    tgt["cached_tokens"] += ct
    if not warm:
        WINDOW["requests"] += 1
        WINDOW["prompt_tokens"] += pt
        WINDOW["cached_tokens"] += ct
    return pt, ct


def _system_of_chat(payload):
    for m in payload.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "system":
            t = _texts(m.get("content"))
            if t:
                return t
    return ""


def register_warm(text):
    # 登记活跃会话 system prompt 用于保温; 超过 idle 时限的条目自动淘汰
    if not WARM_ENABLED or not isinstance(text, str):
        return
    text = text.strip()
    if not text or len(text) > WARM_MAX_CHARS:
        return
    key = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]
    now = time.time()
    e = WARM_PROMPTS.get(key)
    if e is None:
        WARM_PROMPTS[key] = {"text": text, "chars": len(text), "seen": now,
                             "warmed": 0.0, "warms": 0, "hits": 0, "misses": 0}
    else:
        e["seen"] = now
    for k in [k for k, v in WARM_PROMPTS.items() if now - v["seen"] > WARM_IDLE]:
        del WARM_PROMPTS[k]


async def warm_loop(app):
    # 后台保温: 直连上游(绕开用户 sem 限流), max_tokens=1, 关闭 thinking
    if not WARM_ENABLED:
        return
    log(f"WARM loop: interval={WARM_INTERVAL}s idle<={WARM_IDLE}s "
        f"max_chars={WARM_MAX_CHARS} model={WARM_MODEL}")
    timeout = aiohttp.ClientTimeout(total=180, sock_connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        while True:
            await asyncio.sleep(5)
            if not WARM_PROMPTS:
                continue
            now = time.time()
            for key, e in list(WARM_PROMPTS.items()):
                if now - e["warmed"] < WARM_INTERVAL:
                    continue
                e["warmed"] = now
                body = {"model": WARM_MODEL, "stream": False, "max_tokens": 1,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "messages": [{"role": "system", "content": e["text"]},
                                     {"role": "user", "content": "."}]}
                try:
                    async with s.post(f"{UPSTREAM}/v1/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {API_KEY}"}) as r:
                        if r.status == 200:
                            u = (await r.json()).get("usage") or {}
                            record_usage(u, warm=True)
                            ct = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                            e["warms"] += 1
                            e["hits" if ct else "misses"] += 1
                            log(f"WARM {key[:8]} pt={u.get('prompt_tokens')} cached={ct} "
                                f"total={e['warms']} hit={_hit_rate(WARM_STATS)}")
                        else:
                            err = (await r.text())[:120]
                            log(f"WARM {key[:8]} http {r.status} {err}")
                except Exception as ex:
                    log(f"WARM {key[:8]} error {type(ex).__name__}: {ex}")


async def cache_log_loop(app):
    global _last_window_log
    while True:
        await asyncio.sleep(10)
        now = time.monotonic()
        if now - _last_window_log < CACHE_LOG_INTERVAL:
            continue
        _last_window_log = now
        ec = ENGINE_CACHE
        eh = (f"{ec['session_hits'] / ec['session_queries'] * 100:.1f}%"
              if ec["session_queries"] else "n/a")
        log(f"CACHE win={int(CACHE_LOG_INTERVAL)}s req={WINDOW['requests']} "
            f"prompt={WINDOW['prompt_tokens']} cached={WINDOW['cached_tokens']} "
            f"win_hit={_hit_rate(WINDOW)} total_hit={_hit_rate(STATS)} "
            f"engine_hit={eh}")
        WINDOW.update(requests=0, prompt_tokens=0, cached_tokens=0)


async def engine_metrics_loop(app):
    """周期拉取 vLLM /metrics, 差分 prefix cache 计数器 (该版 vLLM 逐请求不报 cached_tokens)"""
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        while True:
            await asyncio.sleep(ENGINE_METRICS_INTERVAL)
            try:
                async with s.get(f"{UPSTREAM}/metrics") as r:
                    if r.status != 200:
                        continue
                    text = await r.text()
            except Exception:
                continue
            vals = _ENGINE_METRIC_RE.findall(text)
            if not vals:
                continue
            try:
                qs = float(vals[0])
                hs = float(vals[1])
            except (ValueError, IndexError):
                continue
            ec = ENGINE_CACHE
            prev_q, prev_h = ec["queries_total"], ec["hits_total"]
            if prev_q is not None and qs >= prev_q and hs >= prev_h:
                ec["session_queries"] += int(qs - prev_q)
                ec["session_hits"] += int(hs - prev_h)
            ec["queries_total"], ec["hits_total"] = qs, hs


async def cache_stats(request):
    return web.json_response({
        "requests": STATS["requests"],
        "prompt_tokens": STATS["prompt_tokens"],
        "cached_tokens": STATS["cached_tokens"],
        "hit_rate": _hit_rate(STATS),
        "window": {**WINDOW, "seconds": int(CACHE_LOG_INTERVAL)},
        "engine": {**ENGINE_CACHE,
                    "hit_rate": (f"{ENGINE_CACHE['session_hits'] / ENGINE_CACHE['session_queries'] * 100:.1f}%"
                                 if ENGINE_CACHE["session_queries"] else "n/a")},
        "warm": {"enabled": WARM_ENABLED, "interval": WARM_INTERVAL,
                 "idle_timeout": WARM_IDLE, "max_chars": WARM_MAX_CHARS,
                 "model": WARM_MODEL, "stats": WARM_STATS,
                 "prompts": [{"key": k, "chars": v["chars"], "warms": v["warms"],
                              "hits": v["hits"], "misses": v["misses"],
                              "seen_ago_s": round(time.time() - v["seen"])}
                             for k, v in WARM_PROMPTS.items()]}})


async def _start_bg(app):
    app["bg_tasks"] = [asyncio.create_task(warm_loop(app)),
                       asyncio.create_task(cache_log_loop(app)),
                       asyncio.create_task(engine_metrics_loop(app))]


async def _stop_bg(app):
    for t in app.get("bg_tasks", []):
        t.cancel()


PATHS = {"/v1/chat/completions", "/v1/responses", "/v1/completions", "/v1/messages"}
BUDGET_KEYS = ("max_tokens", "max_output_tokens", "max_completion_tokens")
BUDGET_CAP = 131072  # xhigh 满配实测一轮 ~52k token(思考~35%), 64k 余量偏紧, 放到窗口内的 128k
BUDGET_DEFAULT = 4096
RESPONSES_BUDGET_DEFAULT = 65536  # Codex 等客户端不带 max_output_tokens; 32768 会让 xhigh 思考+正文撞墙切掉正文
MAX_IMAGES_PER_REQ = 8            # 单请求图片上限, 超出丢弃并日志
IMG_DATAURI_MAX = 30 * 1024 * 1024  # 单图 data URI 上限(字节), 防 base64 巨物撞爆请求


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _texts(content):
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n\n".join(parts)
    if isinstance(content, str):
        return content
    return ""


def merge_system_roles_responses(payload):
    """顶层 instructions + input 里的 developer/system 项合并为唯一一条开头 system。
    vLLM 会把 instructions 转成 system; 若 input 里再有 system/developer, 第二条 system 不在开头即报错。"""
    items = payload.get("input")
    if not isinstance(items, list):
        return
    instr = payload.get("instructions")
    instr = instr.strip() if isinstance(instr, str) else ""
    sys_texts, rest = [], []
    for it in items:
        if isinstance(it, dict) and it.get("role") in ("developer", "system"):
            t = _texts(it.get("content"))
            if t:
                sys_texts.append(t)
        else:
            rest.append(it)
    if sys_texts:
        combined = ([instr] if instr else []) + sys_texts
        payload.pop("instructions", None)
        payload["input"] = [{"role": "system", "type": "message",
                             "content": [{"type": "input_text", "text": "\n\n".join(combined)}]}] + rest


def merge_system_roles_chat(payload):
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return
    sys_texts, rest = [], []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") in ("developer", "system"):
            t = _texts(m.get("content"))
            if t:
                sys_texts.append(t)
        else:
            rest.append(m)
    if sys_texts:
        payload["messages"] = [{"role": "system", "content": "\n\n".join(sys_texts)}] + rest


def normalize_budget(payload, responses_style, default=None):
    val, raw = None, None
    for k in BUDGET_KEYS:
        if payload.get(k) is not None:
            raw = f"{k}={payload[k]}"
            try:
                val = int(payload[k])
            except (TypeError, ValueError):
                val = None
            break
    for k in BUDGET_KEYS:
        payload.pop(k, None)
    if val is None:
        val = default if default is not None else BUDGET_DEFAULT
    val = min(max(val, 1), BUDGET_CAP)
    log(f"budget: requested({raw or 'none'}) -> {val}")
    if responses_style:
        payload["max_output_tokens"] = val
    else:
        payload["max_tokens"] = val


def is_ctx_error(err_body):
    return b"maximum context length" in (err_body or b"")


# ---- thinking 三档适配 ----
# qwen3.8 模板硬校验 reasoning_effort 只认 xhigh/medium/low（传 high 直接 400），
# 因此网关统一把外部档位名映射成模板合法值，并收进 chat_template_kwargs：
#   low -> low   medium -> medium   high/max -> xhigh   none/off/minimal -> 关闭
# 客户端不传任何档位时默认开启 thinking(medium)。
# xhigh 实测(2026-09-09 65536 满预算实验): 会收敛(finish=stop), 但思考+正文总量
# ~52k token, 思考占~35%; 预算必须 >= 65536, 32768 会切掉正文(finish=length)。
EFFORT_MAP = {"low": "low", "medium": "medium", "high": "xhigh", "max": "xhigh", "xhigh": "xhigh"}
EFFORT_OFF = {"none", "off", "disabled", "disable", "false", "minimal"}
THINKING_DEFAULT_EFFORT = "medium"


def normalize_thinking(payload):
    """归一化思考参数；返回日志用的档位描述。显式 chat_template_kwargs 优先级最高。"""
    ctk = dict(payload["chat_template_kwargs"]) if isinstance(payload.get("chat_template_kwargs"), dict) else {}

    eff = None
    for raw in (payload.get("reasoning_effort"),
                payload.get("reasoning", {}).get("effort") if isinstance(payload.get("reasoning"), dict) else None):
        if isinstance(raw, str):
            r = raw.strip().lower()
            eff = "off" if r in EFFORT_OFF else EFFORT_MAP.get(r)
            if eff:
                break
    if eff is None and isinstance(payload.get("enable_thinking"), bool):
        eff = "medium" if payload["enable_thinking"] else "off"

    # 顶层字段统一弹掉，避免与 chat_template_kwargs 双源冲突（vLLM 会把顶层值送进模板）
    for k in ("reasoning_effort", "enable_thinking"):
        payload.pop(k, None)
    if isinstance(payload.get("reasoning"), dict) and "effort" in payload["reasoning"]:
        payload["reasoning"].pop("effort", None)

    explicit = "enable_thinking" in ctk or "reasoning_effort" in ctk
    if explicit:
        # 客户端已显式指定：尊重开关，仅在开启且缺 effort 时补档位
        if ctk.get("enable_thinking", True) and "reasoning_effort" not in ctk and eff not in (None, "off"):
            ctk["reasoning_effort"] = eff
        desc = f"explicit(ctk={ctk})"
    elif eff == "off":
        ctk.update({"enable_thinking": False})
        desc = "off"
    elif eff:
        ctk.update({"enable_thinking": True, "reasoning_effort": eff})
        desc = eff
    else:
        ctk.update({"enable_thinking": True, "reasoning_effort": THINKING_DEFAULT_EFFORT})
        desc = f"default({THINKING_DEFAULT_EFFORT})"
    payload["chat_template_kwargs"] = ctk
    return desc


# ============================================================
# /v1/responses -> /v1/chat/completions 完整适配（v4.1）
# 背景: 1Cat-vLLM 的 responses 实现对带工具调用的长对话不可靠
# （ZCode compact 10473 项校验错误 / 重放 500 list index out of range），
# 网关把 responses 请求翻译成 chat 请求发引擎，响应/流式再翻译回标准
# Responses 协议（事件格式按引擎自身输出采样对齐）。
# ============================================================
def _rid(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _img_url_of(part):
    """从 input_image/image_url 部件取 URL (兼容 str 与 {url:...} 两种形态)"""
    u = part.get("image_url")
    if u is None:
        u = part.get("url")
    if isinstance(u, dict):
        u = u.get("url")
    return u if isinstance(u, str) and u else None


def _content_parts(content):
    """content -> (text, images): 兼容纯字符串 / chat 部件 / responses 部件"""
    if isinstance(content, str):
        return content, []
    texts, imgs = [], []
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            if isinstance(p.get("text"), str):
                texts.append(p["text"])
            elif p.get("type") in ("input_image", "image_url") or "image_url" in p:
                u = _img_url_of(p)
                if u:
                    imgs.append(u)
    return "\n\n".join(texts), imgs


def _mk_content(text, imgs):
    """无图保持纯字符串(老行为), 有图构造 chat 多部件 content 供 Qwen3.5-VL 引擎消费"""
    if not imgs:
        return text
    parts = [{"type": "text", "text": text}] if text else []
    parts += [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
    return parts


def _shape_anomaly(payload):
    """异常形态告警(只记长度/标记, 不记内容): >=30k 字符的巨型文本疑似 base64、图片部件计数"""
    notes = []

    def scan(content, where):
        if isinstance(content, str):
            if len(content) >= 30000:
                notes.append(f"{where}:text[{len(content)}] datauri={content[:11] == 'data:image/'}")
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                if isinstance(p.get("text"), str) and len(p["text"]) >= 30000:
                    notes.append(f"{where}:text[{len(p['text'])}] datauri={p['text'][:11] == 'data:image/'}")
                elif p.get("type") in ("input_image", "image_url") or "image_url" in p:
                    u = _img_url_of(p)
                    notes.append(f"{where}:img[{len(u) if u else 0}]")

    items = payload.get("input")
    if isinstance(items, str):
        scan(items, "input")
    else:
        for i, it in enumerate(items or []):
            if not isinstance(it, dict):
                continue
            t = it.get("type") or "message"
            scan(it.get("content"), f"{t}#{i}")
            o = it.get("output")
            if isinstance(o, str):
                if len(o) >= 30000:
                    notes.append(f"out#{i}:text[{len(o)}] datauri={o[:11] == 'data:image/'}")
            else:
                scan(o, f"out#{i}")
    ins = payload.get("instructions")
    if isinstance(ins, str) and len(ins) >= 60000:
        notes.append(f"instructions[{len(ins)}]")
    return "; ".join(notes)


def responses_to_chat_payload(payload):
    """Responses 请求体 -> chat completions 请求体；instructions/system/developer 合并到开头 system
    返回 (chat, img_dropped, img_attached); 无可用 input 时 chat 为 None"""
    msgs, sys_texts = [], []
    n_drop = n_attach = 0
    instr = payload.get("instructions")
    if isinstance(instr, str) and instr.strip():
        sys_texts.append(instr.strip())

    items = payload.get("input")
    if isinstance(items, str):
        msgs.append({"role": "user", "content": items})
    elif isinstance(items, list):
        pending_imgs = []

        # 引擎 --limit-mm-per-prompt 按"整条请求"校验, 而历史图片随轮次累积,
        # 必须按全请求计数: 超出上限时丢最老、保最新, 保证永不触发引擎硬 400
        scan = []
        for it in items:
            if not isinstance(it, dict):
                continue
            t = it.get("type")
            if t in (None, "message"):
                scan.append(_content_parts(it.get("content"))[1])
            elif t == "input_image":
                u = _img_url_of(it)
                if u:
                    scan.append([u])
            elif t == "function_call_output":
                out = it.get("output")
                scan.append(_content_parts(out)[1] if isinstance(out, list) else [])
        quota_drop = max(0, sum(len(x) for x in scan) - MAX_IMAGES_PER_REQ)

        def _take(imgs):
            nonlocal quota_drop, n_drop
            out = list(imgs)
            while quota_drop > 0 and out:  # 超额: 从最老的开始丢, 保当轮最新图
                out.pop(0)
                quota_drop -= 1
                n_drop += 1
            keep = [u for u in out if len(u) <= IMG_DATAURI_MAX]
            n_drop += len(out) - len(keep)
            return keep

        for it in items:
            if not isinstance(it, dict):
                continue
            t = it.get("type")
            if t in (None, "message"):
                role = it.get("role") or "user"
                text, imgs = _content_parts(it.get("content"))
                if role in ("system", "developer"):
                    n_drop += len(imgs)  # 模板 system 角色不支持图片
                    if text:
                        sys_texts.append(text)
                elif role == "tool":
                    msgs.append({"role": "tool", "tool_call_id": it.get("tool_call_id") or "",
                                 "content": text})
                else:
                    imgs = _take(pending_imgs + imgs)
                    pending_imgs = []
                    n_attach += len(imgs)
                    msgs.append({"role": role, "content": _mk_content(text, imgs)})
            elif t == "input_image":
                u = _img_url_of(it)
                if u:
                    pending_imgs.append(u)
            elif t == "function_call":
                if not (msgs and msgs[-1].get("role") == "assistant"):
                    msgs.append({"role": "assistant", "content": ""})
                msgs[-1].setdefault("tool_calls", []).append({
                    "id": it.get("call_id") or it.get("id") or _rid("call"),
                    "type": "function",
                    "function": {"name": it.get("name") or "",
                                 "arguments": it.get("arguments") or "{}"}})
            elif t == "function_call_output":
                out = it.get("output")
                if isinstance(out, str):
                    text = out
                    imgs = []
                elif isinstance(out, list):
                    text, imgs = _content_parts(out)
                else:
                    text, imgs = (json.dumps(out, ensure_ascii=False) if out is not None else ""), []
                msgs.append({"role": "tool", "tool_call_id": it.get("call_id") or "",
                             "content": text})
                imgs = _take(pending_imgs + imgs)
                pending_imgs = []
                if imgs:
                    # Qwen 模板 tool 角色只收文本: 图转成紧随其后的 user 消息
                    n_attach += len(imgs)
                    msgs.append({"role": "user",
                                 "content": _mk_content("(上一条工具输出附带的图片)", imgs)})
            # reasoning 等其他类型 item：思考内容不回放，跳过
        if pending_imgs:
            n_attach += len(pending_imgs)
            msgs.append({"role": "user", "content": _mk_content("", _take(pending_imgs))})
        if n_attach:
            log(f"images attached: {n_attach}")
        if n_drop:
            log(f"images dropped: {n_drop} (cap {MAX_IMAGES_PER_REQ}/req or oversize)")
    if sys_texts:
        msgs.insert(0, {"role": "system", "content": "\n\n".join(sys_texts)})
    if not msgs:
        return None, 0, 0

    chat = {"model": payload.get("model"), "messages": msgs,
            "stream": bool(payload.get("stream"))}
    for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty",
              "stop", "seed", "user", "parallel_tool_calls", "tool_choice",
              "max_tokens", "max_output_tokens", "max_completion_tokens"):
        if payload.get(k) is not None:
            chat[k] = payload[k]
    if payload.get("chat_template_kwargs"):
        chat["chat_template_kwargs"] = payload["chat_template_kwargs"]
    # 思考档位透传: responses 的 reasoning.effort / reasoning_effort / enable_thinking
    # 不拷进 chat payload 的话 normalize_thinking 看不到, 永远落到 default(medium)
    r = payload.get("reasoning")
    if isinstance(r, dict) and isinstance(r.get("effort"), str):
        chat["reasoning_effort"] = r["effort"]
    if isinstance(payload.get("reasoning_effort"), str):
        chat["reasoning_effort"] = payload["reasoning_effort"]
    if isinstance(payload.get("enable_thinking"), bool):
        chat["enable_thinking"] = payload["enable_thinking"]
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        ct = []
        for tt in tools:
            if isinstance(tt, dict) and tt.get("type") == "function":
                ct.append({"type": "function", "function": {
                    "name": tt.get("name") or "",
                    "description": tt.get("description") or "",
                    "parameters": tt.get("parameters") or {"type": "object", "properties": {}}}})
        if ct:
            chat["tools"] = ct
    if chat["stream"]:
        chat["stream_options"] = {"include_usage": True}
    return chat, n_drop, n_attach


def _resp_skeleton(rid, orig, budget, status="in_progress", output=None, usage=None):
    return {"id": rid, "object": "response", "created_at": int(time.time()),
            "status": status, "incomplete_details": None, "error": None,
            "instructions": orig.get("instructions"), "metadata": {},
            "model": orig.get("model") or "qwen3.8-27b",
            "output": output if output is not None else [],
            "parallel_tool_calls": orig.get("parallel_tool_calls")
            if isinstance(orig.get("parallel_tool_calls"), bool) else True,
            "temperature": orig.get("temperature"),
            "tool_choice": "auto" if orig.get("tools") else "none",
            "tools": orig.get("tools") if isinstance(orig.get("tools"), list) else [],
            "top_p": orig.get("top_p"), "background": False,
            "max_output_tokens": budget, "max_tool_calls": None,
            "previous_response_id": None, "prompt": None, "reasoning": None,
            "service_tier": "auto", "text": None, "top_logprobs": None,
            "truncation": "disabled", "usage": usage, "user": None}


def _usage_of(chat_usage):
    # 所有 token 计数字段必须是非空整数: ZCode 客户端按 i64 解析 response.completed,
    # 引擎缺 usage 或 details 为 null 时透传 null 会导致客户端 "invalid type: null, expected i64" 断连
    u = chat_usage or {}
    cd = u.get("completion_tokens_details") or {}
    pd = u.get("prompt_tokens_details") or {}
    return {"input_tokens": u.get("prompt_tokens") or 0,
            "input_tokens_details": {"cached_tokens": pd.get("cached_tokens") or 0},
            "output_tokens": u.get("completion_tokens") or 0,
            "output_tokens_details": {"reasoning_tokens": cd.get("reasoning_tokens") or 0},
            "total_tokens": u.get("total_tokens") or 0}


def chat_body_to_responses(chat, orig, budget):
    """非流式: chat 响应体 -> responses 响应体"""
    ch0 = (chat.get("choices") or [{}])[0]
    msg = ch0.get("message") or {}
    finish = ch0.get("finish_reason") or "stop"
    items = []
    r = msg.get("reasoning")
    if isinstance(r, str) and r.strip():
        items.append({"id": _rid("rs"), "summary": [], "type": "reasoning",
                      "content": [{"text": r, "type": "reasoning_text"}],
                      "encrypted_content": None, "status": "completed"})
    text = msg.get("content")
    if isinstance(text, str) and text.strip():
        items.append({"id": _rid("msg"), "type": "message", "role": "assistant",
                      "status": "completed",
                      "content": [{"type": "output_text", "text": text, "annotations": []}]})
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        items.append({"id": _rid("fc"), "type": "function_call", "status": "completed",
                      "call_id": tc.get("id") or _rid("call"),
                      "name": fn.get("name") or "",
                      "arguments": fn.get("arguments") or "{}"})
    status, incomplete = "completed", None
    if finish == "length":
        status, incomplete = "incomplete", {"reason": "max_output_tokens"}
    body = _resp_skeleton(_rid("resp"), orig, budget, status, items, _usage_of(chat.get("usage")))
    body["incomplete_details"] = incomplete
    return body


async def _reasoning_tokens_of(session, model, text):
    """思考 token 精确计数: 1Cat fork 的 usage 不带 completion_tokens_details(上游代码未随 fork 发布),
    引擎 /tokenize 端点按真实分词返回 count; 不可用时按字符混合估算兜底 (CJK ~1.6 字/token, 其他 ~3.8 字符/token)"""
    text = (text or "").strip()
    if not text:
        return 0
    if session is not None:
        try:
            async with session.post(f"{UPSTREAM}/tokenize",
                                    json={"model": model or "qwen3.8-27b", "prompt": text},
                                    headers={"Authorization": f"Bearer {API_KEY}"},
                                    timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    return int((await r.json()).get("count") or 0)
        except Exception as e:
            log(f"tokenize unavailable, estimating: {type(e).__name__}")
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return max(1, round(cjk / 1.6 + (len(text) - cjk) / 3.8))


async def stream_chat_as_responses(up, write, orig, budget, http=None):
    """流式: 逐行读上游 chat SSE，翻译成 Responses SSE 事件序列"""
    seq = 0
    rid = _rid("resp")

    async def ev(etype, data):
        nonlocal seq
        data["type"] = etype
        data["sequence_number"] = seq
        seq += 1
        await write(f"event: {etype}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

    await ev("response.created", {"response": _resp_skeleton(rid, orig, budget)})
    await ev("response.in_progress", {"response": _resp_skeleton(rid, orig, budget)})

    out_idx = 0
    r_open = None      # {"id"}
    r_text = ""
    m_open = None
    m_text = ""
    tools = {}         # chat tool index -> {"item_id","call_id","name","args"}
    done_items = []
    finish, usage = None, None

    async def _close_reasoning():
        nonlocal r_open, out_idx
        if not r_open:
            return
        await ev("response.reasoning_text.done",
                 {"item_id": r_open, "output_index": out_idx - 1, "content_index": 0, "text": r_text})
        await ev("response.reasoning_part.done",
                 {"item_id": r_open, "output_index": out_idx - 1, "content_index": 0,
                  "part": {"text": r_text, "type": "reasoning_text"}})
        item = {"id": r_open, "summary": [], "type": "reasoning",
                "content": [{"text": r_text, "type": "reasoning_text"}],
                "encrypted_content": None, "status": "completed"}
        await ev("response.output_item.done", {"item": item, "output_index": out_idx - 1})
        done_items.append((out_idx - 1, item))
        r_open = None

    async def _close_message():
        nonlocal m_open, out_idx
        if not m_open:
            return
        await ev("response.output_text.done",
                 {"item_id": m_open, "output_index": out_idx - 1, "content_index": 0, "text": m_text})
        await ev("response.content_part.done",
                 {"item_id": m_open, "output_index": out_idx - 1, "content_index": 0,
                  "part": {"type": "output_text", "text": m_text, "annotations": []}})
        item = {"id": m_open, "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": m_text, "annotations": []}]}
        await ev("response.output_item.done", {"item": item, "output_index": out_idx - 1})
        done_items.append((out_idx - 1, item))
        m_open = None

    buf = b""
    async for raw in up.content.iter_any():
        buf += raw
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data_str = line[5:].strip()
            if data_str == b"[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except Exception:
                continue
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            finish = ch.get("finish_reason") or finish
            delta = ch.get("delta") or {}
            rd = delta.get("reasoning")
            if isinstance(rd, str) and rd:
                if not r_open:
                    r_open = _rid("rs")
                    await ev("response.output_item.added",
                             {"item": {"id": r_open, "summary": [], "type": "reasoning",
                                       "content": None, "encrypted_content": None,
                                       "status": "in_progress"}, "output_index": out_idx})
                    out_idx += 1
                    await ev("response.reasoning_part.added",
                             {"content_index": 0, "item_id": r_open, "output_index": out_idx - 1,
                              "part": {"text": "", "type": "reasoning_text"}})
                r_text += rd
                await ev("response.reasoning_text.delta",
                         {"content_index": 0, "delta": rd, "item_id": r_open,
                          "output_index": out_idx - 1})
            cd = delta.get("content")
            if isinstance(cd, str) and cd:
                if r_open:
                    await _close_reasoning()
                if not m_open:
                    m_open = _rid("msg")
                    await ev("response.output_item.added",
                             {"item": {"id": m_open, "type": "message", "role": "assistant",
                                       "status": "in_progress", "content": []},
                              "output_index": out_idx})
                    out_idx += 1
                    await ev("response.content_part.added",
                             {"item_id": m_open, "output_index": out_idx - 1, "content_index": 0,
                              "part": {"type": "output_text", "text": "", "annotations": []}})
                m_text += cd
                await ev("response.output_text.delta",
                         {"item_id": m_open, "output_index": out_idx - 1,
                          "content_index": 0, "delta": cd})
            for tc in (delta.get("tool_calls") or []):
                i = tc.get("index", 0)
                st = tools.get(i)
                fn = tc.get("function") or {}
                if st is None:
                    if r_open:
                        await _close_reasoning()
                    if m_open:
                        # 项生命周期不得交错: 先完整关闭 message 项, 再开 function_call 项
                        await _close_message()
                    st = tools[i] = {"item_id": _rid("fc"),
                                     "call_id": tc.get("id") or _rid("call"),
                                     "name": fn.get("name") or "", "args": ""}
                    await ev("response.output_item.added",
                             {"item": {"arguments": "", "call_id": st["call_id"],
                                       "name": st["name"], "type": "function_call",
                                       "id": st["item_id"], "status": "in_progress"},
                              "output_index": out_idx})
                    st["oi"] = out_idx
                    out_idx += 1
                if fn.get("name") and not st["name"]:
                    st["name"] = fn["name"]
                ad = fn.get("arguments")
                if isinstance(ad, str) and ad:
                    st["args"] += ad
                    await ev("response.function_call_arguments.delta",
                             {"delta": ad, "item_id": st["item_id"], "output_index": st["oi"]})

    await _close_reasoning()
    await _close_message()
    for st in tools.values():
        await ev("response.function_call_arguments.done",
                 {"arguments": st["args"], "item_id": st["item_id"],
                  "name": st["name"], "output_index": st["oi"]})
        item = {"arguments": st["args"], "call_id": st["call_id"], "name": st["name"],
                "type": "function_call", "id": st["item_id"], "status": "completed"}
        await ev("response.output_item.done", {"item": item, "output_index": st["oi"]})
        done_items.append((st["oi"], item))

    done_items.sort(key=lambda x: x[0])
    final_items = [it for _, it in done_items]
    if r_text.strip():
        rt = await _reasoning_tokens_of(http, orig.get("model"), r_text)
        if rt:
            usage = usage if isinstance(usage, dict) else {}
            usage.setdefault("completion_tokens_details", {})["reasoning_tokens"] = rt
    if finish == "length":
        sk = _resp_skeleton(rid, orig, budget, "incomplete", final_items, _usage_of(usage))
        sk["incomplete_details"] = {"reason": "max_output_tokens"}
        await ev("response.incomplete", {"response": sk})
    else:
        await ev("response.completed",
                 {"response": _resp_skeleton(rid, orig, budget, "completed", final_items, _usage_of(usage))})
    record_usage(usage)
    u = _usage_of(usage)
    items_desc = "+".join(it["type"] for it in final_items) or "none"
    log(f"RESPSUM finish={finish} out_tokens={u['output_tokens']} items={items_desc}")


async def health(request):
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(f"{UPSTREAM}/health") as r:
                if r.status == 200:
                    return web.json_response({"gateway": "ok", "upstream": "ready"})
                return web.json_response({"gateway": "ok", "upstream": f"http-{r.status}"}, status=503)
    except Exception as e:
        return web.json_response({"gateway": "ok", "upstream": "unreachable", "detail": type(e).__name__}, status=503)


async def models(request):
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{UPSTREAM}/v1/models",
                         headers={"Authorization": request.headers.get("Authorization", "")}) as r:
            body = await r.read()
            return web.Response(body=body, status=r.status, content_type="application/json")


async def acquire_slot():
    """One queue/semaphore shared by OpenAI and Anthropic generation."""
    global waiting
    if waiting >= MAX_QUEUE:
        return web.json_response({"error": {"message": f"server busy: queue full ({MAX_QUEUE} waiting)", "type": "rate_limit_error"}}, status=429)
    waiting += 1
    t0 = time.monotonic()
    try:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=QUEUE_TIMEOUT)
        except asyncio.TimeoutError:
            return web.json_response({"error": {"message": f"server busy: queued for over {int(QUEUE_TIMEOUT)}s", "type": "rate_limit_error"}}, status=429)
    finally:
        waiting -= 1
    log(f"slot acquired after {time.monotonic()-t0:.1f}s queue_wait")
    return None


def proxy_headers(headers, *, request=False):
    """Preserve end-to-end headers, excluding hop-by-hop framing."""
    excluded = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade", "content-length"}
    for value in headers.getall("Connection", []):
        excluded.update(part.strip().lower() for part in value.split(","))
    if request:
        # aiohttp has already decoded any request content encoding.
        excluded.update({"host", "content-encoding"})
    return CIMultiDict((k, v) for k, v in headers.items() if k.lower() not in excluded)


async def messages_proxy(request):
    # No OpenAI budget/thinking/role transforms: Anthropic payload is native.
    body = await request.read()
    rejected = await acquire_slot()
    if rejected is not None:
        return rejected
    response = None
    try:
        headers = proxy_headers(request.headers, request=True)
        headers["Authorization"] = f"Bearer {API_KEY}"
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as session:
            async with session.post(f"{UPSTREAM}{request.rel_url}", data=body,
                                    headers=headers, allow_redirects=False) as upstream:
                out_headers = proxy_headers(upstream.headers)
                if upstream.content_type == "text/event-stream":
                    out_headers.setdefault("Cache-Control", "no-cache")
                    out_headers["X-Accel-Buffering"] = "no"
                response = web.StreamResponse(status=upstream.status, headers=out_headers)
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
    except (asyncio.CancelledError, ConnectionResetError):
        raise
    except Exception as exc:
        log(f"messages proxy error: {type(exc).__name__}")
        if response is not None and response.prepared:
            # Status was sent: terminate the partial stream, never fabricate JSON.
            if request.transport is not None:
                request.transport.close()
            raise
        return web.json_response({"type": "error", "error": {
            "type": "api_error", "message": f"gateway upstream failure: {type(exc).__name__}"}}, status=502)
    finally:
        sem.release()
        log("slot released")


async def gen_proxy(request):
    authorized = request.headers.get("Authorization") == f"Bearer {API_KEY}"
    if request.path == "/v1/messages":
        authorized = authorized or request.headers.get("x-api-key") == API_KEY
    if not authorized:
        return web.json_response({"error": {"message": "invalid or missing API key", "type": "auth_error"}}, status=401)
    if request.path not in PATHS:
        return web.json_response({"error": {"message": f"path {request.path} not exposed by gateway"}}, status=404)
    if request.path == "/v1/messages":
        return await messages_proxy(request)
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON body"}}, status=400)

    responses_origin = request.path == "/v1/responses"
    img_drop = 0
    if responses_origin:
        # v4.1: responses 完整翻译为 chat 发引擎（引擎 responses 实现对工具调用长对话不可靠）
        orig_payload = payload
        anomaly = _shape_anomaly(payload)
        if anomaly:
            log(f"payload anomaly: {anomaly}")
        payload, img_drop, img_attach = responses_to_chat_payload(payload)
        if payload is None:
            return web.json_response({"error": {"message": "no usable input items",
                                                 "type": "invalid_request_error"}}, status=400)
    elif request.path == "/v1/chat/completions":
        merge_system_roles_chat(payload)
    is_stream = bool(payload.get("stream"))

    if not is_stream and responses_origin:
        payload["stream"] = False
    normalize_budget(payload, responses_style=False,
                     default=RESPONSES_BUDGET_DEFAULT if responses_origin else None)
    if request.path != "/v1/completions":  # legacy completions 不走 chat 模板，无需注入
        thinking = normalize_thinking(payload)
        log(f"thinking: {thinking}")

    # v4.2: 登记保温目标 (chat 取开头 system 消息; responses 取 instructions)
    if request.path == "/v1/chat/completions":
        register_warm(_system_of_chat(payload))
    elif responses_origin:
        ins = orig_payload.get("instructions")
        register_warm(ins if isinstance(ins, str) else "")

    rejected = await acquire_slot()
    if rejected is not None:
        return rejected
    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"{UPSTREAM}/v1/chat/completions" if responses_origin else f"{UPSTREAM}{request.path}"
            hdrs = {"Authorization": f"Bearer {API_KEY}"}
            up = await session.post(url, json=payload, headers=hdrs)
            bkey = "max_tokens"
            orig_budget = int(payload.get(bkey, BUDGET_DEFAULT))
            for attempt in range(4):
                if up.status != 400:
                    break
                err_body = await up.read()
                up.release()
                if not is_ctx_error(err_body):
                    shape = [((it.get("role") or it.get("type")) if isinstance(it, dict) else type(it).__name__)
                             for it in (payload.get("input") or payload.get("messages") or [])][:20]
                    log(f"upstream error 400 body={bytes(err_body[:180])!r} shape={shape}")
                    return web.Response(body=err_body, status=400, content_type="application/json")
                cur = payload.pop(bkey, None)
                if cur is None:
                    log("ctx retry give up: no output budget left to drop (prompt alone over window)")
                    try:
                        ej = json.loads(err_body)
                        em = ej.get("error")
                        if isinstance(em, dict) and isinstance(em.get("message"), str):
                            em["message"] = (
                                "会话输入已超过模型上下文窗口 262144 tokens, 无法生成任何输出。"
                                "请新建会话或精简历史后重试。 [engine: " + em["message"] + "]")
                            err_body = json.dumps(ej, ensure_ascii=False).encode()
                    except Exception:
                        pass
                    return web.Response(body=err_body, status=400, content_type="application/json")
                log(f"ctx retry {attempt+1}: drop {bkey}={cur} -> engine-native clamp (max_model_len - prompt_tokens)")
                try:
                    up = await session.post(url, json=payload, headers=hdrs)
                except aiohttp.ServerDisconnectedError:
                    up = await session.post(url, json=payload, headers=hdrs)
            if up.status != 200:
                body = await up.read()
                log(f"upstream error {up.status} body={bytes(body[:180])!r}")
                return web.Response(body=body, status=up.status, content_type="application/json")
            if not is_stream:
                data = await up.read()
                rec = None
                if responses_origin:
                    try:
                        cj = json.loads(data)
                        r_txt = ((cj.get("choices") or [{}])[0].get("message") or {}).get("reasoning") or ""
                        if isinstance(r_txt, str) and r_txt.strip():
                            rt = await _reasoning_tokens_of(session, cj.get("model"), r_txt)
                            if rt:
                                cj.setdefault("usage", {}).setdefault(
                                    "completion_tokens_details", {})["reasoning_tokens"] = rt
                        out = chat_body_to_responses(cj, orig_payload, orig_budget)
                        data = json.dumps(out, ensure_ascii=False).encode()
                        rec = record_usage(out.get("usage"))
                    except Exception as e:
                        log(f"responses convert error: {type(e).__name__}: {e}")
                else:
                    try:
                        rec = record_usage(json.loads(data).get("usage"))
                    except Exception:
                        pass
                hdrs2 = {}
                if img_drop:
                    hdrs2["X-Images-Dropped"] = str(img_drop)
                if rec:
                    pt, ct = rec
                    hdrs2["X-Cached-Tokens"] = str(ct)
                    hdrs2["X-Cache-Hit-Ratio"] = f"{ct / pt:.3f}" if pt else "0"
                return web.Response(body=data, status=200, content_type="application/json",
                                    headers=hdrs2)
            sheaders = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            if img_drop:
                sheaders["X-Images-Dropped"] = str(img_drop)
            resp = web.StreamResponse(status=200, headers=sheaders)
            await resp.prepare(request)
            try:
                if responses_origin:
                    async def write(s):
                        await resp.write(s.encode())
                    await stream_chat_as_responses(up, write, orig_payload, orig_budget, session)
                else:
                    # v4.2: 流式透传同时旁路解析 usage 行, 统计 prefix cache 命中
                    ubuf = b""
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
                        ubuf += chunk
                        if b'"usage"' in ubuf:
                            while b"\n" in ubuf:
                                line, ubuf = ubuf.split(b"\n", 1)
                                if b'"usage"' in line and line.lstrip().startswith(b"data:"):
                                    try:
                                        record_usage(json.loads(line[5:].strip()).get("usage"))
                                    except Exception:
                                        pass
                        if len(ubuf) > 256 * 1024:
                            ubuf = ubuf[-16384:]
            except (asyncio.CancelledError, ConnectionResetError):
                log("client disconnected; cancelling upstream request")
            finally:
                try:
                    up.close()
                except Exception:
                    pass
                await resp.write_eof()
            return resp
    except (asyncio.CancelledError, ConnectionResetError):
        log("client disconnected before/during streaming")
        raise
    except Exception as e:
        log(f"proxy error: {type(e).__name__}: {e}")
        return web.json_response({"error": {"message": f"gateway upstream failure: {type(e).__name__}"}}, status=502)
    finally:
        sem.release()
        log("slot released")


@web.middleware
async def access_log_mw(request, handler):
    t0 = time.monotonic()
    try:
        resp = await handler(request)
    except asyncio.CancelledError:
        log(f"ACCESS {request.method} {request.path} CANCELLED {time.monotonic()-t0:.1f}s")
        raise
    except Exception as e:
        log(f"ACCESS {request.method} {request.path} EXC {type(e).__name__} {time.monotonic()-t0:.1f}s")
        raise
    log(f"ACCESS {request.method} {request.path} -> {resp.status} {time.monotonic()-t0:.1f}s")
    return resp


def main():
    app = web.Application(middlewares=[access_log_mw], client_max_size=64 * 1024 * 1024)
    app.on_startup.append(_start_bg)
    app.on_cleanup.append(_stop_bg)
    app.router.add_get("/health", health)
    app.router.add_get("/cache/stats", cache_stats)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", gen_proxy)
    app.router.add_post("/v1/completions", gen_proxy)
    app.router.add_post("/v1/responses", gen_proxy)
    app.router.add_post("/v1/messages", gen_proxy)
    web.run_app(app, host=os.environ.get("ONECAT_HOST", "127.0.0.1"),
                port=int(os.environ.get("ONECAT_PORT", "8080")), print=None,
                handler_cancellation=True)


if __name__ == "__main__":
    main()
