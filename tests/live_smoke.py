#!/usr/bin/env python3
"""Opt-in live GPU smoke/benchmark. VLLM_API_KEY must be supplied via environment.

Run: python tests/live_smoke.py --output /path/to/report.json --concurrency 6
Requires aiohttp. Never logs credentials. Sends benign generation requests.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import time

import aiohttp

MODEL = "qwen3.8-27b"


async def main(args):
    key = os.environ["VLLM_API_KEY"]
    bearer = {"Authorization": f"Bearer {key}"}
    anthropic = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    results = []
    peaks = [0] * 4
    gpu_samples = 0
    gpu_errors = []
    stop = asyncio.Event()

    async def gpu_sampler():
        nonlocal gpu_samples
        while not stop.is_set():
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi", "-i", "4,5,6,7", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            output, error = await process.communicate()
            try:
                values = [int(v) for v in output.decode().splitlines()]
                if process.returncode != 0 or len(values) != 4 or min(values) <= 0:
                    raise ValueError("missing or invalid GPU measurements")
                for i, value in enumerate(values):
                    peaks[i] = max(peaks[i], value)
                gpu_samples += 1
            except ValueError:
                gpu_errors.append({"returncode":process.returncode, "error":error.decode(errors="replace")[:300]})
            try:
                await asyncio.wait_for(stop.wait(), 0.5)
            except asyncio.TimeoutError:
                pass

    def record(name, ok, **details):
        entry = {"name": name, "ok": bool(ok), **details}
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as client:
        async def post(path, payload, native=False, stream=False):
            start = time.monotonic()
            async with client.post(args.base + path, json=payload,
                                   headers=anthropic if native else bearer) as response:
                if not stream or response.status != 200:
                    raw = await response.read()
                    try:
                        data = json.loads(raw)
                    except ValueError:
                        data = {"raw": raw.decode(errors="replace")[:500]}
                    return response.status, data, time.monotonic() - start
                events, texts = [], []
                first_text = last_text = None
                first_event = None
                content_type = response.content_type
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[5:].strip())
                    now = time.monotonic()
                    if first_event is None:
                        first_event = now
                    events.append(event)
                    text = (event.get("delta") or {}).get("text", "") if native else (
                        ((event.get("choices") or [{}])[0].get("delta") or {}).get("content") or "")
                    if text:
                        texts.append(text)
                        first_text = first_text or now
                        last_text = now
                usage = next((e["usage"] for e in reversed(events) if e.get("usage")), {})
                return response.status, {
                    "content_type": content_type, "events": len(events),
                    "event_types": sorted({e.get("type", "chat.chunk") for e in events}),
                    "text": "".join(texts), "usage": usage,
                    "first_event_s": first_event - start if first_event else None,
                    "first_text_s": first_text - start if first_text else None,
                    "decode_span_s": last_text - first_text if last_text and first_text else 0,
                }, time.monotonic() - start

        async def metrics():
            async with client.get(args.upstream + "/metrics", headers=bearer) as response:
                response.raise_for_status()
                raw = await response.text()
            out = {}
            for line in raw.splitlines():
                match = re.match(r'([a-zA-Z_:][\w:]*)\{[^}]*\}\s+([\d.eE+-]+)$', line)
                if match:
                    name, value = match.groups()
                    if "spec_decode" in name or name in {
                        "vllm:request_decode_time_seconds_sum", "vllm:request_decode_time_seconds_count",
                        "vllm:generation_tokens_total", "vllm:request_success_total"}:
                        out[name] = out.get(name, 0) + float(value)
            return out

        sample = asyncio.create_task(gpu_sampler())
        try:
            async with client.get(args.base + "/v1/models", headers=bearer) as response:
                model = (await response.json())["data"][0]
                record("model_identity", response.status == 200 and model["id"] == MODEL
                       and "Uncensored-NVFP4-1Cat" in model["root"] and model["max_model_len"] == 262144,
                       model=model)
            for native in (False, True):
                path = "/v1/messages" if native else "/v1/chat/completions"
                for stream in (False, True):
                    payload = {"model": MODEL, "max_tokens": 128, "stream": stream,
                               "messages": [{"role":"user", "content":"用中文简要解释前缀缓存的作用。"}]}
                    if not native:
                        payload["chat_template_kwargs"] = {"enable_thinking": False}
                        if stream:
                            payload["stream_options"] = {"include_usage": True}
                    status, body, elapsed = await post(path, payload, native, stream)
                    if stream:
                        valid = body.get("content_type") == "text/event-stream" and bool(body.get("text")) and body.get("events", 0) > 1
                        if native:
                            valid = valid and "message_stop" in body["event_types"]
                    else:
                        valid = bool(body.get("content")) if native else bool(((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
                    record(path + (" SSE" if stream else " JSON"), status == 200 and valid,
                           status=status, seconds=elapsed, response=body)

            payload = {"model": MODEL, "max_tokens": 128,
                       "messages":[{"role":"user","content":"Call get_temperature for Seoul."}],
                       "tools":[{"name":"get_temperature","description":"Get temperature for a city",
                                 "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
                       "tool_choice":{"type":"tool","name":"get_temperature"}}
            status, body, elapsed = await post("/v1/messages", payload, True)
            record("messages_tool_use", status == 200 and any(x.get("type") == "tool_use" for x in body.get("content", [])), status=status, response=body)

            status, body, elapsed = await post("/v1/responses", {"model": MODEL, "input":"Reply OK only.", "max_output_tokens":64, "reasoning":{"effort":"none"}})
            record("responses_regression", status == 200 and body.get("object") == "response", status=status, response=body)

            status, body, elapsed = await post("/v1/messages", {"model":MODEL, "max_tokens":128,
                "messages":[{"role":"user","content":"写一段虚构成年侦探的愤怒独白，允许使用‘他妈的’这类粗口，不要涉及威胁或伤害。直接输出独白。"}]}, True)
            record("uncensored_checkpoint_benign_content_sample", status == 200 and bool(body.get("content")), status=status, response=body)

            before = await metrics()
            status, body, elapsed = await post("/v1/chat/completions", {
                "model":MODEL, "messages":[{"role":"user","content":"Write a detailed technical explanation of how hash tables work, including collisions, resizing, amortized complexity, and practical examples. Write at least 1500 words."}],
                "max_tokens":1024, "temperature":0, "stream":True,
                "chat_template_kwargs":{"enable_thinking":False}, "stream_options":{"include_usage":True}}, stream=True)
            after = await metrics()
            delta = {k: v-before.get(k,0) for k,v in after.items()}
            speculative_keys = ["vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"]
            record("dflash_active", all(delta.get(k,0) > 0 for k in speculative_keys),
                   counters={k:delta.get(k) for k in speculative_keys})
            tokens = body.get("usage", {}).get("completion_tokens", 0)
            decode_seconds = delta.get("vllm:request_decode_time_seconds_sum", 0)
            decode_count = delta.get("vllm:request_decode_time_seconds_count", 0)
            record("single_request_decode_benchmark", status == 200 and tokens > 100,
                   status=status, output_tokens=tokens, wall_seconds=elapsed,
                   end_to_end_tokens_per_second=tokens/elapsed,
                   engine_decode_tokens_per_second=(tokens-1)/decode_seconds if decode_seconds > 0 and decode_count == 1 else None,
                   engine_decode_seconds=decode_seconds, metrics_delta=delta,
                   streaming={k:v for k,v in body.items() if k != "text"})

            async def load(i):
                status, body, elapsed = await post("/v1/chat/completions", {
                    "model":MODEL, "messages":[{"role":"user","content":f"Test {i}: explain why software tests are useful in detail."}],
                    "max_tokens":256, "temperature":0, "chat_template_kwargs":{"enable_thinking":False}})
                return {"status":status, "seconds":elapsed, "tokens":body.get("usage",{}).get("completion_tokens",0)}
            concurrent = await asyncio.gather(*(load(i) for i in range(args.concurrency)))
            record("concurrent_generation", all(x["status"] == 200 and x["tokens"] > 0 for x in concurrent), concurrency=args.concurrency, requests=concurrent)
            final_metrics = await metrics()
        finally:
            stop.set()
            await sample
    record("gpu_memory", gpu_samples > 0 and not gpu_errors and max(peaks) <= 30.5*1024,
           valid_samples=gpu_samples, errors=gpu_errors, peak_gib=[v/1024 for v in peaks])
    report = {"results":results, "gpu_peak_gib":[v/1024 for v in peaks], "final_metrics":final_metrics}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print("GPU peak GiB", report["gpu_peak_gib"], flush=True)
    if not all(x["ok"] for x in results) or max(peaks) > 30.5*1024:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:2223")
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    asyncio.run(main(parser.parse_args()))
