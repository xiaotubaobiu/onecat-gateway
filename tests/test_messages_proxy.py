"""Real HTTP contract tests: native Messages must not enter OpenAI transforms."""
import asyncio
import gzip
import importlib.util
import os
from pathlib import Path
import unittest

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

os.environ.setdefault("VLLM_API_KEY", "test-key")
spec = importlib.util.spec_from_file_location("gateway", Path(__file__).parents[1] / "gateway.py")
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class MessagesProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seen = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.active = self.peak = 0
        gateway.API_KEY = "test-key"
        gateway.sem = asyncio.Semaphore(1)
        gateway.waiting = 0
        gateway.MAX_QUEUE = 10
        gateway.QUEUE_TIMEOUT = 0.1
        upstream = web.Application()
        upstream.router.add_post("/v1/messages", self.backend)
        upstream.router.add_post("/v1/chat/completions", self.backend)
        self.upstream = TestServer(upstream, handler_cancellation=True)
        await self.upstream.start_server()
        gateway.UPSTREAM = str(self.upstream.make_url("/")).rstrip("/")
        app = web.Application()
        app.router.add_post("/v1/messages", gateway.gen_proxy)
        app.router.add_post("/v1/chat/completions", gateway.gen_proxy)
        self.server = TestServer(app, handler_cancellation=True)
        await self.server.start_server()
        self.client = ClientSession()

    async def asyncTearDown(self):
        self.release.set()
        await self.client.close()
        await self.server.close()
        await self.upstream.close()

    async def backend(self, request):
        self.seen.append((await request.read(), dict(request.headers), request.query_string))
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.entered.set()
        try:
            mode = request.headers.get("x-test-mode", "json")
            if mode in {"stream", "hold"}:
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "request-id": "req-test", "Cache-Control":"private, no-store"})
                await response.prepare(request)
                await response.write(b'event: message_start\ndata: {"type":"message_start"}\n\n')
                await self.release.wait()
                await response.write(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
                return response
            if mode == "error":
                return web.Response(status=429, body=b'{"type":"error","error":{"type":"rate_limit_error"}}', headers={"Content-Type": "application/json", "Retry-After": "7", "request-id": "req-error"})
            if mode == "redirect":
                return web.Response(status=307, headers={"Location": "/elsewhere"})
            if mode == "gzip":
                return web.Response(body=gzip.compress(b'{"type":"message","content":[]}'), headers={"Content-Type":"application/json", "Content-Encoding":"gzip"})
            return web.Response(body=b'{"type":"message","content":[{"type":"text","text":"OK"}]}', headers={"Content-Type": "application/json", "request-id": "req-test"})
        finally:
            self.active -= 1
            self.closed.set()

    def headers(self, **extra):
        return {"x-api-key": "test-key", "anthropic-version": "2023-06-01", "anthropic-beta": "tools-test", "Content-Type": "application/json", **extra}

    async def test_body_headers_and_query_survive_without_openai_mutation(self):
        raw = b'{ "model":"qwen3.8-27b", "max_tokens":17, "system":[{"type":"text","text":"hello"}], "thinking":{"type":"enabled","budget_tokens":12}, "tools":[], "messages":[] }'
        async with self.client.post(self.server.make_url("/v1/messages?beta=true"), data=raw, headers=self.headers()) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual((await r.json())["type"], "message")
            self.assertEqual(r.headers["request-id"], "req-test")
        body, headers, query = self.seen[0]
        self.assertEqual(body, raw)
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["anthropic-beta"], "tools-test")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(query, "beta=true")

    async def test_bearer_auth_and_invalid_key(self):
        for headers, status in [({"Authorization": "Bearer test-key"}, 200), ({"x-api-key": "wrong"}, 401), ({}, 401)]:
            async with self.client.post(self.server.make_url("/v1/messages"), json={}, headers=headers) as r:
                self.assertEqual(r.status, status)
                await r.read()
        self.assertEqual(len(self.seen), 1)

    async def test_upstream_error_and_retry_header_are_preserved(self):
        async with self.client.post(self.server.make_url("/v1/messages"), json={}, headers=self.headers(**{"x-test-mode": "error"})) as r:
            self.assertEqual(r.status, 429)
            self.assertEqual(r.headers["Retry-After"], "7")
            self.assertEqual((await r.json())["error"]["type"], "rate_limit_error")

    async def test_redirect_is_not_followed(self):
        async with self.client.post(self.server.make_url("/v1/messages"), json={}, headers=self.headers(**{"x-test-mode": "redirect"}), allow_redirects=False) as r:
            self.assertEqual(r.status, 307)
            self.assertEqual(r.headers["Location"], "/elsewhere")

    async def test_sse_first_event_arrives_before_upstream_finishes(self):
        async with self.client.post(self.server.make_url("/v1/messages"), json={"stream": True}, headers=self.headers(**{"x-test-mode": "stream"})) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.content_type, "text/event-stream")
            self.assertEqual(r.headers["X-Accel-Buffering"], "no")
            self.assertEqual(r.headers["Cache-Control"], "private, no-store")
            first = await asyncio.wait_for(r.content.readuntil(b"\n\n"), 1)
            self.assertIn(b"message_start", first)
            self.assertFalse(self.release.is_set())
            self.release.set()
            self.assertIn(b"message_stop", await r.read())

    async def test_messages_and_chat_share_limit_and_timeout_releases_queue(self):
        first = await self.client.post(self.server.make_url("/v1/messages"), json={"stream": True}, headers=self.headers(**{"x-test-mode": "hold"}))
        self.assertEqual(first.status, 200)
        async with self.client.post(self.server.make_url("/v1/chat/completions"), json={"model":"qwen3.8-27b", "messages":[]}, headers={"Authorization":"Bearer test-key"}) as second:
            self.assertEqual(second.status, 429)
        self.assertEqual(gateway.waiting, 0)
        self.assertEqual(self.peak, 1)
        self.release.set()
        await first.read()
        first.close()
        async with self.client.post(self.server.make_url("/v1/messages"), json={}, headers=self.headers()) as r:
            self.assertEqual(r.status, 200)

    async def test_disconnect_closes_upstream_and_releases_slot(self):
        first = await self.client.post(self.server.make_url("/v1/messages"), json={"stream": True}, headers=self.headers(**{"x-test-mode": "hold"}))
        self.assertEqual(first.status, 200)
        await first.content.readuntil(b"\n\n")
        first.close()
        await asyncio.wait_for(self.closed.wait(), 1)
        async with self.client.post(self.server.make_url("/v1/messages"), json={}, headers=self.headers()) as r:
            self.assertEqual(r.status, 200)

    async def test_compressed_body_and_response_remain_decodable(self):
        raw = b'{"model":"qwen3.8-27b","max_tokens":8,"messages":[]}'
        async with self.client.post(self.server.make_url("/v1/messages"), data=gzip.compress(raw), headers=self.headers(**{"Content-Encoding":"gzip", "x-test-mode":"gzip"})) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual((await r.json())["type"], "message")
        self.assertEqual(self.seen[0][0], raw)
        self.assertNotIn("Content-Encoding", self.seen[0][1])

    async def test_queue_full_and_cancelled_waiter_do_not_leak_capacity(self):
        gateway.QUEUE_TIMEOUT = 5
        gateway.MAX_QUEUE = 1
        first = await self.client.post(self.server.make_url("/v1/messages"), json={"stream":True}, headers=self.headers(**{"x-test-mode":"hold"}))
        self.assertEqual(first.status, 200)
        queued = asyncio.create_task(self.client.post(self.server.make_url("/v1/messages"), json={}, headers=self.headers()))
        async with asyncio.timeout(1):
            while gateway.waiting != 1:
                await asyncio.sleep(0.01)
        async with self.client.post(self.server.make_url("/v1/messages"), json={}, headers=self.headers()) as r:
            self.assertEqual(r.status, 429)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        async with asyncio.timeout(1):
            while gateway.waiting:
                await asyncio.sleep(0.01)
        self.release.set()
        await first.read()
        first.close()
        async with self.client.post(self.server.make_url("/v1/messages"), json={}, headers=self.headers()) as r:
            self.assertEqual(r.status, 200)


if __name__ == "__main__":
    unittest.main()
