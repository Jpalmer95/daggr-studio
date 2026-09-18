"""PrefixProxy: the contract that lets daggr's canvas live under /canvas.

Every assertion here maps to a way a prefix-mounted frontend breaks. The rewrite rules are
deliberately narrow, so most of these tests are about what must NOT be touched.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse

from daggrstudio.web.proxy import PrefixProxy, strip_prefix


def make_inner() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(
            '<html><head><title>inner</title></head><body>'
            '<link rel="stylesheet" href="/theme.css">'
            '<script type="module" src="/assets/index-abc.js"></script>'
            "</body></html>"
        )

    @app.get("/assets/index-abc.js")
    async def asset() -> Response:
        return Response(
            'const g = await fetch("/api/graph");'
            'const ws = new WebSocket(`/ws/${id}`);'
            'const png = "/file/tmp/a.png";',
            media_type="application/javascript",
        )

    @app.get("/api/graph")
    async def graph() -> JSONResponse:
        # JSON must never be rewritten, even though it contains a path-like string
        return JSONResponse({"url": "/api/graph", "path": "/assets/keep.js"})

    @app.get("/tree/other")
    async def other() -> HTMLResponse:
        return HTMLResponse("<html><body>other page</body></html>")

    @app.get("/stream")
    async def stream() -> Response:
        # no content-length -> the proxy must stream it through untouched
        return Response(b"/api/should-not-change", media_type="application/javascript")

    return app


def client_for(proxy: PrefixProxy) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy), base_url="http://t")


@pytest.mark.asyncio
async def test_paths_are_stripped_before_forwarding():
    proxy = PrefixProxy(make_inner(), "/canvas")
    async with client_for(proxy) as c:
        assert (await c.get("/canvas/")).status_code == 200
        assert (await c.get("/canvas/api/graph")).status_code == 200
        assert (await c.get("/canvas/tree/other")).status_code == 200
    assert strip_prefix("/canvas/api/graph", "/canvas") == "/api/graph"
    assert strip_prefix("/canvas", "/canvas") == "/"
    assert strip_prefix("/elsewhere", "/canvas") == "/elsewhere"


@pytest.mark.asyncio
async def test_non_prefixed_paths_are_not_swallowed():
    proxy = PrefixProxy(make_inner(), "/canvas")
    async with client_for(proxy) as c:
        resp = await c.get("/api/graph")  # our own app's route must stay reachable
    assert resp.status_code == 404
    assert "canvas" in resp.json()["hint"]


@pytest.mark.asyncio
async def test_passthrough_mode_forwards_everything():
    proxy = PrefixProxy(make_inner(), "/canvas", passthrough_other=True)
    async with client_for(proxy) as c:
        assert (await c.get("/api/graph")).status_code == 200


@pytest.mark.asyncio
async def test_html_is_rewritten_and_markup_injected():
    proxy = PrefixProxy(make_inner(), "/canvas", inject="<div id='back'>back</div>")
    async with client_for(proxy) as c:
        resp = await c.get("/canvas/")
    body = resp.text
    assert 'href="/canvas/theme.css"' in body
    assert 'src="/canvas/assets/index-abc.js"' in body
    assert '<base href="/canvas/">' in body
    assert "<div id='back'>back</div></body>" in body.replace("\n", "")


@pytest.mark.asyncio
async def test_js_absolute_urls_are_rewritten():
    proxy = PrefixProxy(make_inner(), "/canvas")
    async with client_for(proxy) as c:
        body = (await c.get("/canvas/assets/index-abc.js")).text
    assert 'fetch("/canvas/api/graph")' in body
    assert "new WebSocket(`/canvas/ws/${id}`)" in body
    assert '"/canvas/file/tmp/a.png"' in body


@pytest.mark.asyncio
async def test_rewritten_asset_path_actually_resolves_through_the_proxy():
    proxy = PrefixProxy(make_inner(), "/canvas")
    async with client_for(proxy) as c:
        html = (await c.get("/canvas/")).text
        assert "/canvas/assets/index-abc.js" in html
        asset = await c.get("/canvas/assets/index-abc.js")
    assert asset.status_code == 200 and "fetch(" in asset.text


@pytest.mark.asyncio
async def test_json_is_never_rewritten():
    proxy = PrefixProxy(make_inner(), "/canvas")
    async with client_for(proxy) as c:
        payload = (await c.get("/canvas/api/graph")).json()
    assert payload == {"url": "/api/graph", "path": "/assets/keep.js"}


@pytest.mark.asyncio
async def test_bodies_without_content_length_are_streamed_untouched():
    """Accurate content-length is what makes buffering safe; without it we do not buffer."""
    proxy = PrefixProxy(make_inner(), "/canvas")
    async with client_for(proxy) as c:
        body = (await c.get("/canvas/stream")).text
    assert body == "/api/should-not-change"


def test_rewrite_is_idempotent():
    proxy = PrefixProxy(make_inner(), "/canvas")
    once = proxy.rewrite_text('fetch("/api/graph")')
    twice = proxy.rewrite_text(once)
    assert once == 'fetch("/canvas/api/graph")'
    assert twice == once


def test_rewrite_ignores_prose():
    """Whitespace-preceded paths are prose, not URLs, and must survive untouched."""
    proxy = PrefixProxy(make_inner(), "/canvas")
    sample = "// see /api/docs for details\n# and /assets/ too"
    assert proxy.rewrite_text(sample) == sample


def test_rewrite_handles_css_and_multiple_targets():
    proxy = PrefixProxy(make_inner(), "/canvas")
    css = '@import "/theme.css"; .a{background:url(/assets/bg.png)}'
    out = proxy.rewrite_text(css)
    assert '@import "/canvas/theme.css"' in out
    assert "url(/canvas/assets/bg.png)" in out


def test_stats_count_forwarded_and_rewritten():
    proxy = PrefixProxy(make_inner(), "/canvas")
    assert proxy.stats() == {"requests": 0, "rewritten": 0, "forwarded": 0, "errors": 0}
    proxy.rewrite_text("/api/x")


@pytest.mark.asyncio
async def test_stats_record_real_traffic():
    proxy = PrefixProxy(make_inner(), "/canvas")
    async with client_for(proxy) as c:
        await c.get("/canvas/")              # rewritten
        await c.get("/canvas/api/graph")     # json -> forwarded
        await c.get("/nope")                 # rejected by the guard
    stats = proxy.stats()
    assert stats["requests"] == 3
    assert stats["rewritten"] >= 1 and stats["forwarded"] >= 1


class _EchoWebSocket:
    """Minimal ASGI app that records the websocket scope and echoes one message."""

    def __init__(self) -> None:
        self.scopes: list[dict] = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        await send({"type": "websocket.accept"})
        message = await receive()
        while message.get("type") == "websocket.connect":   # drain the handshake frame
            message = await receive()
        await send({"type": "websocket.send", "text": f"echo:{message.get('text', '')}"})
        await send({"type": "websocket.close", "code": 1000})


@pytest.mark.asyncio
async def test_websockets_are_forwarded_with_the_prefix_stripped():
    inner = _EchoWebSocket()
    proxy = PrefixProxy(inner, "/canvas")
    sent: list[dict] = []
    incoming = [{"type": "websocket.connect"}, {"type": "websocket.receive", "text": "hi"}]

    async def receive():
        return incoming.pop(0)

    async def send(message):
        sent.append(message)

    await proxy({"type": "websocket", "path": "/canvas/ws/session-1", "raw_path": b""},
                receive, send)

    assert inner.scopes[0]["path"] == "/ws/session-1"
    assert inner.scopes[0]["root_path"].endswith("/canvas")
    assert {"type": "websocket.accept"} in sent
    assert {"type": "websocket.send", "text": "echo:hi"} in sent


@pytest.mark.asyncio
async def test_a_failing_rewrite_falls_back_to_the_original_bytes(monkeypatch):
    proxy = PrefixProxy(make_inner(), "/canvas")

    def boom(_text: str) -> str:
        raise RuntimeError("rewriter exploded")

    monkeypatch.setattr(proxy, "rewrite_html_document", boom)
    async with client_for(proxy) as c:
        resp = await c.get("/canvas/")
    assert resp.status_code == 200
    assert 'href="/theme.css"' in resp.text          # untouched, not broken
    assert proxy.stats()["errors"] >= 1


@pytest.mark.asyncio
async def test_oversized_bodies_are_not_buffered():
    proxy = PrefixProxy(make_inner(), "/canvas", max_body=64)
    async with client_for(proxy) as c:
        body = (await c.get("/canvas/assets/index-abc.js")).text
    assert body.startswith("const g = await fetch(\"/api/graph\")")  # passed through


@pytest.mark.asyncio
async def test_websocket_to_an_unprefixed_path_is_closed_not_crashed():
    proxy = PrefixProxy(make_inner(), "/canvas")
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():  # pragma: no cover - never reached
        return {"type": "websocket.connect"}

    await proxy({"type": "websocket", "path": "/ws/other"}, receive, send)
    assert sent[0]["type"] == "websocket.close"