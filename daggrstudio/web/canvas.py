"""
The daggr canvas, mounted at the root of the Space - hot-swapped to whatever was just built.

Why the canvas owns ``/``: daggr's prebuilt frontend requests ``/assets/...``, ``/theme.css``,
``/api/...``, ``/ws/...`` and ``/file/...`` as **absolute** paths. Mounting it under a
sub-path would 404 its own assets, so the Builder lives at ``/builder`` and this shim holds
the root. Consequence: the root must also serve a real page when no workflow exists yet, so
a first-time visitor sees how to start rather than a broken SPA.

The shim is an ASGI callable that can be re-pointed at a freshly built ``DaggrServer``
without restarting the process - which is what makes "build then inspect on the canvas"
work in a single Space.
"""

from __future__ import annotations

import html
import json
from typing import Any

#: Injected into the canvas page so the Builder is always one click away.
BUILDER_BUTTON = """
<div id="daggr-studio-nav" style="position:fixed;left:16px;bottom:16px;z-index:99999;
     font-family:'Space Grotesk',system-ui,sans-serif;">
  <a href="/builder" style="display:inline-flex;align-items:center;gap:8px;
     background:#4f46e5;color:#fff;text-decoration:none;font-weight:600;
     padding:10px 16px;border-radius:999px;box-shadow:0 6px 24px rgba(0,0,0,.35);">
    <span style="font-size:16px">🧱</span> Daggr Studio Builder
  </a>
</div>
"""

_EMPTY_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Daggr Studio — canvas</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
         background:radial-gradient(circle at 20% 20%, #1e1b4b, #09090b 60%);
         color:#e5e7eb; font-family:'Space Grotesk',system-ui,-apple-system,sans-serif; }}
  .card {{ max-width:640px; padding:40px; border-radius:24px;
           background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.12);
           box-shadow:0 24px 80px rgba(0,0,0,.5); }}
  h1 {{ font-size:28px; margin:0 0 8px; }}
  p  {{ line-height:1.6; color:#c7c9d1; }}
  a.cta {{ display:inline-block; margin-top:20px; background:#4f46e5; color:#fff;
           text-decoration:none; padding:12px 22px; border-radius:999px; font-weight:600; }}
  code {{ background:rgba(255,255,255,.08); padding:2px 6px; border-radius:6px; }}
  .muted {{ color:#8b8d98; font-size:13px; margin-top:28px; }}
</style></head>
<body><div class="card">
  <h1>🧱 Daggr Studio</h1>
  <p>This is the daggr canvas. It shows the workflow you build in the Studio — with every
     step's output, so you can inspect and re-run individual bricks.</p>
  <p>{body}</p>
  <a class="cta" href="/builder">Open the Builder</a>
  <p class="muted">No workflow loaded yet. Describe what you want in the Builder and the
     canvas will be rebuilt around it.</p>
</div></body></html>
"""


class CanvasRouter:
    """ASGI app mounted at ``/`` that delegates to the active daggr server."""

    def __init__(self) -> None:
        self._app: Any | None = None
        self._graph: Any | None = None
        self._spec: dict[str, Any] | None = None
        self._status: str = "no workflow built yet"

    # ── state ──────────────────────────────────────────────────────────────────

    @property
    def has_graph(self) -> bool:
        return self._app is not None

    @property
    def status(self) -> str:
        return self._status

    def clear(self) -> None:
        self._app = None
        self._graph = None
        self._spec = None
        self._status = "no workflow built yet"

    def set_graph(self, graph: Any, spec: Any | None = None) -> str:
        """
        Point the canvas at a graph. Returns a human-readable status.

        daggr's ``DaggrServer`` is created here (it does not bind a port - we only use its
        ASGI app), which is why swapping the canvas costs nothing.
        """
        from daggr.server import DaggrServer

        try:
            server = DaggrServer(graph)
        except Exception as exc:  # pragma: no cover - only on a corrupted daggr install
            self._status = f"canvas unavailable: {type(exc).__name__}: {exc}"
            return self._status

        self._app = server.app
        self._graph = graph
        self._spec = spec.to_dict() if hasattr(spec, "to_dict") else spec
        self._status = (f"canvas shows '{getattr(graph, 'name', 'workflow')}' "
                        f"({len(getattr(graph, 'nodes', {}))} nodes)")
        return self._status

    def set_spec_error(self, message: str) -> None:
        self._status = message

    # ── ASGI ───────────────────────────────────────────────────────────────────

    @property
    def _is_websocket(self) -> bool:
        return False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            return

        path = scope.get("path", "/")

        # the root is ours: we render the canvas shell (or an onboarding page)
        if scope["type"] == "http" and path in ("/", "/index.html"):
            await self._send_html(send, self._root_html())
            return

        if self._app is None:
            if scope["type"] == "http":
                if path.startswith("/api/"):
                    await self._send_json(send, {"error": "no workflow built yet",
                                                 "hint": "open /builder and plan a workflow"},
                                          status=404)
                else:
                    await self._send_html(send, self._root_html(), status=200)
            else:
                await send({"type": "websocket.close", "code": 1011})
            return

        # everything else belongs to daggr (assets, api, ws, files)
        await self._app(scope, receive, send)

    # ── rendering ──────────────────────────────────────────────────────────────

    def _root_html(self) -> str:
        if self._app is None:
            return _EMPTY_PAGE.format(
                body="Start in the Builder: describe a goal, and Daggr Studio will assemble "
                     "a workflow from bricks that were verified to actually run."
            )
        return self._index_with_button()

    def _index_with_button(self) -> str:
        from pathlib import Path

        import daggr

        index = Path(daggr.__file__).parent / "frontend" / "dist" / "index.html"
        try:
            page = index.read_text()
        except Exception:
            return _EMPTY_PAGE.format(body="daggr's frontend files could not be read.")

        name = ""
        if isinstance(self._spec, dict):
            name = html.escape(str(self._spec.get("name", "")))
        banner = (
            f'<div id="daggr-studio-title" style="position:fixed;top:12px;right:16px;'
            f'z-index:99999;font-family:\'Space Grotesk\',system-ui,sans-serif;font-size:13px;'
            f'color:#a5b4fc;background:rgba(15,15,25,.75);padding:6px 12px;border-radius:999px">'
            f'built with Daggr Studio{" · " + name if name else ""}</div>'
        ) if name else ""
        injection = banner + BUILDER_BUTTON
        if "</body>" in page:
            return page.replace("</body>", injection + "</body>", 1)
        return page + injection

    async def _send_html(self, send: Any, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        await send({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"text/html; charset=utf-8"),
            (b"content-length", str(len(encoded)).encode()),
            (b"cache-control", b"no-store"),
        ]})
        await send({"type": "http.response.body", "body": encoded})

    async def _send_json(self, send: Any, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload).encode()
        await send({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode()),
        ]})
        await send({"type": "http.response.body", "body": encoded})


#: Process-wide canvas (one Space, one active workflow).
CANVAS = CanvasRouter()


def show_spec(spec: Any) -> str:
    """Build the graph for a spec and put it on the canvas. Returns a status string."""
    from daggrstudio.codegen import build_graph
    from daggrstudio.registry.bricks import get_registry

    if spec is None:
        CANVAS.clear()
        return "canvas cleared"
    try:
        graph = build_graph(spec, get_registry())
    except Exception as exc:
        CANVAS.set_spec_error(f"could not build the canvas: {type(exc).__name__}: {exc}")
        return CANVAS.status
    return CANVAS.set_graph(graph, spec)