"""
Daggr Studio's composition layer — one Space, one process, five surfaces.

    /            the custom single-page frontend (ours; the front door)
    /api/*       JSON + NDJSON endpoints (api.py) — what the frontend and agents call
    /api/...     plus a few @app.api() endpoints, which ALSO get Gradio's queue, streaming
                 and gradio_client/MCP compatibility
    /canvas/     the real daggr canvas (node-by-node inspection), served through the
                 prefix-rewriting proxy because its frontend hardcodes absolute paths
    /builder/    the original Gradio Builder, kept as a fallback surface and for the
                 canvas-style widget flow
    /docs        FastAPI's OpenAPI docs for the /api/* endpoints

Built on `gradio.Server` (gradio >= 6, April 2026): it *is* a FastAPI app with Gradio's API
engine on top, which is what lets a hand-written frontend sit in front of the same backend
that gradio_client can drive.

Security posture (deliberate):
* queued `@app.api()` endpoints never accept a token — credentials stay on the plain
  /api/* routes, which are not recorded in Gradio's run history;
* tokens are used per-request, never stored, never logged, never put in a URL.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from daggrstudio.web.api import router as api_router
from daggrstudio.web.canvas import BUILDER_BUTTON, CANVAS

try:  # the canvas proxy is optional: without it we mount nothing rather than a broken path
    from daggrstudio.web.proxy import PrefixProxy
except Exception:  # pragma: no cover - only while the proxy module is absent
    PrefixProxy = None  # type: ignore[assignment]

#: (name, function) for every endpoint registered on Gradio's queue, so tests can assert
#: that none of them accepts a credential.
QUEUED_ENDPOINTS: list[tuple[str, object]] = []

SPA_DIR = Path(__file__).parent / "spa"
INDEX = SPA_DIR / "index.html"


def _build() -> "object":
    from gradio import Server

    app = Server()

    # ── 1. our JSON/NDJSON API ─────────────────────────────────────────────────
    app.include_router(api_router)

    # ── 2. queued endpoints (gradio_client + MCP friendly, token-free by design) ─
    _register_queued_endpoints(app)

    # ── 3. the custom frontend ─────────────────────────────────────────────────
    if SPA_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(SPA_DIR)), name="ui")

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        return HTMLResponse(_index_html())

    @app.get("/healthz")
    async def healthz():
        from fastapi.responses import JSONResponse

        from daggrstudio.registry.bricks import get_registry

        registry = get_registry()
        return JSONResponse({
            "ok": True,
            "ui": "spa",
            "registry": registry.summary(),
            "canvas": {"status": CANVAS.status, "ready": CANVAS.has_graph,
                       "error": CANVAS.error},
        })

    # ── 4. the daggr canvas, under a prefix ────────────────────────────────────
    if PrefixProxy is not None:
        app.mount("/canvas", PrefixProxy(CANVAS, prefix="/canvas",
                                         inject=_canvas_injection()), name="canvas")
    else:
        print("[studio] canvas proxy unavailable — /canvas/ is disabled "
              "(the Studio and its API are unaffected)")

    # ── 5. the original Gradio Builder stays reachable ─────────────────────────
    try:
        import gradio as gr

        from daggrstudio.web.builder import build_ui

        gr.mount_gradio_app(app, build_ui(), path="/builder")
    except Exception as exc:  # a broken fallback UI must not take the Space down
        print(f"[studio] Gradio builder unavailable: {type(exc).__name__}: {exc}")

    # Gradio mounts the builder at '/builder/' internally; without this redirect the
    # canvas catch-all answers bare '/builder' and the user sees the canvas SPA again
    # (which is exactly the bug this whole rewrite fixes).
    @app.get("/builder")
    async def builder_redirect() -> RedirectResponse:
        return RedirectResponse("/builder/", status_code=307)

    # ── 6. SPA catch-all (client-side routes) — registered LAST ────────────────
    @app.get("/{path:path}", response_class=HTMLResponse)
    async def spa_routes(path: str) -> HTMLResponse:
        return HTMLResponse(_index_html())

    return app


def _index_html() -> str:
    try:
        html = INDEX.read_text()
    except Exception:
        return "<h1>Daggr Studio</h1><p>The frontend files are missing from this image.</p>"
    # a tiny freshness marker so a stale cached shell is obvious in the console
    return html.replace("</head>", f"<!-- {os.environ.get('DAGGRSTUDIO_BUILD', 'dev')} -->\n</head>", 1)


def _canvas_injection() -> str:
    """
    Markup injected into daggr's canvas page: a link back to the Studio, plus a runtime patch
    that makes the canvas sub-path-safe.

    The patch exists because daggr builds its websocket URL at runtime from
    ``window.location.host`` (```${proto}//${host}/ws/${id}```). No amount of response
    rewriting can fix a URL that does not exist until the page runs, so we wrap
    ``WebSocket`` (and ``fetch``/``EventSource``) to add the prefix for same-origin requests.
    """
    link = BUILDER_BUTTON.replace('href="/builder"', 'href="/"').replace(
        "Daggr Studio Builder", "← Daggr Studio")
    return link + _runtime_prefix_patch()


def _runtime_prefix_patch() -> str:
    """A small, defensive shim: same-origin /ws, /api, /file URLs get the canvas prefix."""
    return """
<script>
(function () {
  var PREFIX = "/canvas";
  var ROOTS = ["/ws/", "/api/", "/file/", "/daggr-assets/", "/assets/"];
  function fix(url) {
    if (typeof url !== "string") return url;
    try {
      var parsed = new URL(url, window.location.origin);
      if (parsed.origin !== window.location.origin) return url;
      if (parsed.pathname.indexOf(PREFIX + "/") === 0) return url;
      for (var i = 0; i < ROOTS.length; i++) {
        if (parsed.pathname.indexOf(ROOTS[i]) === 0) {
          parsed.pathname = PREFIX + parsed.pathname;
          return parsed.toString();
        }
      }
    } catch (err) { /* leave unusual URLs alone */ }
    return url;
  }
  var NativeWS = window.WebSocket;
  function PatchedWS(url, protocols) {
    var target = fix(String(url));
    return protocols === undefined ? new NativeWS(target) : new NativeWS(target, protocols);
  }
  PatchedWS.prototype = NativeWS.prototype;
  ["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach(function (k) {
    try { PatchedWS[k] = NativeWS[k]; } catch (err) {}
  });
  window.WebSocket = PatchedWS;

  var nativeFetch = window.fetch;
  window.fetch = function (input, init) {
    if (typeof input === "string") input = fix(input);
    else if (input && input.url) {
      var rewritten = fix(input.url);
      if (rewritten !== input.url) input = new Request(rewritten, input);
    }
    return nativeFetch.call(this, input, init);
  };

  if (window.EventSource) {
    var NativeES = window.EventSource;
    window.EventSource = function (url, config) {
      return new NativeES(fix(String(url)), config);
    };
    window.EventSource.prototype = NativeES.prototype;
  }
  window.__daggrCanvasPrefix = PREFIX;
})();
</script>
"""


def _register_queued_endpoints(app) -> None:
    """
    A few endpoints through Gradio's queue, so this Space is callable from
    `gradio_client` / MCP / other Spaces the way a Gradio app normally would be.

    They take NO token argument on purpose: anything queued can show up in Gradio's run
    history, and credentials must never end up there. Token-bearing work stays on /api/*.
    """
    import json

    from daggrstudio.web import services

    @app.api(name="plan_workflow", description="Describe a goal; get a validated, healed workflow")
    def plan_workflow(intent: str, industry: str = "general",
                      license_posture: str = "commercial-only", max_steps: int = 4) -> dict:
        result = services.plan_workflow(intent, token=None, industry=industry,
                                        license_posture=license_posture,
                                        max_steps=int(max_steps), live_validation=True)
        result.pop("pool", None)
        return {k: v for k, v in result.items() if k != "code"}

    @app.api(name="validate_workflow", description="Validate a workflow JSON (offline checks)")
    def validate_workflow(spec_json: str) -> dict:
        spec = services.as_spec(json.loads(spec_json))
        if spec is None:
            return {"ok": False, "message": "that JSON is not a workflow spec"}
        return services.validate_spec(spec, live=False)

    QUEUED_ENDPOINTS.extend([("plan_workflow", plan_workflow),
                             ("validate_workflow", validate_workflow)])

    @app.api(name="registry_summary", description="What the verified brick registry contains")
    def registry_summary() -> dict:
        from daggrstudio.registry.bricks import get_registry

        registry = get_registry()
        return {
            "summary": registry.summary(),
            "modalities": registry.modalities_present(),
            "industries": registry.industries_present(),
            "proven": [b.id for b in registry.all() if b.is_proven],
        }

    QUEUED_ENDPOINTS.append(("registry_summary", registry_summary))


app = _build()
