"""
Daggr Studio — Space entrypoint.

Two UIs, one process:

* ``/builder``  — the Gradio Builder (mounted first, so it wins over the canvas catch-all)
* ``/``         — the daggr canvas, whose prebuilt frontend uses ABSOLUTE asset/API paths
                  and therefore has to own the root

Run locally with ``python app.py`` (uvicorn on :7860), or as an HF Space with
``sdk: docker`` so we control the Python/gradio versions instead of inheriting them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# make the package importable when running from the repo root (HF Spaces does this too)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from daggrstudio.web.canvas import CANVAS  # noqa: E402

app = FastAPI(title="Daggr Studio")

#: Exposed for HF Spaces and for uptime checks / the cron liveness job.
_STARTED_AT = __import__("time").time()


@app.get("/healthz")
async def healthz():
    from daggrstudio.registry.bricks import get_registry

    registry = get_registry()
    return JSONResponse({
        "ok": True,
        "canvas": CANVAS.status,
        "canvas_ready": CANVAS.has_graph,
        "registry": registry.summary(),
        "uptime_s": round(__import__("time").time() - _STARTED_AT, 1),
    })


@app.get("/api/bricks")
async def api_bricks(modality: str = "any", commercial_only: bool = True, limit: int = 60):
    """Agent-friendly read of the brick catalogue (the same data the UI shows)."""
    from daggrstudio.web import services

    rows = services.bricks_table(modality=modality, only_commercial=commercial_only,
                                 limit=limit)
    return JSONResponse({"headers": services.BRICK_HEADERS, "rows": rows})


@app.get("/api/leaderboard")
async def api_leaderboard(modality: str = "any", industry: str = "any",
                          license_filter: str = "commercial-only", sort: str = "votes"):
    from daggrstudio.web import services

    return JSONResponse(services.leaderboard_view(modality=modality, industry=industry,
                                                  license_filter=license_filter, sort=sort))


@app.get("/api/canvas")
async def api_canvas():
    """Which workflow the canvas is currently showing, if any."""
    return JSONResponse({"status": CANVAS.status, "ready": CANVAS.has_graph})


@app.post("/api/plan")
async def api_plan(request: Request):
    """
    Agent entry point: intent -> healed workflow.

    Body: {"intent": str, "industry": str, "license_posture": str, "max_steps": int,
           "model": str, "token": str, "heal_rounds": int}
    The token is optional and used only for this request (BYOK); without it the Space's own
    token is metered against the community pool. Tokens are never logged or stored.
    """
    from daggrstudio.web import services

    payload = await _json_body(request)
    intent = str(payload.get("intent") or payload.get("prompt") or "").strip()
    if not intent:
        return JSONResponse({"ok": False, "message": "provide an 'intent'"}, status_code=400)
    result = services.plan_workflow(
        intent,
        token=payload.get("token") or None,
        model=payload.get("model") or None,
        industry=str(payload.get("industry") or "general"),
        license_posture=str(payload.get("license_posture") or "commercial-only"),
        max_steps=int(payload.get("max_steps") or 4),
        live_validation=bool(payload.get("live_validation", True)),
        heal_rounds=int(payload.get("heal_rounds") or 4),
    )
    # never echo anything that could carry a credential
    result.pop("pool", None)
    # keep the canvas and the agent API consistent: a planned workflow is *shown*, too
    if result.get("spec"):
        from daggrstudio.web import services as svc
        from daggrstudio.web.canvas import show_spec

        spec = svc.as_spec(result["spec"])
        if spec is not None:
            result["canvas"] = show_spec(spec)
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


@app.post("/api/validate")
async def api_validate(request: Request):
    """Validate (and optionally heal) a spec posted by an agent."""
    from daggrstudio.web import services

    payload = await _json_body(request)
    spec = payload.get("spec") or payload
    if payload.get("heal"):
        result = services.heal_spec(spec, token=payload.get("token") or None,
                                    model=payload.get("model") or None,
                                    live=bool(payload.get("live_validation", True)))
    else:
        result = services.validate_spec(spec, live=bool(payload.get("live_validation", True)),
                                        token=payload.get("token") or None)
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


async def _json_body(request: Request) -> dict:
    try:
        payload = await request.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _compose():
    """Mount the Builder at /builder, then the canvas (and everything else) at /."""
    import gradio as gr

    from daggrstudio.web.builder import build_ui

    # Building the Blocks only defines the UI; nothing is served yet.
    demo = build_ui()
    composed = gr.mount_gradio_app(app, demo, path="/builder")
    # Registered last on purpose: daggr's frontend is a catch-all SPA.
    composed.mount("/", CANVAS)
    return composed


app = _compose()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")