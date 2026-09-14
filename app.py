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

from fastapi import FastAPI  # noqa: E402
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