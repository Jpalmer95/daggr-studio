"""
The Studio API: everything the custom frontend (and any external agent) needs, as JSON.

Why this exists separately from the Gradio Builder: a custom frontend must not be limited to
what Gradio components can express. These endpoints are plain JSON/NDJSON, so the SPA, a
`curl` agent, and `gradio_client` (via the `@app.api()` wrappers in studio.py) all drive the
same code paths in `services.py`.

Design rules:
* **No endpoint takes a token from a URL.** Tokens arrive in the request body only, are used
  for that call, and are never stored, logged, or echoed back (see tests/test_api.py).
* **Artifacts are served by opaque id**, never by path. A run registers the files it produced
  and the client fetches `/api/artifact/<id>`; there is no way to ask this server for an
  arbitrary path on disk.
* **Long work streams.** Planning, healing and running report progress as NDJSON so the UI can
  show what is happening instead of a spinner.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from daggrstudio.registry.bricks import get_registry
from daggrstudio.spec import WorkflowSpec
from daggrstudio.web import services

router = APIRouter(prefix="/api", tags=["studio"])

MAX_ARTIFACTS = 400


class ArtifactStore:
    """
    id -> path registry for files produced by runs.

    Deliberately not a filesystem server: a client can only fetch ids we handed out during a
    run in this process, so there is no path-traversal surface. Bounded so a long-lived Space
    cannot grow without limit.
    """

    def __init__(self, limit: int = MAX_ARTIFACTS):
        self._items: OrderedDict[str, Path] = OrderedDict()
        self._limit = limit

    def register(self, path: str | Path) -> str | None:
        p = Path(path)
        if not p.is_file():
            return None
        for known, existing in self._items.items():
            if existing == p:
                return known
        artifact_id = uuid.uuid4().hex[:16]
        self._items[artifact_id] = p
        while len(self._items) > self._limit:
            self._items.popitem(last=False)
        return artifact_id

    def register_many(self, paths: list[str]) -> list[dict[str, str]]:
        out = []
        for path in paths:
            artifact_id = self.register(path)
            if artifact_id:
                out.append({"id": artifact_id, "name": Path(path).name,
                            "url": f"/api/artifact/{artifact_id}",
                            "kind": _kind_of(path)})
        return out

    def resolve(self, artifact_id: str) -> Path | None:
        return self._items.get(artifact_id)

    def __len__(self) -> int:
        return len(self._items)


def _kind_of(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
        return "image"
    if suffix in (".mp4", ".webm", ".mov"):
        return "video"
    if suffix in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        return "audio"
    if suffix in (".glb", ".gltf", ".obj", ".stl", ".fbx"):
        return "model3d"
    if suffix in (".json", ".txt", ".md", ".csv"):
        return "text"
    return "file"


ARTIFACTS = ArtifactStore()


def _body(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _spec_or_400(payload: dict[str, Any]) -> WorkflowSpec:
    spec = services.as_spec(payload.get("spec") or payload.get("workflow"))
    if spec is None:
        raise HTTPException(status_code=400, detail="provide a 'spec' object")
    return spec


# ─── status ───────────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> JSONResponse:
    from daggrstudio.web.canvas import CANVAS

    registry = get_registry()
    return JSONResponse({
        "ok": True,
        "registry": registry.summary(),
        "counts": {
            "bricks": len(registry),
            "reachable": sum(1 for b in registry.all() if b.status == "running"),
            "proven": sum(1 for b in registry.all() if b.is_proven),
            "commercial": sum(1 for b in registry.all() if b.commercial),
        },
        "pool": services.pool_snapshot(),
        "canvas": {"status": CANVAS.status, "ready": CANVAS.has_graph, "error": CANVAS.error},
        "artifacts": len(ARTIFACTS),
    })


@router.get("/artifact/{artifact_id}")
async def artifact(artifact_id: str):
    """Serve a file this process produced during a run (by opaque id, never by path)."""
    path = ARTIFACTS.resolve(artifact_id)
    if path is None:
        raise HTTPException(status_code=404, detail="unknown artifact")
    media_types = {
        "image": "image/png", "video": "video/mp4", "audio": "audio/wav",
        "model3d": "model/gltf-binary", "text": "text/plain",
    }
    return FileResponse(path, media_type=media_types.get(_kind_of(str(path)), "application/octet-stream"))


# ─── settings ─────────────────────────────────────────────────────────────────


@router.post("/verify_token")
async def verify_token(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    token = (body.get("token") or "").strip()
    if not token:
        return JSONResponse({"ok": True, "username": "", "message": "no token set"})
    username = services.verify_username(token)
    return JSONResponse({
        "ok": bool(username),
        "username": username or "",
        "message": f"signed in as {username}" if username else "that token was rejected",
    })


@router.get("/models")
async def models(token: str | None = None) -> JSONResponse:
    """Curated + live model list for the planner/medic dropdown."""
    return JSONResponse({"models": services.list_models_for_ui(token),
                         "default": services.default_model()})


# ─── the core journey ─────────────────────────────────────────────────────────


@router.post("/plan")
async def plan(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    intent = str(body.get("intent") or "").strip()
    if not intent:
        raise HTTPException(status_code=400, detail="provide an 'intent'")
    result = services.plan_workflow(
        intent,
        token=body.get("token") or None,
        model=body.get("model") or None,
        industry=str(body.get("industry") or "general"),
        license_posture=str(body.get("license_posture") or "commercial-only"),
        max_steps=int(body.get("max_steps") or 4),
        live_validation=bool(body.get("live_validation", True)),
        heal_rounds=int(body.get("heal_rounds") or 4),
    )
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


@router.post("/plan/stream")
async def plan_stream(payload: dict = Body(default={})):
    """NDJSON progress while planning+healing: stage events, then the final payload."""
    body = _body(payload)
    intent = str(body.get("intent") or "").strip()
    if not intent:
        raise HTTPException(status_code=400, detail="provide an 'intent'")
    return _ndjson_response(_plan_events(body, intent))


async def _plan_events(body: dict[str, Any], intent: str) -> AsyncIterator[bytes]:
    yield _event({"stage": "planning", "message": "asking the model to assemble bricks…"})
    result = await asyncio.to_thread(
        services.plan_workflow,
        intent,
        body.get("token") or None,
        body.get("model") or None,
        str(body.get("industry") or "general"),
        str(body.get("license_posture") or "commercial-only"),
        int(body.get("max_steps") or 4),
        bool(body.get("live_validation", True)),
        int(body.get("heal_rounds") or 4),
    )
    spec = result.get("spec") or {}
    for index, step in enumerate(spec.get("steps") or [], start=1):
        yield _event({"stage": "workflow", "message": f"brick {index}: {step.get('brick_id')}",
                      "step": step.get("id")})
    status = result.get("status") or "unknown"
    yield _event({"stage": "healing", "message": f"status: {status}",
                  "timeline": result.get("timeline", ""),
                  "issues": result.get("issues") or []})
    yield _event({"stage": "done", "result": result})


@router.post("/validate")
async def validate(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    spec = _spec_or_400(body)
    result = services.validate_spec(spec, live=bool(body.get("live_validation", True)),
                                    token=body.get("token") or None)
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


@router.post("/heal")
async def heal(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    spec = _spec_or_400(body)
    result = services.heal_spec(spec, token=body.get("token") or None,
                                model=body.get("model") or None,
                                live=bool(body.get("live_validation", True)))
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


@router.post("/check")
async def check(payload: dict = Body(default={})) -> JSONResponse:
    """Re-read the live APIs of every Space this workflow uses."""
    body = _body(payload)
    spec = _spec_or_400(body)
    return JSONResponse(services.check_workflow_bricks(spec, token=body.get("token") or None))


@router.post("/advise")
async def advise(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    spec = _spec_or_400(body)
    return JSONResponse(services.advise_upgrades(spec, token=body.get("token") or None,
                                                 model=body.get("model") or None))


@router.post("/apply_upgrade")
async def apply_upgrade(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    spec = _spec_or_400(body)
    result = services.apply_upgrade(spec, brick_id=str(body.get("brick_id") or ""),
                                    step_id=str(body.get("step") or ""))
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


def _event(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, default=str) + "\n").encode()


def _ndjson_response(generator: AsyncIterator[bytes]):
    from fastapi.responses import StreamingResponse

    return StreamingResponse(generator, media_type="application/x-ndjson",
                             headers={"cache-control": "no-store",
                                      "x-accel-buffering": "no"})


# ─── running (streamed) ───────────────────────────────────────────────────────


@router.post("/run")
async def run(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    spec = _spec_or_400(body)
    result = await asyncio.to_thread(services.run_workflow, spec,
                                     body.get("values") or {}, body.get("token") or None)
    result["artifacts"] = ARTIFACTS.register_many(result.get("artifacts") or [])
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


@router.post("/run/stream")
async def run_stream(payload: dict = Body(default={})):
    """
    Run the workflow and stream per-step events (NDJSON).

    This is the workflow that matters in the UI: the user watches each brick execute, sees
    repairs as they happen, and gets artifacts inline instead of waiting on one blocking call.
    """
    body = _body(payload)
    spec = _spec_or_400(body)
    return _ndjson_response(_run_events(spec, body))


async def _run_events(spec: WorkflowSpec, body: dict[str, Any]) -> AsyncIterator[bytes]:
    from daggrstudio.runner import run_spec

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress(step_run: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"stage": "step", **step_run.to_dict()})

    def worker() -> dict[str, Any]:
        try:
            result = run_spec(spec, values=body.get("values") or {},
                              token=body.get("token") or None,
                              registry=get_registry(), progress=progress)
            return {"ok": result.ok, "message": result.summary(), "note": result.note,
                    "artifacts": result.artifacts, "rows": [s.to_dict() for s in result.steps]}
        except Exception as exc:  # a crashed run is reported, not swallowed
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}", "artifacts": [],
                    "rows": []}

    task = asyncio.create_task(asyncio.to_thread(worker))
    yield _event({"stage": "started", "message": f"running {len(spec.steps)} steps"})
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if task.done():
                break
            yield _event({"stage": "heartbeat"})
            continue
        yield _event(event)
    result = await task
    result["artifacts"] = ARTIFACTS.register_many(result.get("artifacts") or [])
    yield _event({"stage": "done", "result": result})


# ─── sharing ──────────────────────────────────────────────────────────────────


@router.post("/publish")
async def publish(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    spec = _spec_or_400(body)
    result = services.publish_workflow(spec, token=body.get("token") or None,
                                       note=str(body.get("note") or ""))
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


@router.post("/load")
async def load(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    slug = str(body.get("slug") or "").strip()
    if not slug:
        raise HTTPException(status_code=400, detail="provide a 'slug'")
    result = services.load_workflow(slug, token=body.get("token") or None)
    return JSONResponse(result, status_code=200 if result.get("ok") else 404)


@router.get("/leaderboard")
async def leaderboard(modality: str = "any", industry: str = "any",
                      license_filter: str = "commercial-only", compute_tier: str = "any",
                      sort: str = "votes") -> JSONResponse:
    return JSONResponse(services.leaderboard_view(
        modality=modality, industry=industry, license_filter=license_filter,
        compute_tier=compute_tier, sort=sort))


@router.post("/vote")
async def vote(payload: dict = Body(default={})) -> JSONResponse:
    body = _body(payload)
    slug = str(body.get("slug") or "").strip()
    if not slug:
        raise HTTPException(status_code=400, detail="provide a 'slug'")
    result = services.cast_vote(slug, int(body.get("direction") or 1),
                                token=body.get("token") or None)
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


@router.post("/deploy")
async def deploy(payload: dict = Body(default={})) -> JSONResponse:
    """Create the user's own Space running this workflow (BYOK)."""
    body = _body(payload)
    spec = _spec_or_400(body)
    result = services.deploy_space(spec, token=body.get("token") or None,
                                   space_name=str(body.get("name") or ""))
    return JSONResponse(result, status_code=200 if result.get("ok") else 422)


# ─── registry ─────────────────────────────────────────────────────────────────


@router.get("/bricks")
async def bricks(modality: str = "any", commercial_only: bool = True,
                 reachable_only: bool = False, limit: int = 80) -> JSONResponse:
    return JSONResponse({
        "headers": services.BRICK_HEADERS,
        "rows": services.bricks_table(modality=modality, only_commercial=commercial_only,
                                      only_running=reachable_only, limit=limit),
        "modalities": ["any"] + get_registry().modalities_present(),
        "summary": get_registry().summary(),
    })


@router.post("/registry/verify")
async def registry_verify(payload: dict = Body(default={})) -> JSONResponse:
    """
    Re-verify bricks right now (introspect + optionally execute one real call).

    Deliberately capped: this makes network calls to other people's Spaces, so it is meant
    for a handful of ids at a time, not a full refresh (that is `scripts/verify_registry.py`).
    """
    from daggrstudio.medic.autofix import run_autofixes  # noqa: F401  (import kept for parity)
    from daggrstudio.validator import validate

    body = _body(payload)
    ids = [str(i) for i in (body.get("ids") or [])][:6]
    registry = get_registry()
    report_rows: list[dict[str, Any]] = []
    for brick_id in ids:
        brick = registry.by_id(brick_id)
        if brick is None or not brick.source:
            report_rows.append({"id": brick_id, "ok": False, "error": "unknown brick"})
            continue
        probe = None
        if brick.kind == "space" and brick.api_name:
            from scripts.verify_registry import probe_space  # noqa: PLC0415

            probe = probe_space(brick.source, brick.api_name, [], body.get("token") or None)
        report_rows.append({"id": brick_id, "ok": bool(probe and probe.get("ok")),
                            "error": (probe or {}).get("error"),
                            "api_name": brick.api_name, "source": brick.source})
    spec = services.as_spec(body.get("spec"))
    findings = []
    if spec is not None:
        findings = [i.to_dict() for i in validate(spec, registry, live=False).issues]
    return JSONResponse({"ok": True, "bricks": report_rows, "findings": findings})
