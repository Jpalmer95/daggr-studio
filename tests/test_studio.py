"""The Studio's HTTP surface: routing, tokens, artifacts, and the bug that broke the button.

Regression under test: daggr's canvas owns the root and answers unknown paths with its SPA.
Without an explicit redirect, `GET /builder` returned that SPA (200, no Gradio), so clicking
"Builder" looked like a page reload that did nothing. `/builder` must now redirect to the
Gradio mount.
"""

from __future__ import annotations

import json

import httpx
import pytest

from daggrstudio.web import api as api_module


def client() -> httpx.AsyncClient:
    from daggrstudio.web.studio import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://studio")


@pytest.mark.asyncio
async def test_frontend_shell_and_assets_are_served():
    async with client() as c:
        home = await c.get("/")
        css = await c.get("/ui/styles.css")
        js = await c.get("/ui/app.js")
    assert home.status_code == 200 and "Daggr Studio" in home.text
    assert "styles.css" in home.text and "app.js" in home.text
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


@pytest.mark.asyncio
async def test_unknown_paths_fall_through_to_the_spa_for_client_side_routes():
    async with client() as c:
        resp = await c.get("/some/deep/route")
    assert resp.status_code == 200
    assert "Daggr Studio" in resp.text


@pytest.mark.asyncio
async def test_builder_redirects_instead_of_being_swallowed_by_the_canvas():
    async with client() as c:
        resp = await c.get("/builder")
    assert resp.status_code == 307
    assert resp.headers["location"] == "/builder/"


@pytest.mark.asyncio
async def test_builder_mount_serves_gradio_at_the_trailing_slash():
    async with client() as c:
        resp = await c.get("/builder/")
    assert resp.status_code == 200
    # the Gradio shell, not our SPA
    assert "gradio" in resp.text.lower()


@pytest.mark.asyncio
async def test_health_reports_the_new_ui_and_the_registry():
    async with client() as c:
        plain = await c.get("/healthz")
        rich = await c.get("/api/health")
    assert plain.json()["ui"] == "spa"
    payload = rich.json()
    assert payload["counts"]["bricks"] > 0
    assert "proven" in payload["counts"]
    assert payload["pool"]["cap"] > 0
    assert "canvas" in payload


@pytest.mark.asyncio
async def test_plan_requires_an_intent():
    async with client() as c:
        resp = await c.post("/api/plan", json={"industry": "art"})
    assert resp.status_code == 400
    assert "intent" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_validate_and_heal_need_a_spec():
    async with client() as c:
        for path in ("/api/validate", "/api/heal", "/api/check", "/api/advise"):
            resp = await c.post(path, json={})
            assert resp.status_code == 400, path


@pytest.mark.asyncio
async def test_validate_returns_findings_for_a_real_spec():
    from tests.conftest import concept_spec

    async with client() as c:
        resp = await c.post("/api/validate",
                            json={"spec": concept_spec().to_dict(), "live_validation": False})
    payload = resp.json()
    assert "issues" in payload and isinstance(payload["issues"], list)


@pytest.mark.asyncio
async def test_bricks_and_leaderboard_are_agent_readable():
    async with client() as c:
        bricks = (await c.get("/api/bricks?limit=3")).json()
        board = (await c.get("/api/leaderboard")).json()
        models = (await c.get("/api/models")).json()
    assert bricks["headers"][0] == "id" and len(bricks["rows"]) <= 3
    assert "rows" in board and "message" in board
    assert models["models"] and models["default"]


@pytest.mark.asyncio
async def test_tokens_are_never_echoed_back():
    secret = "hf_this_must_not_come_back"
    from tests.conftest import concept_spec

    async with client() as c:
        for path, body in (
            ("/api/validate", {"spec": concept_spec().to_dict(), "token": secret,
                               "live_validation": False}),
            ("/api/verify_token", {"token": secret}),
            ("/api/health", {}),
        ):
            resp = await c.post(path, json=body) if path != "/api/health" else await c.get(path)
            assert secret not in resp.text, path


def test_artifact_store_serves_only_registered_files(tmp_path):
    """Artifacts are addressed by opaque id: no path input means no traversal surface."""
    store = api_module.ArtifactStore(limit=3)
    produced = tmp_path / "sprite.png"
    produced.write_bytes(b"\x89PNG\r\n\x1a\n")
    artifact_id = store.register(produced)
    assert artifact_id and store.resolve(artifact_id) == produced

    # unregistered paths are simply unknown
    assert store.resolve("../../etc/passwd") is None
    assert store.resolve("0" * 16) is None

    # the store is bounded, and re-registering the same file keeps one id
    assert store.register(produced) == artifact_id
    for index in range(5):
        extra = tmp_path / f"extra{index}.png"
        extra.write_bytes(b"x")
        store.register(extra)
    assert len(store) <= 3


@pytest.mark.asyncio
async def test_artifact_endpoint_404s_for_an_unknown_id():
    async with client() as c:
        resp = await c.get("/api/artifact/deadbeefdeadbeef")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_artifact_endpoint_serves_a_registered_file(tmp_path):
    produced = tmp_path / "keep.png"
    produced.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    artifact_id = api_module.ARTIFACTS.register(produced)
    assert artifact_id
    async with client() as c:
        resp = await c.get(f"/api/artifact/{artifact_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")


@pytest.mark.asyncio
async def test_run_stream_emits_ndjson_for_a_spec_with_no_network_steps():
    """A workflow of pure local helpers must stream start → steps → done without network."""
    from daggrstudio.spec import Binding, Step, WorkflowSpec

    spec = WorkflowSpec(name="Local only", steps=[
        Step(id="upper", kind="fn", fn="json_report",
             inputs={"payload": Binding(value="hello")},
             outputs={"report": {"component": "json", "label": "Report"}}),
    ])
    async with client() as c:
        async with c.stream("POST", "/api/run/stream",
                            json={"spec": spec.to_dict(), "values": {}}) as resp:
            assert resp.status_code == 200
            assert "ndjson" in resp.headers["content-type"]
            events = [json.loads(line) async for line in resp.aiter_lines() if line.strip()]
    stages = [e["stage"] for e in events]
    assert stages[0] == "started"
    assert "done" in stages
    assert any(s in ("step", "heartbeat") for s in stages)


def test_queued_api_endpoints_never_accept_a_credential():
    """Anything on Gradio's queue can land in run history, so it must not take a token."""
    import inspect

    from daggrstudio.web.studio import QUEUED_ENDPOINTS

    assert QUEUED_ENDPOINTS, "expected queued endpoints to be registered"
    forbidden = ("token", "secret", "password", "api_key", "apikey", "key")
    for name, fn in QUEUED_ENDPOINTS:
        params = set(inspect.signature(fn).parameters)
        assert not (params & set(forbidden)), f"{name} accepts a credential: {params}"
        assert "spec_json" in params or "intent" in params or not params, name
