"""The composed app: routes must not collide, and the agent API must degrade safely.

These run against the real ASGI app in-process (no ports), which is exactly how the Space
serves it in production.
"""

from __future__ import annotations

import json

import httpx
import pytest

import app as studio
from tests.conftest import concept_spec


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=studio.app),
                             base_url="http://test")


@pytest.mark.asyncio
async def test_health_reports_the_registry_and_canvas_state():
    async with client() as c:
        resp = await c.get("/healthz")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert "bricks" in payload["registry"]
    assert payload["canvas_ready"] in (True, False)


@pytest.mark.asyncio
async def test_builder_and_canvas_both_serve_html_from_one_process():
    """The whole reason for the ASGI shim: daggr's absolute paths vs a mounted Builder."""
    async with client() as c:
        builder = await c.get("/builder")
        canvas = await c.get("/")
    assert builder.status_code == 200 and "<!DOCTYPE html>" in builder.text
    assert canvas.status_code == 200 and "<!DOCTYPE html>" in canvas.text
    # the canvas page always offers a way back to the Builder
    assert "/builder" in canvas.text


@pytest.mark.asyncio
async def test_canvas_api_is_honest_before_a_workflow_exists():
    async with client() as c:
        resp = await c.get("/api/graph")
    assert resp.status_code == 404
    assert "no workflow built yet" in resp.json()["error"]


@pytest.mark.asyncio
async def test_brick_and_leaderboard_endpoints_are_agent_readable():
    async with client() as c:
        bricks = await c.get("/api/bricks?limit=3")
        leaderboard = await c.get("/api/leaderboard")
        canvas = await c.get("/api/canvas")
    assert bricks.status_code == 200
    payload = bricks.json()
    assert payload["headers"][0] == "id" and len(payload["rows"]) <= 3
    assert leaderboard.status_code == 200 and "stats" in leaderboard.json()
    assert "status" in canvas.json()


@pytest.mark.asyncio
async def test_agent_plan_endpoint_requires_an_intent():
    async with client() as c:
        resp = await c.post("/api/plan", json={"industry": "art"})
    assert resp.status_code == 400
    assert "intent" in resp.json()["message"]


@pytest.mark.asyncio
async def test_agent_validate_endpoint_accepts_a_spec_and_reports_findings():
    spec = concept_spec().to_dict()
    async with client() as c:
        resp = await c.post("/api/validate", json={"spec": spec, "live_validation": False})
    payload = resp.json()
    assert resp.status_code in (200, 422)
    assert "issues" in payload and "message" in payload
    assert isinstance(payload["issues"], list)


@pytest.mark.asyncio
async def test_agent_endpoints_never_echo_a_token():
    async with client() as c:
        resp = await c.post("/api/validate",
                            json={"spec": concept_spec().to_dict(), "token": "hf_not_a_real_token",
                                  "live_validation": False})
    assert "hf_not_a_real_token" not in resp.text
    assert "hf_not_a_real_token" not in json.dumps(resp.json())
