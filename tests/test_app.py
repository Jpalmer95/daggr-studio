"""Entrypoint: the Space boots from app.py, which composes the Studio app.

The interesting surface tests live in tests/test_studio.py; this file guards the things that
only matter at process start (import order, port selection, and that the entrypoint really
exposes a servable app).
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


def test_entrypoint_exposes_a_servable_app():
    import app as entry

    assert hasattr(entry, "app")
    assert callable(entry.main)
    # gradio.Server *is* a FastAPI/ASGI app - that is the whole reason we can compose it
    assert callable(entry.app)
    assert hasattr(entry.app, "mount") and hasattr(entry.app, "include_router")


def test_main_reads_the_port_from_the_environment(monkeypatch):
    import app as entry

    captured: dict[str, object] = {}

    class FakeApp:
        def launch(self, **kwargs):
            captured.update(kwargs)
            return ("app", "url", "share")

    monkeypatch.setattr(entry, "app", FakeApp())
    monkeypatch.setenv("PORT", "9999")
    entry.main()
    assert captured["server_port"] == 9999
    assert captured["server_name"] == "0.0.0.0"


def test_main_defaults_to_the_space_port(monkeypatch):
    import app as entry

    captured: dict[str, object] = {}

    class FakeApp:
        def launch(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(entry, "app", FakeApp())
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
    entry.main()
    assert captured["server_port"] == 7860


def test_entrypoint_is_importable_from_a_clean_interpreter_syspath(tmp_path):
    """The Space runs `python app.py` from the repo root; the package must import from there."""
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-c", "import app; print(type(app.app).__name__)"],
        cwd=root, capture_output=True, text=True, timeout=180,
        env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Server" in result.stdout or "FastAPI" in result.stdout


@pytest.mark.asyncio
async def test_the_studio_routes_are_mounted_in_order():
    """Cheap structural guard: the four surfaces exist and do not shadow each other."""
    import httpx

    from daggrstudio.web.studio import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as c:
        assert (await c.get("/")).status_code == 200          # SPA owns the root
        assert (await c.get("/api/health")).status_code == 200  # API under its own prefix
        assert (await c.get("/builder")).status_code == 307     # explicit redirect
        assert (await c.get("/builder/")).status_code == 200    # Gradio behind it
