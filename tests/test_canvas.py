"""Canvas state honesty: readiness and status must never contradict each other.

Bug this file exists for: a failed canvas build used to leave the *previous* graph mounted,
so /healthz reported canvas_ready=true while the canvas showed something else. Readiness must
never be more optimistic than reality.
"""

from __future__ import annotations

import pytest

from daggrstudio.web.canvas import CanvasRouter
from tests.conftest import concept_spec


def test_a_fresh_router_reports_no_workflow_and_not_ready():
    router = CanvasRouter()
    assert router.has_graph is False
    assert router.error is None
    assert "no workflow" in router.status


def test_a_successful_build_reports_ready_and_no_error():
    from daggrstudio.codegen import build_graph
    from daggrstudio.registry.bricks import get_registry

    router = CanvasRouter()
    spec = concept_spec()
    status = router.set_graph(build_graph(spec, get_registry()), spec)
    assert router.has_graph is True
    assert router.error is None
    assert "canvas shows" in status


def test_a_failed_build_takes_the_stale_graph_down():
    from daggrstudio.codegen import build_graph
    from daggrstudio.registry.bricks import get_registry

    router = CanvasRouter()
    router.set_graph(build_graph(concept_spec(), get_registry()), concept_spec())
    assert router.has_graph is True

    router.set_spec_error("could not build the canvas: KeyError: bad step")
    assert router.has_graph is False          # the stale graph must NOT linger
    assert router.error is not None
    assert "KeyError" in router.error


def test_show_spec_reports_a_broken_spec_instead_of_raising():
    """A spec with an unknown helper must produce a readable status, not a traceback."""
    from daggrstudio.web.canvas import show_spec

    broken = concept_spec()
    broken.steps[2].fn = None
    status = show_spec(broken)
    assert "could not build the canvas" in status
    assert "unknown fn" in status


def test_show_spec_clears_the_canvas_for_none():
    from daggrstudio.web.canvas import show_spec

    assert show_spec(None) == "canvas cleared"


@pytest.mark.asyncio
async def test_health_and_canvas_endpoints_expose_the_error_field():
    import httpx

    import app as studio

    transport = httpx.ASGITransport(app=studio.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = (await client.get("/healthz")).json()
        canvas = (await client.get("/api/canvas")).json()
    assert "canvas_error" in health
    assert "error" in canvas
    assert canvas["ready"] in (True, False)