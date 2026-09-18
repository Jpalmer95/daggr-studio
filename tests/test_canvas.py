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
async def test_health_exposes_the_canvas_error_field():
    """Readiness and the recorded error must both be visible from the outside."""
    import httpx

    from daggrstudio.web.studio import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as client:
        health = (await client.get("/healthz")).json()
        rich = (await client.get("/api/health")).json()
    for payload in (health, rich):
        assert "canvas" in payload
        assert "ready" in payload["canvas"]
        assert "error" in payload["canvas"]

def test_publish_to_canvas_is_the_shared_entry_point():
    """
    Planning, healing, loading and upgrades all route through services.publish_to_canvas, so
    the canvas always shows the workflow the user is looking at. Regression: the SPA-era
    rewrite briefly dropped this, leaving /canvas/ permanently empty.
    """
    from daggrstudio.spec import Binding, Step, WorkflowSpec
    from daggrstudio.web import services

    # a network-free spec (local helper only) so this test never touches the Hub
    spec = WorkflowSpec(name="Local only", steps=[
        Step(id="report", kind="fn", fn="json_report",
             inputs={"payload": Binding(value="hi")},
             outputs={"report": {"component": "json", "label": "Report"}}),
    ])
    status = services.publish_to_canvas(spec)
    assert "canvas shows" in status and "Local only" in status

    from daggrstudio.web.canvas import CANVAS

    assert CANVAS.has_graph is True and CANVAS.error is None


def test_publish_to_canvas_reports_failure_instead_of_raising():
    """A canvas problem must never break the thing that builds workflows."""
    from daggrstudio.web import services

    # None is not a workflow: the helper answers, it does not explode
    assert services.publish_to_canvas(None) == "nothing to show"
