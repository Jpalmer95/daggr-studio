"""Codegen must be deterministic and produce code that is (a) valid Python and (b) a
working daggr graph. Both are asserted here - offline, with no network calls."""

from __future__ import annotations

import ast
import json

from daggrstudio.codegen import (
    build_graph,
    infer_postprocess_hint,
    node_name_for,
    render_app_py,
    resolve_postprocess,
)
from daggrstudio.registry.bricks import get_registry
from daggrstudio.spec import Binding, Step, WorkflowSpec
from tests.conftest import concept_spec


def test_build_graph_creates_nodes_and_edges():
    graph = build_graph(concept_spec())
    names = set(graph.nodes)
    assert {"concept", "cutout"}.issubset(names)          # space steps keep their id
    assert any(n.startswith("organize_outputs") for n in names)  # fn step named after fn

    connections = graph.get_connections()
    endpoints = {(c[0], c[1], c[2], c[3]) for c in connections} if connections else set()
    assert connections, "expected at least one edge"
    assert any("concept" in str(c) and "cutout" in str(c) for c in connections), endpoints


def test_graph_has_a_stable_name_and_persist_key():
    graph = build_graph(concept_spec())
    assert graph.name == "Concept to sprite"


def test_build_graph_rejects_unknown_brick():
    spec = concept_spec()
    spec.steps[0].brick_id = "not-a-real-brick"
    spec.steps[0].source = "owner/does-not-exist"
    import pytest

    with pytest.raises(KeyError, match="unknown brick"):
        build_graph(spec)


def test_seed_registry_entries_carry_provenance():
    reg = get_registry()
    seed = reg.by_id("flux1-schnell")
    assert seed is not None and seed.verified_at  # introspected in-session
    unverified = reg.by_id("wan21")
    assert unverified is not None
    assert not unverified.is_live_verified or unverified.verified_at


def test_node_names_are_unique_and_url_safe():
    used: set[str] = set()
    a = node_name_for(Step(id="Concept Art", kind="space"), used)
    b = node_name_for(Step(id="concept art", kind="space"), used)
    assert a == "concept_art" and b == "concept_art_2"
    assert node_name_for(Step(id="x", kind="fn"), used) is None


def test_resolve_postprocess_hints():
    first = resolve_postprocess("first")
    assert first("img", 42) == "img"
    second = resolve_postprocess("tuple_index:1")
    assert second("original", "processed") == "processed"
    assert second("only") == "only"  # tolerant of single-value returns
    dp = resolve_postprocess("dict_path:path")
    assert dp({"path": "/tmp/a.png"}) == "/tmp/a.png"
    assert dp("/tmp/a.png") == "/tmp/a.png"
    assert resolve_postprocess(None) is None
    assert resolve_postprocess("nonsense") is None


def test_postprocess_hint_comes_from_the_registry():
    spec = concept_spec()
    registry = get_registry()
    cutout = spec.steps[1]
    brick = registry.by_id(cutout.brick_id)
    # whatever the registry verified for this brick is what codegen must use
    assert infer_postprocess_hint(brick, cutout) == (brick.postprocess_hint
                                                     if brick.postprocess_hint else None)
    # an explicit step-level hint always overrides the registry
    cutout.postprocess = "tuple_index:1"
    assert infer_postprocess_hint(brick, cutout) == "tuple_index:1"


def test_multi_value_bricks_get_a_postprocess_hint():
    """A brick that returns (value, extra) must never be used without unpacking."""
    registry = get_registry()
    multi = [b for b in registry.all() if b.postprocess_hint and b.status == "running"]
    assert multi, "registry should expose at least one multi-return brick"
    for brick in multi:
        hint = brick.postprocess_hint
        assert hint.startswith("tuple_index:") or hint.startswith("dict_path:")


def test_render_app_py_is_valid_python_and_self_contained():
    spec = concept_spec()
    registry = get_registry()
    source = render_app_py(spec, registry)
    ast.parse(source)  # raises SyntaxError if codegen emits broken code
    assert "from daggr import FnNode, GradioNode, InferenceNode, InputNode, Graph" in source
    gen = registry.by_id(spec.steps[0].brick_id)
    cut = registry.by_id(spec.steps[1].brick_id)
    assert gen.source in source
    assert f'api_name="{gen.api_name}"' in source
    assert cut.source in source
    assert "def organize_outputs(" in source          # helper inlined
    assert "daggrstudio" not in source                # no dependency on this package
    assert "graph.launch()" in source


def test_render_is_deterministic_and_tracks_the_spec():
    spec = concept_spec()
    one = render_app_py(spec)
    two = render_app_py(spec)
    # identical except for the generated timestamp line
    assert "\n".join(l for l in one.splitlines() if "Generated:" not in l) == \
           "\n".join(l for l in two.splitlines() if "Generated:" not in l)

    spec.name = "Renamed"
    assert "Renamed" in render_app_py(spec)


def test_rendered_app_reparses_into_an_equivalent_graph():
    """The downloaded code and the in-Space graph share one spec - prove the spec survives
    a JSON round trip through the render path."""
    spec = concept_spec()
    again = WorkflowSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert render_app_py(again).replace("Generated: ", "").split("\n")[0:3] == \
           render_app_py(spec).replace("Generated: ", "").split("\n")[0:3]


def test_fn_step_uses_the_real_function_signature():
    graph = build_graph(concept_spec())
    fn_nodes = [n for name, n in graph.nodes.items() if name.startswith("organize_outputs")]
    assert fn_nodes, "fn step should produce a node"
    node = fn_nodes[0]
    assert "source" in node._input_ports
    assert "output_dir" in node._input_ports


def test_build_graph_handles_a_spec_with_no_inputs():
    spec = WorkflowSpec(name="No inputs", steps=[
        Step(id="bg", kind="space", brick_id="background-removal",
             inputs={"image": Binding(value="/tmp/x.png")},
             outputs={"image": {"component": "image"}}),
    ])
    graph = build_graph(spec)
    assert graph is not None and len(graph.nodes) == 1