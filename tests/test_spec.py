"""Spec is the contract between the LLM, the codegen, and the shared Dataset."""

from __future__ import annotations

import json

from daggrstudio.spec import (
    Binding,
    InputPort,
    Step,
    WorkflowSpec,
    spec_from_json,
    slugify,
)
from tests.conftest import concept_spec


def test_round_trip_preserves_everything():
    spec = concept_spec()
    restored = WorkflowSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.to_dict() == spec.to_dict()
    assert restored.steps[0].inputs["seed"].callable == "random_int"
    from tests.conftest import file_param
    cut = __import__("daggrstudio.registry.bricks", fromlist=["get_registry"]).get_registry().by_id(spec.steps[1].brick_id)
    assert restored.steps[1].inputs[file_param(cut)].step_ref == ("concept", "image")
    assert restored.inputs[0].port == "prompt"


def test_binding_kinds_are_classified():
    assert Binding(source="user:prompt").kind == "user"
    assert Binding(source="user:prompt").user_port == "prompt"
    assert Binding(source="step:a.b").kind == "step"
    assert Binding(source="step:a.b").step_ref == ("a", "b")
    assert Binding(value=7).kind == "value"
    assert Binding(callable="random_int").kind == "callable"


def test_edges_and_used_inputs():
    from daggrstudio.registry.bricks import get_registry
    from tests.conftest import file_param

    spec = concept_spec()
    cut = get_registry().by_id(spec.steps[1].brick_id)
    assert spec.edges == [("concept", "image", "cutout", file_param(cut)),
                          ("cutout", "image", "collect", "source")]
    assert "prompt" in spec.inputs_used()


def test_healthy_spec_has_no_shape_problems():
    assert concept_spec().validate_shape() == []


def _codes(spec: WorkflowSpec) -> str:
    return " | ".join(spec.validate_shape())


def test_detects_missing_steps_and_duplicate_ids():
    empty = WorkflowSpec(name="empty")
    assert "no steps" in _codes(empty)

    spec = concept_spec()
    spec.steps[1].id = "concept"
    assert "duplicate step id" in _codes(spec)


def test_detects_undeclared_user_input():
    spec = concept_spec()
    spec.steps[0].inputs["prompt"] = Binding(source="user:missing_port")
    assert "undeclared input" in _codes(spec)


def test_detects_malformed_edge_and_unknown_step():
    spec = concept_spec()
    spec.steps[1].inputs["image"] = Binding(source="step:concept")
    assert "malformed edge" in _codes(spec)

    spec = concept_spec()
    spec.steps[1].inputs["image"] = Binding(source="step:ghost.image")
    assert "unknown step 'ghost'" in _codes(spec)


def test_detects_reading_a_port_that_does_not_exist():
    spec = concept_spec()
    spec.steps[1].inputs["image"] = Binding(source="step:concept.mask")
    assert "only exposes" in _codes(spec)


def test_detects_unknown_callable_component_and_fn():
    spec = concept_spec()
    spec.steps[0].inputs["seed"] = Binding(callable="run_arbitrary_code")
    assert "unknown callable" in _codes(spec)

    spec = concept_spec()
    spec.inputs[1].component = "wiggle"
    assert "unknown component" in _codes(spec)

    spec = concept_spec()
    spec.steps[2].fn = "os.system"
    assert "unknown fn" in _codes(spec)


def test_detects_cycles():
    spec = concept_spec()
    spec.steps[0].inputs["image"] = Binding(source="step:cutout.image")
    assert "cycle detected" in _codes(spec)


def test_detects_step_without_outputs_or_source():
    spec = concept_spec()
    spec.steps[0].outputs = {}
    assert "declares no output ports" in _codes(spec)

    spec = concept_spec()
    spec.steps[0].brick_id = None
    spec.steps[0].source = None
    assert "neither brick_id nor source" in _codes(spec)


def test_tolerates_loose_json_shapes():
    loose = {
        "name": "Loose",
        "inputs": {"prompt": {"component": "textbox"}},
        "steps": [
            {"id": "a", "source": "owner/space", "api_name": "/run",
             "inputs": {"text": {"from": "user:prompt"}},
             "outputs": {"out": "textbox"}},
        ],
    }
    spec = WorkflowSpec.from_dict(loose)
    assert spec.inputs[0].port == "prompt"
    assert spec.steps[0].outputs["out"]["component"] == "textbox"
    assert spec.validate_shape() == []


def test_spec_from_json_handles_fences_and_prose():
    raw = (
        "Sure! Here is the workflow:\n```json\n"
        + json.dumps(concept_spec().to_dict())
        + "\n```\nLet me know if you want changes."
    )
    spec = spec_from_json(raw)
    assert spec is not None and spec.name == "Concept to sprite"


def test_spec_from_json_rejects_non_specs():
    assert spec_from_json("no json here") is None
    assert spec_from_json('{"hello": "world"}') is None


def test_slugify_is_stable_and_safe():
    assert slugify("Concept → Sprite / v2!") == "concept-sprite-v2"
    assert slugify("") == "workflow"


def test_step_clone_is_deep():
    spec = concept_spec()
    clone = spec.steps[0].clone()
    clone.inputs["prompt"] = Binding(value="changed")
    assert spec.steps[0].inputs["prompt"].kind == "user"


def test_input_port_dict_omits_empty_optionals():
    port = InputPort(port="prompt", component="textbox", label="Prompt", default="hi")
    d = port.to_dict()
    assert "minimum" not in d and "choices" not in d
    assert d["lines"] == 3


def test_step_accepts_plain_string_outputs():
    step = Step.from_dict({"id": "x", "kind": "fn", "fn": "json_report",
                           "inputs": {}, "outputs": {"report": "json"}})
    assert step.outputs["report"]["component"] == "json"