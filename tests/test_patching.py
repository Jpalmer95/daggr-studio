"""The patch language: what the Medic is allowed to do, and what must be rejected."""

from __future__ import annotations

from daggrstudio.medic.patching import apply_ops, normalise_ops
from daggrstudio.registry.bricks import BrickRegistry, get_registry
from daggrstudio.spec import Binding
from tests.conftest import concept_spec


def _apply(spec, *ops, registry: BrickRegistry | None = None):
    return apply_ops(spec, list(ops), registry or get_registry())


def test_normalise_ops_accepts_every_shape_the_model_might_return():
    assert normalise_ops({"ops": [{"op": "set_meta", "notes": "x"}]}) == [{"op": "set_meta", "notes": "x"}]
    assert normalise_ops({"patches": [{"op": "drop_input"}]}) == [{"op": "drop_input"}]
    assert normalise_ops([{"op": "drop_input"}]) == [{"op": "drop_input"}]
    assert normalise_ops({"op": "drop_input", "step": "a"}) == [{"op": "drop_input", "step": "a"}]
    assert normalise_ops(None) == []
    assert normalise_ops({"diagnosis": "nothing wrong"}) == []
    assert normalise_ops("a string") == []


def test_set_input_wires_a_user_binding():
    spec = concept_spec()
    result = _apply(spec, {"op": "set_input", "step": "concept", "param": "prompt",
                           "from": "user:prompt"})
    assert result.changed and not result.skipped
    assert spec.steps[0].inputs["prompt"].kind == "user"


def test_set_input_accepts_a_literal_via_value_key():
    spec = concept_spec()
    _apply(spec, {"op": "set_input", "step": "concept", "param": "height", "value": 768})
    assert spec.steps[0].inputs["height"].value == 768


def test_drop_input_removes_a_param_the_space_no_longer_takes():
    spec = concept_spec()
    result = _apply(spec, {"op": "drop_input", "step": "concept", "param": "num_inference_steps"})
    assert result.changed
    assert "num_inference_steps" not in spec.steps[0].inputs


def test_unknown_op_is_rejected_with_a_reason():
    spec = concept_spec()
    result = _apply(spec, {"op": "run_arbitrary_python", "code": "import os"})
    assert not result.changed
    assert "unknown op" in result.skipped[0]["reason"]


def test_edge_to_a_nonexistent_step_is_rejected():
    spec = concept_spec()
    result = _apply(spec, {"op": "set_input", "step": "cutout", "param": "image",
                           "from": "step:ghost.image"})
    assert not result.changed
    assert "unknown step 'ghost'" in result.skipped[0]["reason"]


def test_edge_to_a_port_that_does_not_exist_is_rejected():
    spec = concept_spec()
    result = _apply(spec, {"op": "set_input", "step": "cutout", "param": "image",
                           "from": "step:concept.mask"})
    assert not result.changed
    assert "no output port 'mask'" in result.skipped[0]["reason"]


def test_binding_to_an_undeclared_user_input_is_rejected():
    spec = concept_spec()
    result = _apply(spec, {"op": "set_input", "step": "concept", "param": "prompt",
                           "from": "user:not_declared"})
    assert not result.changed
    assert "not declared" in result.skipped[0]["reason"]


def test_replace_brick_only_accepts_verified_bricks():
    spec = concept_spec()
    bad = _apply(spec, {"op": "replace_brick", "step": "concept", "brick_id": "imaginary-space"})
    assert not bad.changed
    assert "not a verified brick" in bad.skipped[0]["reason"]

    good = _apply(spec, {"op": "replace_brick", "step": "concept", "brick_id": "z-image-turbo"})
    assert good.changed
    assert spec.steps[0].brick_id == "z-image-turbo"
    assert spec.steps[0].source == "hf-applications/Z-Image-Turbo"


def test_replace_brick_preserves_edges_and_drops_stale_literals():
    spec = concept_spec()
    _apply(spec, {"op": "replace_brick", "step": "cutout", "brick_id": "not-lain-bg-removal"})
    # the edge coming from the upstream step must survive the swap
    assert spec.steps[1].inputs["image"].step_ref == ("concept", "image")


def test_set_outputs_validates_components():
    spec = concept_spec()
    bad = _apply(spec, {"op": "set_outputs", "step": "concept",
                        "outputs": {"image": {"component": "hologram"}}})
    assert not bad.changed and "unknown component" in bad.skipped[0]["reason"]

    good = _apply(spec, {"op": "set_outputs", "step": "concept",
                         "outputs": {"image": {"component": "image", "label": "Art"}}})
    assert good.changed
    assert spec.steps[0].outputs["image"]["label"] == "Art"


def test_set_postprocess_rejects_invented_hints():
    spec = concept_spec()
    bad = _apply(spec, {"op": "set_postprocess", "step": "cutout", "hint": "take_the_last_one"})
    assert not bad.changed
    assert "unrecognised postprocess hint" in bad.skipped[0]["reason"]

    good = _apply(spec, {"op": "set_postprocess", "step": "cutout", "hint": "tuple_index:1"})
    assert good.changed and spec.steps[1].postprocess == "tuple_index:1"


def test_remove_step_refuses_to_break_edges():
    spec = concept_spec()
    result = _apply(spec, {"op": "remove_step", "step": "concept"})
    assert not result.changed
    assert "still referenced" in result.skipped[0]["reason"]

    ok = _apply(spec, {"op": "remove_step", "step": "collect"})
    assert ok.changed and len(spec.steps) == 2


def test_add_step_requires_a_verified_brick_and_a_fresh_id():
    spec = concept_spec()
    dup = _apply(spec, {"op": "add_step", "step": {"id": "concept", "kind": "fn", "fn": "json_report"}})
    assert not dup.changed and "already exists" in dup.skipped[0]["reason"]

    invented = _apply(spec, {"op": "add_step",
                             "step": {"id": "x", "kind": "space", "brick_id": "made-up"}})
    assert not invented.changed and "not verified" in invented.skipped[0]["reason"]

    good = _apply(spec, {"op": "add_step",
                         "step": {"id": "report", "kind": "fn", "fn": "json_report",
                                  "inputs": {"payload": {"from": "step:cutout.image"}},
                                  "outputs": {"report": {"component": "json"}}}})
    assert good.changed and spec.step("report") is not None


def test_set_meta_enforces_field_types_and_posture_values():
    spec = concept_spec()
    bad = _apply(spec, {"op": "set_meta", "tags": "game-dev"})
    assert not bad.changed and "must be list" in bad.skipped[0]["reason"]

    bad2 = _apply(spec, {"op": "set_meta", "license_posture": "whatever"})
    assert not bad2.changed

    good = _apply(spec, {"op": "set_meta", "name": "Better name", "license_posture": "any",
                         "tags": ["art"]})
    assert good.changed and spec.name == "Better name" and spec.license_posture == "any"


def test_declare_inputs_replaces_the_input_list_safely():
    spec = concept_spec()
    bad = _apply(spec, {"op": "declare_inputs",
                        "inputs": [{"port": "x", "component": "teleporter"}]})
    assert not bad.changed and "unknown component" in bad.skipped[0]["reason"]

    good = _apply(spec, {"op": "declare_inputs",
                         "inputs": [{"port": "prompt", "component": "textbox", "label": "Prompt"}]})
    assert good.changed and [i.port for i in spec.inputs] == ["prompt"]


def test_a_batch_with_one_bad_op_still_applies_the_good_ones():
    spec = concept_spec()
    result = _apply(spec,
                    {"op": "set_meta", "name": "Mixed batch"},
                    {"op": "drop_input", "step": "ghost", "param": "x"},
                    {"op": "set_input", "step": "concept", "param": "height", "value": 512})
    assert len(result.applied) == 2 and len(result.skipped) == 1
    assert spec.name == "Mixed batch"
    assert spec.steps[0].inputs["height"].value == 512


def test_spec_timestamp_moves_only_when_something_changed():
    spec = concept_spec()
    before = spec.updated_at
    _apply(spec, {"op": "drop_input", "step": "ghost", "param": "nope"})
    assert spec.updated_at == before
    _apply(spec, {"op": "set_meta", "notes": "touched"})
    assert spec.updated_at >= before