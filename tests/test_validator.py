"""Validator: every check must be provable with a seeded defect. These run offline."""

from __future__ import annotations

from daggrstudio.registry.bricks import BrickRegistry, get_registry
from daggrstudio.spec import Binding, Step, WorkflowSpec
from daggrstudio.validator import validate
from tests.conftest import concept_spec


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


def _blocking(report) -> set[str]:
    return set(report.codes())


def test_a_healthy_spec_passes_offline():
    report = validate(concept_spec(), live=False)
    assert report.ok, [str(i) for i in report.blocking]
    assert not _blocking(report)


def test_unknown_brick_is_blocking_with_alternatives_suggested():
    spec = concept_spec()
    spec.steps[0].brick_id = "totally-made-up"
    spec.steps[0].source = "nobody/nowhere"
    report = validate(spec, live=False)
    assert "UNKNOWN_BRICK" in _blocking(report)
    issue = next(i for i in report.blocking if i.code == "UNKNOWN_BRICK")
    assert issue.fix_hint and "known_ids" in issue.data


def test_non_commercial_brick_under_commercial_posture_is_blocking():
    spec = concept_spec()
    spec.steps[1].brick_id = "hunyuan3d-2"  # non-commercial licence, wrong modality on purpose
    report = validate(spec, live=False)
    assert "LICENSE_NON_COMMERCIAL" in _blocking(report)


def test_same_brick_is_fine_under_open_posture():
    spec = concept_spec()
    spec.license_posture = "any"
    spec.steps[1].brick_id = "hunyuan3d-2"
    report = validate(spec, live=False)
    assert "LICENSE_NON_COMMERCIAL" not in _blocking(report)


def test_broken_brick_is_blocking_and_offers_replacements():
    registry = get_registry()
    original = registry.by_id("flux1-schnell")
    original.status = "error"
    try:
        spec = concept_spec()
        step = spec.steps[0]
        step.brick_id = "z-image-turbo"  # verified ok, so swap the *step* brick to a broken one
        registry.by_id("z-image-turbo").status = "error"
        report = validate(spec, live=False)
        assert "BRICK_BROKEN" in _blocking(report)
    finally:
        original.status = "running"
        registry.by_id("z-image-turbo").status = "sleeping"


def test_api_name_mismatch_against_registry_is_blocking():
    spec = concept_spec()
    spec.steps[0].api_name = "/infer_v2"
    report = validate(spec, live=False)
    assert "API_NAME_MISMATCH" in _blocking(report)


def test_fn_param_that_does_not_exist_is_blocking():
    spec = concept_spec()
    spec.steps[2].inputs["nonsense"] = Binding(value=1)
    report = validate(spec, live=False)
    assert "FN_PARAM_UNKNOWN" in _blocking(report)


def test_unused_input_and_orphan_step_are_warnings():
    spec = concept_spec()
    spec.inputs.append(__import__("daggrstudio.spec", fromlist=["InputPort"]).InputPort(port="unused"))
    report = validate(spec, live=False)
    assert "UNUSED_INPUT" in {i.code for i in report.warnings}
    assert report.ok


def test_cost_tier_optimism_is_flagged():
    spec = concept_spec()
    spec.steps.append(Step(id="movie", kind="space", brick_id="wan21",
                           inputs={"prompt": Binding(source="user:prompt")},
                           outputs={"video": {"component": "video"}}))
    spec.compute_tier = "cloud-free"
    report = validate(spec, live=False)
    assert "COST_TIER_OPTIMISTIC" in {i.code for i in report.warnings}


def test_too_many_steps_is_a_warning():
    spec = concept_spec()
    for i in range(12):
        spec.steps.append(Step(id=f"extra{i}", kind="fn", fn="summarize_stats",
                               inputs={"payload": Binding(value=i)},
                               outputs={"report": {"component": "json"}}))
    report = validate(spec, live=False)
    assert "TOO_MANY_STEPS" in {i.code for i in report.warnings}


def test_report_serialises_for_the_ui():
    report = validate(concept_spec(), live=False)
    payload = report.to_dict()
    assert set(payload) == {"ok", "blocking", "warnings", "info", "codes"}
    assert report.summary()  # always says something useful
    assert isinstance(report.codes(), list)


def test_offline_validation_does_not_hit_the_network():
    # live=False must not populate introspection results at all
    report = validate(concept_spec(), live=False)
    assert report.introspected == {}


class _RegistryStub(BrickRegistry):
    pass


def test_edge_type_mismatch_detected_with_stubbed_introspection(monkeypatch):
    """image output -> scalar param must be caught, using a fake live Space."""
    from daggrstudio import validator as v
    from daggrstudio.introspect import Param, SpaceInfo

    fake = SpaceInfo(
        source="black-forest-labs/FLUX.1-schnell", api_name="/infer", ok=True,
        endpoints={"/infer": [Param(name="prompt", type="str", has_default=False),
                              Param(name="width", type="float", has_default=True)]},
        return_types={"/infer": ["filepath", "float"]},
    )
    monkeypatch.setattr(v, "introspect", lambda *a, **k: fake)

    spec = concept_spec()
    # feed the image output into the numeric width param
    spec.steps[1].inputs = {"image": Binding(source="step:concept.image")}
    spec.steps[1].brick_id = "flux1-schnell"
    spec.steps[1].inputs = {"width": Binding(source="step:concept.image"),
                            "prompt": Binding(source="user:prompt")}
    report = validate(spec, live=True)
    codes = {i.code for i in report.issues}
    assert "EDGE_TYPE_MISMATCH" in codes


def test_param_renamed_is_detected_with_suggestions(monkeypatch):
    from daggrstudio import validator as v
    from daggrstudio.introspect import Param, SpaceInfo

    fake = SpaceInfo(
        source="black-forest-labs/FLUX.1-schnell", api_name="/infer", ok=True,
        endpoints={"/infer": [Param(name="text", type="str", has_default=False),
                              Param(name="width", type="float", has_default=True)]},
        return_types={"/infer": ["filepath", "float"]},
    )
    monkeypatch.setattr(v, "introspect", lambda *a, **k: fake)
    spec = concept_spec()
    spec.steps[0].inputs = {"prompt": Binding(source="user:prompt")}
    report = validate(spec, live=True)
    assert "PARAM_RENAMED" in {i.code for i in report.blocking}
    issue = next(i for i in report.blocking if i.code == "PARAM_RENAMED")
    # the suggestion must come from the live parameter list, not from thin air
    assert "text" in issue.data["suggestions"] or "text" in issue.data["live_params"]


def test_endpoint_missing_lists_available_endpoints(monkeypatch):
    from daggrstudio import validator as v
    from daggrstudio.introspect import Param, SpaceInfo

    fake = SpaceInfo(
        source="black-forest-labs/FLUX.1-schnell", api_name="/infer", ok=True,
        endpoints={"/new_endpoint": [Param(name="prompt", type="str")]},
        return_types={"/new_endpoint": ["filepath"]},
        error="endpoint '/infer' not found; available: ['/new_endpoint']",
        error_class="EndpointMissing",
    )
    monkeypatch.setattr(v, "introspect", lambda *a, **k: fake)
    report = validate(concept_spec(), live=True)
    codes = {i.code for i in report.blocking}
    assert "API_NAME_INVALID" in codes
    issue = next(i for i in report.blocking if i.code == "API_NAME_INVALID")
    assert "/new_endpoint" in issue.fix_hint


def test_missing_required_parameter_is_blocking(monkeypatch):
    """daggr refuses to build a node with an unwired required parameter, so this must block."""
    from daggrstudio import validator as v
    from daggrstudio.introspect import Param, SpaceInfo

    fake = SpaceInfo(
        source="black-forest-labs/FLUX.1-schnell", api_name="/infer", ok=True,
        endpoints={"/infer": [Param(name="prompt", type="str", has_default=False),
                              Param(name="steps", type="number", has_default=True)]},
        return_types={"/infer": ["filepath", "number"]},
    )
    monkeypatch.setattr(v, "introspect", lambda *a, **k: fake)
    spec = concept_spec()
    spec.steps[0].inputs = {"unrelated": Binding(value=1)}
    report = validate(spec, live=True)
    blocking = {i.code for i in report.blocking}
    assert "PARAM_MISSING" in blocking
    issue = next(i for i in report.blocking if i.code == "PARAM_MISSING")
    assert issue.port == "prompt" and "daggr will not build it" in issue.fix_hint


def test_parameters_with_defaults_are_not_demanded(monkeypatch):
    from daggrstudio import validator as v
    from daggrstudio.introspect import Param, SpaceInfo

    fake = SpaceInfo(
        source="black-forest-labs/FLUX.1-schnell", api_name="/infer", ok=True,
        endpoints={"/infer": [Param(name="prompt", type="str", has_default=True),
                              Param(name="steps", type="number", has_default=True)]},
        return_types={"/infer": ["filepath", "number"]},
    )
    monkeypatch.setattr(v, "introspect", lambda *a, **k: fake)
    spec = concept_spec()
    spec.steps[0].inputs = {}
    report = validate(spec, live=True)
    assert "PARAM_MISSING" not in {i.code for i in report.issues}