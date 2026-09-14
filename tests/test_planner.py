"""Planner: the model may only pick catalogue bricks, and we complete what the registry knows."""

from __future__ import annotations

import json

from daggrstudio.planner import (
    build_catalogue,
    complete_spec,
    modalities_for_intent,
    plan,
)
from daggrstudio.registry.bricks import get_registry
from daggrstudio.spec import WorkflowSpec


class StubClient:
    model = "stub-planner"

    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def ask(self, prompt, system=None, **kwargs):
        self.prompts.append(prompt)
        self.systems.append(system or "")
        if isinstance(self.payload, str):
            return self.payload, self.model
        return json.dumps(self.payload), self.model


GOOD_PLAN = {
    "name": "Sprite with voiceover",
    "intent": "sprite + narration",
    "industry": "game-dev",
    "tags": ["game-dev"],
    "compute_tier": "cloud-free",
    "inputs": [{"port": "prompt", "component": "textbox", "label": "Idea",
                "default": "a slime knight"}],
    "steps": [
        {"id": "art", "kind": "space", "brick_id": "flux1-schnell", "title": "Art",
         "inputs": {"prompt": {"from": "user:prompt"},
                    "width": {"value": 512},
                    "bogus_param": {"value": 1}},
         "outputs": {"image": {"component": "image", "label": "Art"}}},
        {"id": "cut", "kind": "space", "brick_id": "background-removal", "title": "Cut out",
         "inputs": {"image": {"from": "step:art.image"}}, "outputs": {}},
        {"id": "voice", "kind": "space", "brick_id": "edge-tts", "title": "Narrate",
         "inputs": {"text": {"from": "user:narration"}}, "outputs": {}},
    ],
}


def test_intent_keywords_pick_relevant_modalities():
    assert "image-to-3d" in modalities_for_intent("make a 3D model of my character")
    assert "audio-tts" in modalities_for_intent("narrate this blog post")
    assert "music" in modalities_for_intent("generate a jingle for my shop")
    assert modalities_for_intent("") == ["image-gen", "utility"]


def test_catalogue_only_contains_allowed_and_relevant_bricks():
    registry = get_registry()
    catalogue = build_catalogue("make a transparent game sprite", registry)
    ids = [b.id for b in catalogue]
    assert "flux1-schnell" in ids or "flux1-dev" in ids
    assert "background-removal" in ids
    assert catalogue  # never empty
    # commercial-only by default: no NC brick may leak into the model's options
    assert all(b.commercial for b in catalogue)
    # a video brick is not relevant to a sprite task
    assert "wan21" not in ids or registry.by_id("wan21").status != "running"


def test_catalogue_includes_nc_bricks_when_the_posture_allows_it():
    registry = get_registry()
    permissive = build_catalogue("make a game sprite", registry, commercial_only=False)
    assert len(permissive) >= len(build_catalogue("make a game sprite", registry))


def test_plan_completes_endpoints_outputs_and_postprocess_from_the_registry():
    result = plan("make a sprite with a voiceover", client=StubClient(GOOD_PLAN),
                  industry="game-dev", max_steps=5)
    assert result.ok, result.error
    spec = result.spec
    assert spec is not None
    assert spec.steps[0].api_name == "/infer"            # filled from the registry
    assert spec.steps[1].outputs                          # derived from the brick
    assert spec.steps[1].postprocess == "tuple_index:1"   # bg-removal returns 2 values
    assert spec.planner_model == "stub-planner"


def test_plan_drops_parameters_the_brick_does_not_have():
    result = plan("make a sprite", client=StubClient(GOOD_PLAN), max_steps=5)
    assert "bogus_param" not in result.spec.steps[0].inputs
    assert any("dropped unknown parameter" in n for n in result.notes)


def test_plan_declares_user_inputs_the_model_forgot():
    result = plan("make a sprite", client=StubClient(GOOD_PLAN), max_steps=5)
    ports = [i.port for i in result.spec.inputs]
    assert "prompt" in ports and "narration" in ports   # narration was referenced, not declared
    assert any("declared missing input 'narration'" in n for n in result.notes)


def test_plan_trims_runaway_step_lists():
    payload = dict(GOOD_PLAN)
    payload["steps"] = GOOD_PLAN["steps"] * 3  # 9 steps, duplicate ids
    result = plan("make a sprite", client=StubClient(payload), max_steps=4)
    assert len(result.spec.steps) <= 4
    assert any("trimmed" in n for n in result.notes)


def test_plan_flags_bricks_that_do_not_exist():
    payload = json.loads(json.dumps(GOOD_PLAN))
    payload["steps"][0]["brick_id"] = "imaginary-space"
    result = plan("make a sprite", client=StubClient(payload), max_steps=5)
    assert any("do not exist" in n for n in result.notes)


def test_plan_reports_a_missing_model_instead_of_failing():
    result = plan("anything", client=None)
    assert not result.ok and "token" in result.error


def test_plan_reports_unusable_model_output():
    result = plan("anything", client=StubClient("I'm sorry, I can't help with that."))
    assert not result.ok and "usable workflow spec" in result.error


def test_plan_coerces_an_invalid_licence_posture():
    payload = json.loads(json.dumps(GOOD_PLAN))
    payload["license_posture"] = "whatever-i-want"
    result = plan("make a sprite", client=StubClient(payload), max_steps=5,
                  license_posture="commercial-only")
    assert result.spec.license_posture == "commercial-only"


def test_the_prompt_carries_the_catalogue_and_the_rules():
    client = StubClient(GOOD_PLAN)
    plan("make a sprite with a voiceover", client=client, max_steps=5)
    prompt, system = client.prompts[0], client.systems[0]
    assert "BRICK CATALOGUE" in prompt
    assert "flux1-schnell" in prompt
    assert "LOCAL HELPERS" in prompt
    assert "ONLY use bricks from it" in system


def test_complete_spec_is_idempotent():
    registry = get_registry()
    result = plan("make a sprite", client=StubClient(GOOD_PLAN), max_steps=5)
    once = result.spec.to_dict()
    twice = complete_spec(result.spec.clone(), registry).to_dict()
    assert once["steps"] == twice["steps"]


def test_complete_spec_handles_a_spec_with_no_inputs_at_all():
    spec = WorkflowSpec(name="Bare", intent="make an image", steps=[])
    completed = complete_spec(spec, get_registry())
    assert [i.port for i in completed.inputs] == ["prompt"]