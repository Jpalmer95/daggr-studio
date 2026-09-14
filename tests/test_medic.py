"""Autofix + heal loop, exercised without a network or a real model."""

from __future__ import annotations

import json

from daggrstudio.llm import LLMClient, PoolExhausted
from daggrstudio.medic import apply_suggestions, heal, upgrade_advisor
from daggrstudio.medic.autofix import run_autofixes
from daggrstudio.registry.bricks import get_registry
from daggrstudio.spec import Binding
from daggrstudio.validator import Issue, Report
from tests.conftest import concept_spec


class StubClient:
    """Stands in for LLMClient: records prompts, replays canned JSON payloads."""

    model = "stub-model"

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def ask_json(self, prompt, system=None, model=None):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0) if self.payloads else {}
        return payload, self.model


# ─── deterministic autofixes ──────────────────────────────────────────────────


def test_autofix_renames_a_param_that_moved_in_the_live_space():
    spec = concept_spec()
    report = Report(issues=[Issue(
        code="PARAM_RENAMED", severity="blocking",
        message="no 'prompt'", step_id="concept", port="prompt",
        data={"live_params": ["text", "width"], "suggestions": ["text"]},
    )])
    fixes = run_autofixes(spec, report, get_registry())
    assert fixes and fixes[0].kind == "param_rename"
    assert "text" in spec.steps[0].inputs and "prompt" not in spec.steps[0].inputs
    # the wiring is preserved, only the parameter name changed
    assert spec.steps[0].inputs["text"].kind == "user"


def test_autofix_does_not_repoint_step_edges_on_a_rename():
    """Repointing an edge silently would change *what data flows where* - never automatic."""
    spec = concept_spec()
    report = Report(issues=[Issue(
        code="PARAM_RENAMED", severity="blocking", message="x", step_id="cutout", port="image",
        data={"live_params": ["img"], "suggestions": ["img"]},
    )])
    fixes = run_autofixes(spec, report, get_registry())
    assert not any(f.kind == "param_rename" for f in fixes)
    from tests.conftest import file_param
    cut = get_registry().by_id(spec.steps[1].brick_id)
    assert file_param(cut) in spec.steps[1].inputs


def test_autofix_fails_over_from_a_dead_brick():
    registry = get_registry()
    dead = registry.by_id("z-image-turbo")
    original_status = dead.status
    dead.status = "error"
    try:
        spec = concept_spec()
        spec.steps[0].brick_id = "z-image-turbo"
        spec.steps[0].source = dead.source
        report = Report(issues=[Issue(
            code="BRICK_BROKEN", severity="blocking",
            message="dead", step_id="concept", data={"modality": "image-gen"},
        )])
        fixes = run_autofixes(spec, report, registry)
        assert fixes and fixes[0].kind == "brick_failover"
        assert spec.steps[0].brick_id != "z-image-turbo"
        replacement = registry.by_id(spec.steps[0].brick_id)
        assert replacement.output_kind == "image" and replacement.status == "running"
        assert spec.steps[0].inputs["prompt"].kind == "user"  # wiring survived
    finally:
        dead.status = original_status


def test_autofix_swaps_a_non_commercial_brick():
    """A non-commercial image brick must be swapped for a commercial one of the same kind."""
    registry = get_registry()
    nc = registry.by_id("sd35-large")
    assert nc is not None and not nc.commercial
    spec = concept_spec()
    spec.license_posture = "commercial-only"
    spec.steps[0].brick_id = "sd35-large"
    spec.steps[0].source = nc.source
    report = Report(issues=[Issue(
        code="LICENSE_NON_COMMERCIAL", severity="blocking", message="nc", step_id="concept",
        data={"alternatives": []},  # force the registry fallback
    )])
    fixes = run_autofixes(spec, report, registry)
    assert any(f.kind == "license_swap" for f in fixes)
    replacement = registry.by_id(spec.steps[0].brick_id)
    assert replacement.commercial and replacement.output_kind == nc.output_kind
    assert spec.steps[0].inputs["prompt"].kind == "user"  # wiring survived the swap


# ─── heal loop ────────────────────────────────────────────────────────────────


def test_heal_returns_healthy_without_calling_a_model():
    client = StubClient()
    result = heal(concept_spec(), client=client, live=False)
    assert result.status == "healthy" and result.healed
    assert client.prompts == []  # a valid spec costs nothing


def test_heal_is_deterministic_only_when_no_client_is_given():
    spec = concept_spec()
    spec.steps[2].inputs["bogus_param"] = Binding(value=1)
    result = heal(spec, client=None, live=False)
    assert result.status == "needs_human"
    assert "no model configured" in result.note
    assert not result.healed


def test_heal_uses_the_model_to_fix_an_unfixable_structurally():
    spec = concept_spec()
    spec.steps[2].inputs["bogus_param"] = Binding(value=1)
    client = StubClient({"diagnosis": "helper has no such parameter",
                         "ops": [{"op": "drop_input", "step": "collect", "param": "bogus_param"}]})
    result = heal(spec, client=client, live=False)
    assert result.status == "healed"
    assert result.model == "stub-model"
    assert result.log and result.log[0].actor == "model"
    assert result.log[0].codes_before == ["FN_PARAM_UNKNOWN"]
    assert result.log[0].codes_after == []
    assert "collect" in result.spec.steps[2].id


def test_heal_prompt_contains_the_catalogue_and_the_issues():
    spec = concept_spec()
    spec.steps[2].inputs["bogus_param"] = Binding(value=1)
    client = StubClient({"ops": [{"op": "drop_input", "step": "collect", "param": "bogus_param"}]})
    heal(spec, client=client, live=False)
    prompt = client.prompts[0]
    assert "BRICK CATALOGUE" in prompt
    assert "FN_PARAM_UNKNOWN" in prompt
    assert "flux1-schnell" in prompt


def test_heal_rejects_hallucinated_repairs_and_says_so():
    spec = concept_spec()
    spec.steps[2].inputs["bogus_param"] = Binding(value=1)
    client = StubClient({"ops": [
        {"op": "replace_brick", "step": "collect", "brick_id": "invented-space"},
        {"op": "set_input", "step": "collect", "param": "bogus_param", "from": "step:ghost.out"},
    ]})
    result = heal(spec, client=client, live=False, max_rounds=2)
    assert result.status == "needs_human"
    assert result.log[0].skipped and all("reason" in s for s in result.log[0].skipped)
    # the rejection reasons are fed back to the model on the next round
    assert "REJECTED" in client.prompts[1]


def test_heal_stops_when_rounds_stop_converging():
    spec = concept_spec()
    spec.steps[2].inputs["bogus_param"] = Binding(value=1)
    client = StubClient(
        {"ops": [{"op": "set_meta", "notes": "attempt 1"}]},
        {"ops": [{"op": "set_meta", "notes": "attempt 2"}]},
        {"ops": [{"op": "set_meta", "notes": "attempt 3"}]},
    )
    result = heal(spec, client=client, live=False, max_rounds=4)
    assert result.status == "needs_human"
    assert "converg" in result.note
    assert len(result.log) == 2  # round 2 repeated round 1's diagnosis, so it stopped


def test_heal_survives_a_model_outage():
    class Broken(StubClient):
        def ask_json(self, prompt, system=None, model=None):
            from daggrstudio.llm import LLMError

            raise LLMError("every model in the fallback chain failed")

    spec = concept_spec()
    spec.steps[2].inputs["bogus_param"] = Binding(value=1)
    result = heal(spec, client=Broken(), live=False)
    assert result.status == "model_unavailable"
    assert "fallback chain" in result.note


def test_heal_records_its_timeline_on_the_spec():
    spec = concept_spec()
    spec.steps[2].inputs["bogus_param"] = Binding(value=1)
    client = StubClient({"ops": [{"op": "drop_input", "step": "collect", "param": "bogus_param"}]})
    result = heal(spec, client=client, live=False)
    assert result.spec.heal_log  # persisted with the workflow when saved/shared
    assert "round 1" in result.timeline()[0]


def test_heal_never_mutates_the_callers_spec():
    spec = concept_spec()
    spec.steps[2].inputs["bogus_param"] = Binding(value=1)
    before = json.dumps(spec.to_dict(), sort_keys=True)
    heal(spec, client=StubClient({"ops": [{"op": "drop_input", "step": "collect",
                                           "param": "bogus_param"}]}), live=False)
    assert json.dumps(spec.to_dict(), sort_keys=True) == before


# ─── upgrade advisor ──────────────────────────────────────────────────────────


def test_advisor_heuristic_mode_needs_no_model():
    registry = get_registry()
    for brick in registry.all():
        if brick.id == "flux1-schnell":
            brick.likes = 1
        elif brick.modality == "image-gen" and brick.status == "running":
            brick.likes = 5000
    spec = concept_spec()
    spec.license_posture = "any"
    suggestions = upgrade_advisor(spec, registry=registry, client=None)
    assert all(s["brick_id"] != "flux1-schnell" for s in suggestions)
    assert all(s["source"] == "heuristic" for s in suggestions)


def test_advisor_drops_pairs_the_model_invented():
    client = StubClient({"suggestions": [
        {"step": "concept", "brick_id": "totally-invented", "why": "trust me", "confidence": 0.99},
    ]})
    registry = get_registry()
    registry.by_id("z-image-turbo").status = "running"
    try:
        assert upgrade_advisor(concept_spec(), registry=registry, client=client) == []
    finally:
        registry.by_id("z-image-turbo").status = "sleeping"


def test_advisor_accepts_a_real_pairing_and_it_applies_cleanly():
    """Use whatever alternative the registry actually offers for this step."""
    registry = get_registry()
    spec = concept_spec()
    brick = registry.by_id(spec.steps[0].brick_id)
    alts = [b for b in registry.alternatives(brick, commercial_only=True)
            if b.status == "running" and b.output_kind == brick.output_kind]
    assert alts, "registry should offer an image-gen alternative"
    target = alts[0]

    client = StubClient({"suggestions": [
        {"step": "concept", "brick_id": target.id, "why": "higher quality", "confidence": 0.8},
    ]})
    suggestions = upgrade_advisor(spec, registry=registry, client=client)
    assert suggestions and suggestions[0]["brick_id"] == target.id
    updated, applied = apply_suggestions(spec, suggestions, registry=registry)
    assert applied and updated.steps[0].brick_id == target.id


# ─── cost invariants ──────────────────────────────────────────────────────────


def test_no_token_is_an_actionable_error_not_a_crash(monkeypatch, tmp_path):
    from daggrstudio import llm

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no cached token either
    try:
        llm.LLMClient(token=None).chat([{"role": "user", "content": "hi"}])
        raise AssertionError("expected an LLMError")
    except llm.LLMError as exc:
        assert "token" in str(exc).lower()


def test_the_space_token_is_metered_by_the_community_pool(monkeypatch, tmp_path):
    """The cost invariant: the platform's own token can never spend without a budget."""
    from daggrstudio import llm

    monkeypatch.setenv("HF_TOKEN", "hf_space_owned_token")  # not BYOK -> pool applies
    monkeypatch.setattr(llm, "POOL_PATH", tmp_path / "pool.json")
    monkeypatch.setattr(llm, "POOL_CAP", 0)
    try:
        llm.LLMClient(token=None).chat([{"role": "user", "content": "hi"}])
        raise AssertionError("expected the pool cap to stop the call")
    except PoolExhausted as exc:
        assert "Add your own Hugging Face token" in str(exc)


def test_byok_token_bypasses_the_pool_entirely(monkeypatch, tmp_path):
    from daggrstudio import llm

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(llm, "POOL_PATH", tmp_path / "pool.json")
    monkeypatch.setattr(llm, "POOL_CAP", 0)
    token, is_byok = llm.resolve_token("hf_user_supplied")
    assert (token, is_byok) == ("hf_user_supplied", True)
    # pool_spend is the only thing that can raise PoolExhausted, and BYOK never reaches it
    assert llm.pool_status()["used"] == 0


def test_pool_counts_only_non_byok_calls(monkeypatch, tmp_path):
    from daggrstudio import llm

    monkeypatch.setattr(llm, "POOL_PATH", tmp_path / "pool.json")
    token, is_byok = llm.resolve_token("hf_user_supplied")
    assert is_byok is True
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert llm.resolve_token(None)[0] is None


def test_pool_status_reports_remaining(monkeypatch, tmp_path):
    from daggrstudio import llm

    monkeypatch.setattr(llm, "POOL_PATH", tmp_path / "pool.json")
    monkeypatch.setattr(llm, "POOL_CAP", 5)
    status = llm.pool_status()
    assert status["remaining"] == 5
    llm.pool_spend(2)
    assert llm.pool_status()["used"] == 2 and llm.pool_status()["remaining"] == 3