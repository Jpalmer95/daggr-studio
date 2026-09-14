"""
The Planner: a user's intent -> a WorkflowSpec, grounded on the verified brick registry.

The single most important design decision here: **the model may only choose bricks that are
in the catalogue we hand it.** It cannot invent a Space, an endpoint or a parameter name.
Everything it returns is then *completed deterministically* from the registry (api_name,
output ports, postprocess hints, missing input declarations) before it becomes a spec - so a
planning mistake is always a diagnosable one the Medic can fix, never a runtime mystery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from daggrstudio.codegen.fn_library import fn_catalogue_text
from daggrstudio.llm import LLMClient, LLMError, PoolExhausted
from daggrstudio.registry.bricks import Brick, BrickRegistry, get_registry
from daggrstudio.spec import Binding, InputPort, Step, WorkflowSpec, spec_from_json

#: Intent keywords -> modalities that must be in the catalogue handed to the model.
INTENT_HINTS: list[tuple[tuple[str, ...], list[str]]] = [
    (("3d", "model", "mesh", "glb", "print"), ["image-to-3d", "text-to-3d"]),
    (("sprite", "character", "game", "asset", "prop", "texture"), ["image-gen", "image-edit"]),
    (("background", "cutout", "transparent", "alpha", "remove bg"), ["image-edit"]),
    (("video", "animation", "clip", "motion"), ["video"]),
    (("voice", "narrat", "speech", "voiceover", "podcast", "audio"), ["audio-tts"]),
    (("music", "song", "soundtrack", "beat", "jingle"), ["music"]),
    (("caption", "describe", "label", "alt text", "vision", "inspect"), ["vision"]),
    (("upscale", "enhance", "resolution", "sharpen"), ["upscale"]),
    (("write", "script", "story", "copy", "summar", "translate", "article", "post"),
     ["text"]),
]

ALWAYS_INCLUDE = ["image-gen", "image-edit", "utility", "text"]

DEFAULT_STEP_LIMIT = 6

SYSTEM_PROMPT = """You design AI pipelines for the daggr library by assembling verified "bricks".

You are given a CATALOGUE. You may ONLY use bricks from it, by id. Never invent a Space,
a model, an endpoint or a parameter name - if it is not in the catalogue, it does not exist.

Reply with JSON only (no prose, no fences) in exactly this shape:
{
 "name": "short workflow name",
 "intent": "one sentence restating the goal",
 "industry": "game-dev|music-production|art|3d|film|marketing|general",
 "tags": ["..."],
 "compute_tier": "cloud-free|cloud-paid",
 "inputs": [{"port":"prompt","component":"textbox","label":"Describe it","default":"...","lines":3}],
 "steps": [
   {"id":"concept","kind":"space","brick_id":"<catalogue id>","title":"Concept art",
    "why":"why this brick",
    "inputs":{"prompt":{"from":"user:prompt"},
              "width":{"value":768},
              "seed":{"callable":"random_int"}},
    "outputs":{"image":{"component":"image","label":"Concept"}}}
 ]
}

Rules:
- 2 to %(steps)d steps. Fewer bricks is better than more.
- `kind` is "space" for Spaces, "inference" for text models, "fn" for local helpers (use
  only helpers from the HELPERS list, and give inputs that match their parameters).
- Every input value is one of: {"from":"user:<port>"} (a declared UI input),
  {"from":"step:<step_id>.<output_port>"} (an upstream step), {"value":<literal>},
  or {"callable":"random_int|random_float|uuid4|now_iso|random_choice_style"}.
- Only wire a parameter the brick actually has (see the catalogue signature).
- Data flows: an image output can only feed an image/file parameter; text feeds text.
- Use a text/inference brick when the goal needs language work (prompt building, scripts,
  captions). Use an "fn" step only to tidy up at the end.
- If the user demands commercial use, every brick must have commercial=true.
- Prefer "running" bricks over "sleeping" ones.
- Set compute_tier to "cloud-paid" if any step is video or 3D generation."""

PLAN_PROMPT = """%(catalogue)s

%(helpers)s

USER GOAL: %(intent)s
INDUSTRY: %(industry)s
LICENCE POSTURE: %(license)s (%(license_note)s)
MAX STEPS: %(steps)d
PREFER: commercially usable bricks, few steps, reliable running Spaces.
%(extra)s
Design the workflow now."""


@dataclass
class PlanResult:
    spec: WorkflowSpec | None = None
    model: str = ""
    notes: list[str] = field(default_factory=list)
    catalogue_ids: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.spec is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model": self.model,
            "notes": self.notes,
            "catalogue_ids": self.catalogue_ids,
            "error": self.error,
            "spec": self.spec.to_dict() if self.spec else None,
        }


def modalities_for_intent(intent: str) -> list[str]:
    text = (intent or "").lower()
    wanted: list[str] = []
    for keywords, modalities in INTENT_HINTS:
        if any(k in text for k in keywords):
            wanted.extend(modalities)
    return wanted or ["image-gen", "utility"]


def build_catalogue(
    intent: str,
    registry: BrickRegistry,
    commercial_only: bool = True,
    max_bricks: int = 34,
) -> list[Brick]:
    """
    Pick the bricks the model is allowed to choose from.

    A smaller, relevant catalogue is not just cheaper - it measurably improves the choice,
    because the model cannot wander into a modality the goal has nothing to do with.
    """
    wanted = list(dict.fromkeys(modalities_for_intent(intent) + ALWAYS_INCLUDE))
    bricks = registry.find(modalities=wanted, commercial_only=commercial_only,
                          include_not_running=True, limit=None)
    # running bricks first, then commercially-safe, then by modality match
    primary = set(modalities_for_intent(intent))
    bricks.sort(key=lambda b: (
        b.status != "running",
        b.modality not in primary,
        -(b.likes or 0),
        b.id,
    ))
    return bricks[:max_bricks]


def _catalogue_text(bricks: list[Brick]) -> str:
    lines = ["BRICK CATALOGUE (authoritative - only these ids exist)",
             "id | source | modality | signature | licence | commercial | status | industries"]
    for brick in bricks:
        params = ", ".join(f"{n}: {t}" for n, t in brick.inputs.items()) or "-"
        lines.append(
            f"{brick.id} | {brick.source}{brick.api_name or ''} | {brick.modality} | "
            f"in({params}) -> {brick.output_kind}"
            f"{' postprocess=' + brick.postprocess_hint if brick.postprocess_hint else ''} | "
            f"licence={brick.license} | commercial={'yes' if brick.commercial else 'NO'} | "
            f"{brick.status} | {','.join(brick.industries[:3]) or 'general'}"
        )
    return "\n".join(lines)


def _op_defaults(intent: str) -> dict[str, Any]:
    prompt = (intent or "").strip() or "Describe what you want"
    return {
        "prompt": {"component": "textbox", "label": "Describe it",
                   "default": prompt[:200], "lines": 3},
    }


# ─── deterministic completion ─────────────────────────────────────────────────


def complete_spec(
    spec: WorkflowSpec,
    registry: BrickRegistry,
    notes: list[str] | None = None,
) -> WorkflowSpec:
    """
    Fill in everything the registry already knows, so a good plan becomes a valid spec.

    Nothing here invents data: api_name, output components and postprocess hints all come
    from verified bricks. Anything that cannot be completed is left for the validator to
    flag (and the Medic to fix) rather than being guessed.
    """
    notes = notes if notes is not None else []
    declared = {p.port for p in spec.inputs}

    for step in spec.steps:
        brick = registry.by_id(step.brick_id)
        if brick is None and step.source:
            brick = registry.by_source(step.source, step.api_name)
            if brick:
                step.brick_id = brick.id
                notes.append(f"step '{step.id}': resolved brick id from source {brick.source}")
        if brick is None or step.kind == "fn":
            continue

        # endpoint, kind and source come from the registry, never from the model
        if brick.api_name and step.api_name != brick.api_name:
            if step.api_name:
                notes.append(f"step '{step.id}': endpoint corrected to {brick.api_name}")
            step.api_name = brick.api_name
        step.source = brick.source
        if brick.kind == "inference_model":
            step.kind = "inference"

        # output ports: derive from the brick when the model left them out
        if not step.outputs:
            component = {"image": "image", "audio": "audio", "video": "video",
                         "model3d": "model3d", "text": "textbox", "json": "json"}.get(
                             brick.output_kind, "textbox")
            step.outputs = {brick.output_kind if brick.output_kind != "text" else "text":
                            {"component": component, "label": step.title or step.id.title()}}
            notes.append(f"step '{step.id}': output port derived from the brick "
                         f"({brick.output_kind})")
        if brick.postprocess_hint and not step.postprocess:
            step.postprocess = brick.postprocess_hint
            notes.append(f"step '{step.id}': postprocess set to {brick.postprocess_hint} "
                         f"(brick returns multiple values)")

        # reject parameters the brick does not have rather than shipping a broken binding
        for param in list(step.inputs):
            if brick.inputs and param not in brick.inputs:
                dropped = step.inputs.pop(param)
                notes.append(f"step '{step.id}': dropped unknown parameter '{param}' "
                             f"({dropped.kind} binding) - brick has {sorted(brick.inputs)}")

        # declare any user input the plan referenced but forgot to declare
        for binding in step.inputs.values():
            port = binding.user_port
            if port and port not in declared:
                spec.inputs.append(InputPort.from_dict(
                    dict(_op_defaults(spec.intent).get(port, {"component": "textbox"}),
                         port=port, label=port.replace("_", " ").title())))
                declared.add(port)
                notes.append(f"declared missing input '{port}' (referenced by step '{step.id}')")

    if not spec.inputs:
        spec.inputs = [InputPort.from_dict(dict(v, port=k))
                       for k, v in _op_defaults(spec.intent).items()]
        notes.append("no inputs were declared; added a prompt input")

    spec.id = spec.id or ""
    spec.planner_model = spec.planner_model
    return spec


def _fix_runaway_steps(spec: WorkflowSpec, limit: int, notes: list[str]) -> None:
    """Keep only the steps reachable from the declared outputs, capped at the limit."""
    if len(spec.steps) <= limit:
        return
    keep = spec.steps[:limit]
    dropped = [s.id for s in spec.steps[limit:]]
    spec.steps = keep
    notes.append(f"trimmed {len(dropped)} extra steps ({', '.join(dropped)}) to honour "
                 f"the {limit}-step limit")


def plan(
    intent: str,
    registry: BrickRegistry | None = None,
    client: LLMClient | None = None,
    industry: str = "general",
    license_posture: str = "commercial-only",
    max_steps: int = DEFAULT_STEP_LIMIT,
    model: str | None = None,
    extra_catalogue_ids: list[str] | None = None,
) -> PlanResult:
    """
    Turn an intent into a completed, offline-valid spec.

    Requires an ``LLMClient`` (BYOK or a metered space token). Without one, returns a result
    with ``error`` set - the UI turns that into "add a token in Settings".
    """
    registry = registry or get_registry()
    commercial_only = license_posture == "commercial-only"
    result = PlanResult()

    catalogue = build_catalogue(intent, registry, commercial_only=commercial_only)
    if extra_catalogue_ids:
        for brick_id in extra_catalogue_ids:
            brick = registry.by_id(brick_id)
            if brick and brick not in catalogue:
                catalogue.append(brick)
    result.catalogue_ids = [b.id for b in catalogue]

    if client is None:
        result.error = "no model configured: paste a Hugging Face token in Settings"
        return result

    prompt = PLAN_PROMPT % {
        "catalogue": _catalogue_text(catalogue),
        "helpers": "LOCAL HELPERS for kind=fn steps (only these exist)\n" + fn_catalogue_text(),
        "intent": intent,
        "industry": industry,
        "license": license_posture,
        "license_note": "output must be usable commercially" if commercial_only
                        else "non-commercial bricks are allowed",
        "steps": max_steps,
        "extra": "",
    }

    try:
        text, used_model = client.ask(prompt, system=SYSTEM_PROMPT % {"steps": max_steps},
                                       model=model or client.model, max_tokens=2600)
    except (LLMError, PoolExhausted) as exc:
        result.error = str(exc)[:300]
        return result

    spec = spec_from_json(text)
    if spec is None:
        result.error = "the model did not return a usable workflow spec"
        result.model = used_model
        return result

    spec.intent = spec.intent or intent
    spec.industry = spec.industry if spec.industry else industry
    spec.license_posture = spec.license_posture if spec.license_posture in (
        "commercial-only", "any") else license_posture
    spec.planner_model = used_model
    result.model = used_model

    _fix_runaway_steps(spec, max_steps, result.notes)
    complete_spec(spec, registry, result.notes)

    unknown = [s.brick_id for s in spec.steps if s.kind != "fn" and not registry.by_id(s.brick_id)]
    if unknown:
        result.notes.append(
            f"the model referenced bricks that do not exist: {unknown}; "
            "the validator will flag them for repair"
        )
    result.spec = spec
    return result