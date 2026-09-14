"""
Shared fixtures.

Important principle: the fixtures pick bricks **from the live registry** rather than naming
them. Bricks break and recover constantly (that is the whole point of this project), so a
test that hardcodes `flux1-schnell` is a test that will fail for reasons unrelated to the
code under test.
"""

from __future__ import annotations

import pytest

from daggrstudio.registry.bricks import Brick, BrickRegistry, get_registry
from daggrstudio.spec import Binding, InputPort, Step, WorkflowSpec


def healthy_image_gen(registry: BrickRegistry | None = None) -> Brick | None:
    reg = registry or get_registry()
    candidates = [b for b in reg.find(modalities=["image-gen"], commercial_only=True)
                  if b.status == "running" and b.api_name]
    return candidates[0] if candidates else None


def healthy_image_edit(registry: BrickRegistry | None = None) -> Brick | None:
    """
    Prefer a cutout-style brick: image in, image out, **no prompt**.

    An instruction-editing brick (Kontext, Qwen-Image-Edit) also needs a text instruction,
    which is a different job - and a test fixture that ignores that would produce a spec
    that daggr rightly refuses to build.
    """
    reg = registry or get_registry()
    candidates = [b for b in reg.find(modalities=["image-edit"], commercial_only=True)
                  if b.status == "running" and b.api_name and b.output_kind == "image"]
    cutout_style = [b for b in candidates if "prompt" not in b.inputs]
    return (cutout_style or candidates or [None])[0]


def healthy_tts(registry: BrickRegistry | None = None) -> Brick | None:
    reg = registry or get_registry()
    candidates = [b for b in reg.find(modalities=["audio-tts"], commercial_only=True)
                  if b.status == "running" and b.api_name]
    return candidates[0] if candidates else None


def file_param(brick: Brick) -> str:
    """The brick's own name for its main file input ('image', 'input_image', 'img', ...)."""
    for param in brick.inputs:
        low = param.lower()
        if any(k in low for k in ("image", "img", "photo")):
            return param
    for param, ptype in brick.inputs.items():
        if "path" in (ptype or "").lower() or (ptype or "").lower() == "file":
            return param
    return next(iter(brick.inputs), "image")


def _bindings_for(brick: Brick, prompt_source: str = "user:prompt") -> dict[str, Binding]:
    """Wire the brick's required-ish parameters, using the brick's own parameter names."""
    bindings: dict[str, Binding] = {}
    for param in brick.inputs:
        low = param.lower()
        if low in ("prompt", "text", "texts") or low.startswith("prompt"):
            bindings[param] = Binding(source=prompt_source)
        elif any(k in low for k in ("image", "img", "photo")):
            bindings[param] = Binding(source=prompt_source)  # tests never execute this
        elif "seed" in low:
            bindings[param] = Binding(callable="random_int")
        else:
            bindings[param] = Binding(value=True)
    return bindings


def concept_spec(registry: BrickRegistry | None = None) -> WorkflowSpec:
    """
    The canonical bricks-to-asset flow: concept art -> cutout -> tidy up.

    Bricks are chosen from whatever the registry currently reports as healthy; if the
    registry has nothing suitable the flow falls back to the seed brick, because the
    *shape* of the spec is what most tests care about.
    """
    reg = registry or get_registry()
    gen = healthy_image_gen(reg) or reg.by_id("flux1-schnell")
    cut = healthy_image_edit(reg) or reg.by_id("background-removal")
    assert gen is not None and cut is not None, "registry is empty"

    steps = [
        Step(id="concept", kind="space", brick_id=gen.id, title="Concept art",
             why="text idea -> image", inputs=_bindings_for(gen),
             outputs={"image": {"component": "image", "label": "Concept"}}),
        Step(id="cutout", kind="space", brick_id=cut.id, title="Cut out background",
             why="sprites need alpha",
             inputs={**{p: Binding(value="remove the background, keep the subject"
                                  ) for p in cut.inputs if p.lower() in ("prompt", "text")},
                     file_param(cut): Binding(source="step:concept.image")},
             outputs={"image": {"component": "image", "label": "Sprite"}}),
        Step(id="collect", kind="fn", fn="organize_outputs", title="Collect artifacts",
             inputs={"source": Binding(source="step:cutout.image"),
                     "output_dir": Binding(value="./output/sprites")},
             outputs={"report": {"component": "json", "label": "Run report"}}),
    ]
    return WorkflowSpec(
        name="Concept to sprite",
        intent="Turn a text idea into a sprite with a transparent background",
        industry="game-dev",
        tags=["game-dev", "image-gen", "image-edit"],
        license_posture="commercial-only",
        compute_tier="cloud-free",
        inputs=[
            InputPort(port="prompt", component="textbox", label="Describe your sprite",
                      default="a fire elemental, cartoon style", lines=3),
            InputPort(port="width", component="slider", label="Width", default=1024,
                      minimum=256, maximum=2048, step=64),
        ],
        steps=steps,
    )


@pytest.fixture()
def spec() -> WorkflowSpec:
    return concept_spec()


@pytest.fixture()
def spec_dict(spec: WorkflowSpec) -> dict:
    return spec.to_dict()