#!/usr/bin/env python
"""
End-to-end smoke test against live Hugging Face Spaces.

This is the test that matters: it builds a real workflow from the registry, validates it
against live Space APIs, heals it if needed, renders the code, RUNS it with daggr's executor
and reports the artifacts that came out. Run it before any deploy.

    python scripts/smoke_run.py                 # concept -> sprite (2 bricks + 1 fn)
    python scripts/smoke_run.py --intent "..."  # plan a workflow from scratch (needs a token)
    python scripts/smoke_run.py --dry           # validate + render only, no execution
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from daggrstudio.codegen import render_app_py  # noqa: E402
from daggrstudio.medic import heal  # noqa: E402
from daggrstudio.registry.bricks import get_registry  # noqa: E402
from daggrstudio.runner import run_spec  # noqa: E402
from daggrstudio.spec import Binding, InputPort, Step, WorkflowSpec  # noqa: E402


def demo_spec(registry) -> WorkflowSpec:
    """concept art -> background removal -> collect: the canonical bricks-to-asset flow."""
    image_gen = next(
        (b for b in registry.find(modalities=["image-gen"], commercial_only=True)
         if b.status == "running"), None)
    cutout = next(
        (b for b in registry.find(modalities=["image-edit"], commercial_only=True)
         if b.status == "running" and b.output_kind == "image"), None)
    if image_gen is None or cutout is None:
        raise SystemExit("registry has no usable image-gen/image-edit bricks; run "
                         "scripts/verify_registry.py first")

    return WorkflowSpec(
        name="Concept to sprite",
        intent="Turn a one-line idea into a game sprite with a transparent background",
        industry="game-dev",
        tags=["game-dev", "image-gen", "image-edit"],
        license_posture="commercial-only",
        compute_tier="cloud-free",
        inputs=[InputPort(port="prompt", component="textbox", label="Describe your sprite",
                          default="a small fire elemental, cartoon style, centered", lines=3)],
        steps=[
            Step(id="concept", kind="space", brick_id=image_gen.id, title="Concept art",
                 why="text idea -> image", inputs={
                     "prompt": Binding(source="user:prompt"),
                     **{p: Binding(value=768) for p in ("width", "height") if p in image_gen.inputs},
                     **({"seed": Binding(callable="random_int")} if "seed" in image_gen.inputs else {}),
                     **({"randomize_seed": Binding(value=True)} if "randomize_seed" in image_gen.inputs else {}),
                     **({"num_inference_steps": Binding(value=4)} if "num_inference_steps" in image_gen.inputs else {}),
                 },
                 outputs={"image": {"component": "image", "label": "Concept"}}),
            Step(id="cutout", kind="space", brick_id=cutout.id, title="Cut out background",
                 why="sprites need alpha", inputs={"image": Binding(source="step:concept.image")},
                 outputs={"image": {"component": "image", "label": "Sprite"}}),
            Step(id="collect", kind="fn", fn="organize_outputs", title="Collect artifacts",
                 inputs={"source": Binding(source="step:cutout.image"),
                         "output_dir": Binding(value="./output/sprites")},
                 outputs={"report": {"component": "json", "label": "Run report"}}),
        ],
    )


def planned_spec(intent: str, token: str | None, industry: str, license_posture: str) -> WorkflowSpec:
    from daggrstudio.llm import LLMClient
    from daggrstudio.planner import plan

    client = LLMClient(token=token)
    return plan(intent, registry=get_registry(), client=client, industry=industry,
                license_posture=license_posture).spec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", default=None, help="plan from a goal instead of the demo")
    parser.add_argument("--industry", default="game-dev")
    parser.add_argument("--license", default="commercial-only",
                        choices=["commercial-only", "any"])
    parser.add_argument("--dry", action="store_true", help="no execution")
    parser.add_argument("--no-heal", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    registry = get_registry()
    print(f"registry: {registry.summary()}")

    if args.intent:
        print(f"\n── planning from intent: {args.intent}")
        t0 = time.time()
        spec = planned_spec(args.intent, token, args.industry, args.license)
        print(f"   planned in {time.time() - t0:.1f}s with {spec.planner_model or '?'}")
    else:
        spec = demo_spec(registry)
    print(f"\n── workflow '{spec.name}' ({len(spec.steps)} steps)")
    for step in spec.steps:
        brick = registry.by_id(step.brick_id)
        print(f"   {step.id:12s} {step.kind:9s} {brick.source if brick else (step.source or step.fn)}"
              f"{brick.api_name if brick and brick.api_name else ''}")

    if not args.no_heal:
        print("\n── validate + heal")
        result = heal(spec, registry=registry, token=token, live=True)
        spec = result.spec
        print(f"   status={result.status} model={result.model or '-'}")
        for line in result.timeline():
            print(f"   {line}")
        for issue in result.report.blocking:
            print(f"   STILL BLOCKING: {issue}")

    print("\n── rendered app.py (first 40 lines)")
    code = render_app_py(spec, registry)
    for line in code.splitlines()[:40]:
        print(f"   {line}")
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "smoke_app.py").write_text(code)
    print(f"   ... full file: {ROOT / 'output' / 'smoke_app.py'}")

    if args.dry:
        return 0

    print("\n── running (live Spaces; first call may take 60-120s)")
    values = {"prompt": "a small fire elemental, cartoon style, centered, plain white background"}
    result = run_spec(spec, values=values, token=token, registry=registry)
    for step in result.steps:
        flag = "ok " if step.ok else "FAIL"
        print(f"   [{flag}] {step.step_id:12s} {step.seconds:6.1f}s {step.brick_id or step.title}")
        for repair in step.repairs:
            print(f"          ↳ repaired: {repair}")
        if step.error:
            print(f"          ! {step.error_class}: {step.error[:200]}")
        for art in step.artifacts[:3]:
            size = Path(art).stat().st_size if Path(art).exists() else 0
            print(f"          → {art} ({size:,} bytes)")
    print(f"\n   {result.summary()}")
    if result.note:
        print(f"   note: {result.note}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())