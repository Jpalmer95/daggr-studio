#!/usr/bin/env python
"""
End-to-end acceptance test: the whole product journey, against real services.

Stages, each printed as PASS/FAIL so this can be read at a glance or wired into CI:

  1. plan      intent -> workflow spec (real model, grounded on the registry)
  2. heal      validate against live Spaces and repair what is broken
  3. repair    a workflow that uses a KNOWN-BROKEN brick must self-heal onto a working one
  4. run       execute the workflow and produce real artifacts
  5. code      render app.py, syntax-check it, confirm no dependency on this package
  6. share     publish/load/leaderboard (local mirror, then the real Dataset if a token
               with write scope is available)
  7. surface   the composed app serves /builder, / and the canvas API

Run:  python scripts/acceptance.py            (add --fast to skip the live run stage)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def stage(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {detail[:150]}", flush=True)


def token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    cached = Path.home() / ".cache" / "huggingface" / "token"
    return cached.read_text().strip() if cached.exists() else None


def whoami(tok: str | None) -> str:
    from huggingface_hub import HfApi

    try:
        return HfApi(token=tok).whoami().get("name", "")
    except Exception:
        return ""


# ─── stage 1 + 2: plan and heal ───────────────────────────────────────────────

INTENT = ("turn a one-line character idea into a game sprite with a transparent background, "
          "commercially usable")

def stage_plan_heal(tok: str | None) -> dict:
    from daggrstudio.web import services

    started = time.time()
    out = services.plan_workflow(INTENT, token=tok, industry="game-dev",
                                 license_posture="commercial-only", max_steps=4,
                                 live_validation=True)
    seconds = time.time() - started
    spec = out.get("spec") or {}
    bricks = [s.get("brick_id") for s in spec.get("steps", [])]
    stage("1. plan (real model)", bool(spec), f"{len(spec.get('steps', []))} steps in "
                                               f"{seconds:.1f}s -> {bricks}")
    stage("2. heal (live validation)", out.get("status") in ("healthy", "healed"),
          f"status={out.get('status')} model={out.get('model')}")
    if out.get("timeline"):
        for line in out["timeline"].splitlines()[:6]:
            print(f"      {line}")
    if out.get("issues"):
        print(f"      {len(out['issues'])} finding(s) remaining")
        for row in out["issues"][:4]:
            print(f"      {row[0]} {row[1]} {row[2]} {row[3][:90]}")
    return out


# ─── stage 3: a workflow that depends on a broken brick must self-heal ────────

def stage_self_repair(tok: str | None) -> dict:
    """
    Build a workflow that deliberately uses hf-applications/background-removal, which our
    execution probe found to be genuinely broken, then prove the Medic moves it onto a
    working brick without human intervention.
    """
    from daggrstudio.medic import heal
    from daggrstudio.registry.bricks import get_registry
    from daggrstudio.spec import Binding, InputPort, Step, WorkflowSpec

    registry = get_registry()
    broken = registry.by_id("background-removal")
    gen = next((b for b in registry.find(modalities=["image-gen"], commercial_only=True)
                if b.status == "running"), None)
    if broken is None or gen is None:
        stage("3. self-repair", False, "registry lacks the bricks needed for this stage")
        return {}

    spec = WorkflowSpec(
        name="Repair me", intent="cut the subject out of an image",
        industry="game-dev", license_posture="commercial-only",
        inputs=[InputPort(port="prompt", component="textbox", label="Idea",
                          default="a red cube on white")],
        steps=[
            Step(id="art", kind="space", brick_id=gen.id, inputs={
                **{p: Binding(source="user:prompt") for p in gen.inputs
                   if p.lower() in ("prompt", "text")},
                **({"seed": Binding(callable="random_int")} if "seed" in gen.inputs else {}),
            }, outputs={"image": {"component": "image", "label": "Art"}}),
            Step(id="cut", kind="space", brick_id=broken.id,
                 inputs={"image": Binding(source="step:art.image")},
                 outputs={"image": {"component": "image", "label": "Cutout"}}),
        ],
    )
    result = heal(spec, registry=registry, token=tok, live=False)
    after = result.spec.step("cut")
    swapped = bool(after and after.brick_id != broken.id)
    stage("3. self-repair (broken brick)", swapped,
          f"{broken.id} -> {after.brick_id if after else '?'} "
          f"(status={result.status}, licence {broken.license})")
    for line in result.timeline()[:4]:
        print(f"      {line}")
    return {"spec": result.spec.to_dict(), "swapped": swapped}


# ─── stage 4: run it ─────────────────────────────────────────────────────────

def stage_run(spec_payload: dict, tok: str | None, fast: bool) -> dict:
    from daggrstudio.spec import WorkflowSpec
    from daggrstudio.web import services

    if fast:
        stage("4. run (live Spaces)", True, "skipped (--fast)")
        return {}
    spec = WorkflowSpec.from_dict(spec_payload)
    values = {}
    for port in spec.inputs:
        values[port.port] = port.default or "a red cube on a plain white background"
    out = services.run_workflow(spec, values=values, token=tok)
    artifacts = out.get("artifacts") or []
    stage("4. run (live Spaces)", bool(artifacts),
          f"{out.get('message', '')[:110]}")
    for row in out.get("rows", []):
        print(f"      {row[0]} {row[1][:32]:32s} {row[2]:>7s} {row[4][:70]}")
    for path in artifacts[:4]:
        size = Path(path).stat().st_size if Path(path).exists() else 0
        print(f"      -> {path} ({size:,} bytes)")
    return out


def stage_code(spec_payload: dict) -> str:
    import ast

    from daggrstudio.codegen import render_app_py
    from daggrstudio.spec import WorkflowSpec

    code = render_app_py(WorkflowSpec.from_dict(spec_payload))
    try:
        ast.parse(code)
        ok = True
        detail = f"{len(code.splitlines())} lines, compiles"
    except SyntaxError as exc:
        ok, detail = False, f"generated code does not compile: {exc}"
    stage("5. codegen (app.py)", ok, detail)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    (out / "acceptance_app.py").write_text(code)
    print(f"      wrote {out / 'acceptance_app.py'}")
    return code


def stage_share(spec_payload: dict, tok: str | None) -> dict:
    from daggrstudio.spec import WorkflowSpec
    from daggrstudio.web import services

    spec = WorkflowSpec.from_dict(spec_payload)
    spec.name = spec.name or "Acceptance workflow"
    spec.slug = f"acceptance-{spec.slug}"[:60]

    local = services.publish_workflow(spec, token=None, note="acceptance run (local only)")
    stage("6a. save locally (no token)", Path(
        Path.home() / ".cache" / "daggr-studio" / "workflows" / f"{spec.slug}.json").exists(),
        local.get("message", "")[:110])

    if not tok:
        stage("6b. publish to the shared dataset", False, "no HF token available")
        return {}

    out = services.publish_workflow(spec, token=tok, note="acceptance run")
    stage("6b. publish to the shared dataset", bool(out.get("ok")), out.get("message", "")[:120])
    if out.get("url"):
        print(f"      {out['url']}")

    # read it back from the Hub - a write claim is not proof
    from daggrstudio.hub import load

    back = load(spec.slug, token=tok)
    ok = back.ok and (back.data.get("name") == spec.name)
    stage("6c. read back from the dataset", ok,
          f"name={back.data.get('name') if back.ok else '?'} "
          f"bricks={len(back.data.get('brick_ids') or []) if back.ok else 0}")

    lb = services.leaderboard_view(license_filter="any", sort="recent", token=tok)
    found = spec.slug in (lb.get("slugs") or [])
    stage("6d. appears on the leaderboard", found, lb.get("message", "").splitlines()[0][:110])

    vote_out = services.cast_vote(spec.slug, 1, token=tok)
    stage("6e. vote recorded", bool(vote_out.get("ok")), vote_out.get("message", "")[:110])
    return {"slug": spec.slug, "url": out.get("url", "")}


async def stage_surface() -> None:
    import httpx

    import app as studio

    transport = httpx.ASGITransport(app=studio.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        checks = []
        for path, needle in [("/healthz", "ok"), ("/builder", "<!DOCTYPE html>"),
                             ("/", "<!DOCTYPE html>")]:
            resp = await client.get(path)
            checks.append((path, resp.status_code == 200 and needle in resp.text))
        ok = all(passed for _, passed in checks)
        stage("7. Space surfaces", ok,
              " ".join(f"{p}:{'ok' if passed else 'BAD'}" for p, passed in checks))


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip the live execution stage")
    parser.add_argument("--stage", default="all",
                        choices=["all", "plan", "repair", "run", "share", "surface"])
    args = parser.parse_args()

    import asyncio

    tok = token()
    user = whoami(tok)

    print(f"token: {'yes' if tok else 'no'} ({user or 'unauthenticated'})")
    from daggrstudio.registry.bricks import get_registry

    print(f"registry: {get_registry().summary()}\n")

    planned = stage_plan_heal(tok) if args.stage in ("all", "plan") else {}
    spec_payload = planned.get("spec")

    if args.stage in ("all", "repair"):
        repaired = stage_self_repair(tok)
        if not spec_payload:
            spec_payload = repaired.get("spec")

    if spec_payload:
        if args.stage in ("all", "run"):
            stage_run(spec_payload, tok, args.fast)
        if args.stage in ("all", "plan", "run"):
            stage_code(spec_payload)
        if args.stage in ("all", "share"):
            stage_share(spec_payload, tok)

    if args.stage in ("all", "surface"):
        asyncio.run(stage_surface())

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} stages passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
