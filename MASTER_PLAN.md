# Daggr Studio — MASTER_PLAN

> **For Hermes / future agents:** this is the authoritative roadmap for this repo.
> Work phase by phase. Mark `[x]` only after the phase's Success Criteria pass.
> Never commit secrets. Keep the Space self-contained (no private deps).

**Goal:** make building with the [daggr](https://github.com/gradio-app/daggr) library as
easy as assembling bricks — describe an intent, get a *validated, self-healing* daggr
workflow, run it, then save/share it and compete on a community leaderboard.

**Owner:** Jonathan (Jpalmer95 / HF: jkorstad) · **Started:** 2026-09-14 · **Status:** in build

---

## 1. Vision

A user types "make a 3D game character from a text idea, transparent background,
commercially usable output". Daggr Studio:

1. picks bricks from a **live-verified registry** of Spaces/models,
2. emits a **spec + real daggr Python code**,
3. **validates** it against live Space APIs (endpoint names, param names, port types),
4. hands failures to a **Medic** (lightweight model on HF Inference, default
   `deepseek-ai/DeepSeek-V4.1-Flash`) that patches the spec and re-validates until clean,
5. **runs it in-process** with daggr's own executor, with runtime failover if a sister
   Space is asleep/broken/changed its API,
6. and lets the user **save, share, and rank** the workflow by modality / industry / license.

Two UIs, one process: the **Builder** at `/builder` and the real **daggr canvas** at `/`.

## 2. Architecture

```
                 ┌─────────────────────────── Hugging Face Space (docker, cpu-basic) ──────────────────────────┐
                 │  uvicorn :7860                                                                              │
  /         ───▶ │  ASGI shim ──▶ current DaggrServer(FastAPI) app   ← hot-swapped when a workflow is built     │
  /builder  ───▶ │  Builder (Gradio Blocks, mounted before the canvas mount)                                   │
  /assets,/api,/ws ─▶ canvas (daggr's own absolute paths must stay at root)                                   │
                 └────────────────────────────────────────────────────────────────────────────────────────────┘
   intent ─▶ Planner(LLM) ─▶ Spec(JSON) ─▶ CodeGen(deterministic) ─▶ Validator(live introspection)
                                            ▲                              │ issues?
                                            └──── Medic(LLM patch) ◀───────┘
   Spec ─▶ BrickRunner (daggr SequentialExecutor) ─▶ per-step outputs + runtime repair log
   Spec ─▶ Hub store (HF Dataset) ─▶ save / share / leaderboard / votes
```

**Root RFC:** the ASGI shim is required because daggr's prebuilt frontend uses
**absolute** URLs (`/assets/…`, `/api/…`, `/ws/…`, `/theme.css`). Mounting it under a
sub-path would 404 its own assets, so the canvas must own the root and the Builder lives
under a prefix that is registered *before* the canvas catch-all mount.

### Modules (`daggrstudio/`)

| File | Responsibility |
|---|---|
| `registry/bricks.py` | load `spaces.json`, query bricks by modality/industry/license/cost |
| `registry/spaces.json` | **live-verified** brick registry (subagent-built, regenerable) |
| `spec.py` | `WorkflowSpec` dataclasses + JSON (de)serialization + validation of shape |
| `planner.py` | intent + filters + brick catalogue → spec (LLM, grounded on registry) |
| `codegen.py` | spec → runnable daggr `app.py` **and** spec → live `Graph` object (no LLM) |
| `validator.py` | static + live checks → `Issue(code, severity, node_id, message, fix_hint)` |
| `medic.py` | heal loop (patch spec via LLM), upgrade advisor, model-fallback chain |
| `runner.py` | execute a `Graph` in-process via daggr executor; runtime failover |
| `hub.py` | HF Dataset store: workflows, votes, leaderboard aggregation |
| `web/builder.py` | Gradio UI (intent → spec → code → run → publish → leaderboard) |
| `web/canvas.py` | ASGI hot-swap shim + `/` landing page + injected Builder button |
| `app.py` (root) | compose: builder at `/builder`, shim at `/` |

## 3. Cost / sustainability model (§ required)

Reading is free, generating uses the caller's compute (BYOK-first):

| Action | Who pays |
|---|---|
| Browsing templates, registry, leaderboard | nobody (static JSON / dataset reads) |
| Planning + healing LLM calls | **user's own HF token** (`HF_TOKEN` pasted in Settings, session-only) |
| Pipeline execution (Space bricks) | free tier for public Spaces; user's token for ZeroGPU bricks |
| Publishing a workflow / voting | user's token (commit to the shared dataset) |
| No token at all | Space's own token, hard-capped **community pool** (`POOL_DAILY_CAP`, default 40 calls/day, server-side counter) |

Hosting: one `cpu-basic` docker Space (no GPU) + one public HF Dataset. Adding users adds
no fixed cost. If funding disappears, browsing/saving-by-download still works.

## 4. Current state (2026-09-14, end of build)

**Shipped and verified.** GitHub: `Jpalmer95/daggr-studio`. Space: `jkorstad/daggr-studio`
(docker SDK, cpu-basic). Dataset: `jkorstad/daggr-studio-workflows`.

Evidence, not claims:

* `pytest -q` → **123 passed**, no network.
* `python scripts/acceptance.py` → **11/11 stages**: plan (real model, 19.6s) → heal (healthy
  first pass) → **self-repair** (`hf-applications/background-removal`, genuinely broken, was
  swapped to `kontext-dev` with the reason logged) → live run (2/2 steps, 16s, real 53KB webp
  + 488KB png artifacts) → codegen (75 lines, compiles) → publish to the Dataset → **read back
  verified** → leaderboard entry → vote recorded → `/healthz`, `/builder`, `/` all 200.
* `scripts/verify_registry.py --refresh --discover --probe` → 44 bricks; **27 reachable,
  9 proven by execution**, 27 commercially usable. Dead Spaces are recorded with their error,
  not silently retried.
* Model facts re-verified live on the HF router: `deepseek-ai/DeepSeek-V4.1-Flash` (~1s, clean
  JSON) is the default planner/medic model; `Qwen/Qwen2.5-7B-Instruct` is **not** served by any
  provider, so it must never be a default.
* `gradio_client.Client(...).view_api(return_dict)` gives exact param names/types →
  the Validator is grounded, not guessy. daggr's own GradioNode validation also rejects unknown
  parameters with suggestions, so both layers agree.
* `DaggrServer(graph).app` is mountable; daggr's frontend uses **absolute** paths, hence the
  canvas owning `/` with the Builder at `/builder` (verified: both 200 in one process).

### Deployed (live, verified over the public internet)

* Space: **https://jkorstad-daggr-studio.hf.space** (Builder at `/builder`, canvas at `/`)
* Repo: https://github.com/Jpalmer95/daggr-studio
* Dataset: https://huggingface.co/datasets/jkorstad/daggr-studio-workflows
* Live checks: `/healthz` 200 with the registry summary; `/` and `/builder` both 200 from one
  process; a real plan driven through the deployed container returned
  **"healthy — ready to run"** (`Sprite Forge`, `flux1-schnell → not-lain-bg-removal`, 0
  findings); the canvas then reported `canvas shows 'Sprite Forge' (3 nodes)` and
  `/api/graph` returned 200 JSON with nodes+edges — i.e. the hot-swap works in production,
  not just locally.
* Agent API added after the first deploy: `POST /api/plan`, `POST /api/validate`,
  `GET /api/canvas`, `GET /api/bricks`, `GET /api/leaderboard`, `GET /healthz`
  (130 tests passing, including a check that a posted token is never echoed back).

### Build failures hit and fixed (worth remembering)

* `gradio==6.27.0` + `gradio_client==2.5.0` is unresolvable (gradio 6.27 pins
  gradio-client==2.7.0). The HF build log only says "exit code 1" — the real reason is in
  `https://huggingface.co/api/spaces/<repo>/logs/build`. Reproduce locally with
  `pip download --python-version 3.12 -d /tmp/x -r requirements.txt` before deploying.
* `pytest-asyncio` is required for the ASGI surface tests (`asyncio_mode = auto`).

## 5. Phased execution

### Phase 0 — Repo + plan  [x]
Scaffold repo, write this plan, `.gitignore`, LICENSE (MIT), README skeleton.

### Phase 1 — Foundations (spec, registry, codegen, tests)
- [x] `spec.py` with `WorkflowSpec/Step/Binding`; round-trip `to_dict`/`from_dict`; unknown
      fields tolerated; `schema_version`.
- [x] `registry/bricks.py` — loader + `find(modality=, industry=, commercial_only=, exclude=)`,
      `by_id()`, graceful handling of `status != running`.
- [x] `codegen.py` — spec → (a) `build_graph(spec)` live Graph, (b) `render_app_py(spec)` text.
      Port wiring, fixed values, callables, `postprocess` from `postprocess_hint`, file-path
      conventions handled centrally (`preprocess`/`postprocess` helpers).
- [x] Tests: `pytest tests/` green, no network.

**Success criteria:** `pytest -q` passes; a hand-written 3-step spec renders an `app.py`
that imports and constructs a `Graph` with correct edges; loader filters are unit-tested.

### Phase 2 — Validation + Medic (the healing pipeline)
- [x] `validator.py` static checks: unknown brick, brick not running, commercial violation,
      api_name mismatch, param-name mismatch, missing required param, edge type mismatch,
      dangling reference, cycle, orphan step, duplicate output port.
- [x] Live introspection cache (`view_api` with TTL + disk cache in `.cache/`) → `refresh=True` mode.
- [x] `medic.py` heal loop: `plan → validate → patch(spec, issues, model) → re-validate`
      (max 4 rounds), records `HealEvent(round, codes, patch_summary, model)`; JSON-only
      prompting with tolerant parse + repair of fenced code blocks; model fallback chain.
- [x] Runtime repair in `runner.py`: on node failure → classify (sleeping / quota /
      signature-changed / dead) → retry-with-backoff, live re-introspection remap,
      registry-substitute failover, then re-heal spec.
- [x] Tests: validator catches each seeded defect; medic loop is unit-tested with a fake
      LLM that returns canned patches (no network).

**Success criteria:** deliberately corrupted specs (bad api_name, wrong param, nc-license
brick under commercial-only, image→audio edge) are all detected with distinct codes, and
the medic loop repairs at least the param/api_name classes using a stub model.

### Phase 3 — Execution + Builder UI
- [x] `runner.py` executes a Graph in-process, emits per-node status/results, saves artifacts
      to a temp dir and returns them as Gradio file paths.
- [x] `web/builder.py`: intent box, filters (modality/industry/commercial/cost/brick count),
      Settings (HF token, planner model, medic model), Plan → Validate → Heal → Code → Run →
      Publish tabs; Heal log + issue table; artifact gallery; download `app.py`.
- [x] "Deploy as my Space" (BYOK): create repo + push generated app + poll status.

**Success criteria:** for one intent, end-to-end in one click: spec → 0 blocking issues →
run → at least one real image artifact produced by a real Space; builder mounts alongside
the canvas with no route collisions (both `/builder` and `/` return 200).

### Phase 4 — Canvas
- [x] `web/canvas.py` ASGI shim: serves `/` landing (or injected canvas index.html),
      delegates `/assets`, `/api`, `/ws`, `/file`, `/theme.css` to the active DaggrServer.
- [x] Inject a floating "🧱 Builder" button into the canvas HTML.

**Success criteria:** after building a workflow, `/` renders the daggr canvas (HTTP 200 on
`/api/graph` returning the generated nodes) and a run started from the canvas executes.

### Phase 5 — Share, leaderboard, community
- [x] `hub.py` over HF Dataset `jkorstad/daggr-studio-workflows`:
      `workflows/<slug>.json`, `votes/<slug>.json`, `index.json` rebuilt on write.
- [x] Publish (BYOK commit), Load by slug, Export/Import local JSON.
- [x] Leaderboard: filters modality / industry / license (commercial-only default) /
      cost tier / sort (votes, recency, brick count); vote + unvote; one-vote-per-user keyed
      by HF username (anonymous → download/manual).
- [x] License posture: index-time validation, `commercial_ok` surfaced as a badge, NC bricks
      flagged in the UI before publishing, contributor policy shown on first publish.

**Success criteria:** publishing writes a real commit to the dataset; leaderboard lists it
with correct filters; voting increments and persists across a Space restart (fresh clone).

### Phase 6 — Ship
- [x] Dockerfile (`python:3.12-slim`, no gradio pin conflicts), README frontmatter
      (`sdk: docker`, `cpu-basic`), `.env`-free secrets (HF_TOKEN as Space secret).
- [x] Push GitHub repo (public), create Space, poll to RUNNING, smoke-test `/`, `/builder`,
      `/api/graph` from the public URL.
- [x] Update the `daggr` / `daggr-pipelines` skills with what was learned.

**Success criteria:** public Space URL serves the builder; a live end-to-end run works on
the deployed Space (not just locally).

## 6. Design decisions (recorded)

- **Bricks, not free-form code, for the planner.** The LLM may only pick brick ids from the
  catalogue handed to it (`allowed_ids`). This kills most hallucination classes and makes
  every failure a *diagnosable* one the Medic can fix.
- **Deterministic codegen.** No LLM writes the final Python. Only the *spec* is LLM-authored;
  codegen is pure and unit-tested. Repair = spec patch, never free-text code editing.
- **Canvas owns `/`.** Non-negotiable given daggr's absolute asset paths (see §2).
- **HF Dataset as the database.** No server, free, git-auditable, forkable; votes and
  workflows are commits. Trade-off: no atomic counters — last-write-wins per voter file,
  which is fine at community scale.
- **In-process execution** via daggr's own `SequentialExecutor` (not a second server), so the
  canvas and the runner share one engine and one session store.
- **Avoid "Lego" in the product name** (trademark): the metaphor is "bricks".
- **A verification probe must never guess parameters.** An early probe passed `width=256`
  to FLUX, which rejected it, and the registry condemned a perfectly healthy brick as broken.
  False negatives are worse than no probe: the probe now omits anything with a default, only
  supplies the *subject* (prompt/image/audio), retries once, and declines when unsure.
- **Tests must not depend on brick health.** The first test suite named specific bricks; the
  moment one legitimately broke, tests failed for reasons unrelated to the code. Fixtures now
  select healthy bricks from the registry at test time — a rule worth repeating in any project
  that consumes live third-party APIs.
- **A missing required parameter is blocking, not a warning.** daggr refuses to build such a
  node, so the Medic must fix it (wire a literal/edge/input) or pick another brick.
- **Probe cheap modalities only.** video/3D are never executed during a registry refresh
  (anti-social and slow); they are marked "introspected only" rather than implied-working.

## 7. Future roadmap (do NOT execute)

- Agent API for the Space (`/api/plan`, `/api/heal`) so other agents can build pipelines.
- Auto-publish a nightly liveness job that marks retired bricks and opens PRs to workflows.
- Per-brick cost/latency telemetry from real runs; leaderboard sort by actual speed.
- Workflow forking/diffing; "remix" a leaderboard entry.
- Optional local-GPU brick class (ComfyUI endpoints) for BYOC users.

## 8. Metadata

Authoritative document for `daggr-studio`. If code and this file disagree, fix one of them
in the same commit.
