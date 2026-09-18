---
title: Daggr Studio
emoji: 🧱
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Assemble daggr workflows from verified bricks
---

# 🧱 Daggr Studio

**Describe what you want. Get a running [daggr](https://github.com/gradio-app/daggr) workflow
built from bricks that were proven to work — and that repairs itself when the bricks change
underneath you.**

Daggr Studio turns "make me a game sprite with a transparent background" into a validated,
runnable daggr pipeline. Everything it assembles comes from a registry of Spaces and models
that were **introspected *and* executed** during verification — never guessed.

> The Studio lives at **`/`**. The real daggr canvas — every step's output, re-runnable node by
> node — lives at **`/canvas/`** (behind a prefix proxy, because daggr's frontend uses absolute
> asset paths). The original Gradio Builder is still at **`/builder/`**.

## What it actually does

| Step | What happens | Where the intelligence is |
|---|---|---|
| **Plan** | intent + filters → a workflow *spec* (JSON), choosing only from the verified catalogue | a small model (`deepseek-ai/DeepSeek-V4.1-Flash` by default), constrained to catalogue ids |
| **Complete** | endpoints, output ports and `postprocess` hints are filled in from the registry | deterministic |
| **Validate** | every endpoint and parameter is checked against the Space's **live** API | deterministic (`gradio_client.view_api`) |
| **Heal** | breakage is repaired: renamed parameters remapped, dead Spaces swapped, NC bricks replaced | deterministic fixes first, then a model that may only emit ops from a closed vocabulary |
| **Run** | executed in-process with daggr's own executor, node by node | daggr |
| **Share** | published as a commit to a public HF Dataset; leaderboard with modality / industry / licence filters | git |

**No LLM ever writes the generated code.** The spec is model-authored; `app.py` is produced by
deterministic, unit-tested codegen, so what you run in the Space and what you download are
the same thing by construction.

## The registry is the product

`daggrstudio/registry/spaces.json` is rebuilt by `scripts/verify_registry.py`:

* **introspected** — real endpoint names, real parameter names and types;
* **executed** — for cheap modalities a real call is made (the `probe` column). Introspection
  alone is not enough: `hf-applications/background-removal` introspects perfectly and fails
  every single call. It is marked broken in our registry, and workflows that used it get moved
  onto a working brick automatically.
* **licensed** — the licence is read from the Hub card and commercial usability is decided by
  the licence *text*, not by a hand-written flag.

Re-run it any time (it is resumable and polite to the Hub):

```bash
export HF_TOKEN=hf_...          # any token; probing public Spaces works anonymously too
python scripts/verify_registry.py --refresh --discover --probe
```

## Self-healing, concretely

* **A sister Space renamed a parameter** (`prompt` → `text`): the validator reports
  `PARAM_RENAMED` with the live parameter list, and the Medic rewires the binding — deterministically.
* **A sister Space changed its endpoint** (`/run` → `/image`): `API_NAME_INVALID`, repaired.
* **A sister Space died or was deleted**: `SPACE_UNREACHABLE`, failed over to a verified
  alternative of the same modality and licence class.
* **A sister Space went non-commercial**: your commercial-only workflow refuses it and swaps
  it for a commercially licensed brick.
* **A failure at run time**: sleeping Space → retry with backoff; GPU quota → retry with your
  token; interface change → live re-introspection and remap; deleted → brick failover.

Every repair is recorded in the workflow's `heal_log` and shown in the Builder, so you can see
exactly what your workflow became and why. Repairs the model *cannot* justify are rejected with
a reason rather than silently applied.

## Your token, your compute (BYOK)

* **Bring your own Hugging Face token** (Settings, session-only, never stored): planning,
  healing and publishing are then billed to *you* and commits are attributed to *you*.
* **Without a token** you can still explore, plan and heal using a small shared community pool
  with a hard daily cap (`DAGGRSTUDIO_POOL_DAILY_CAP`, default 40 calls). Browsing the
  leaderboard and the brick catalogue is always free and needs no token.
* **Executing workflows** calls public Spaces; some ZeroGPU Spaces will want a token (yours).

The Space is CPU-only by design — it orchestrates, it does not run GPU models.

## Sharing & the community leaderboard

Publishing commits one JSON file to
[`jkorstad/daggr-studio-workflows`](https://huggingface.co/datasets/jkorstad/daggr-studio-workflows):
no server, no database, fully auditable and forkable, and every entry keeps its provenance
(bricks used, planner model, licences, repair log). Votes are one-per-HF-user files. The
leaderboard filters by **modality, industry, licence posture and compute tier**, and defaults
to commercially usable workflows only.

Workflow *metadata* is shared under CC-BY-4.0. The assets your workflow produces keep the
licence posture you chose — Daggr Studio never relicenses your output.

## Architecture

One Space, one process, five surfaces — built on **`gradio.Server`** (Gradio ≥ 6), i.e. a custom
frontend with Gradio's backend (queue, streaming, MCP, `gradio_client`, Spaces hosting):

| Path | What | Why there |
|---|---|---|
| `/` | the Studio (custom single-page frontend: HTML/CSS/vanilla ES modules, no build step) | the front door; nothing about the UI is constrained by Gradio components |
| `/api/*` | JSON + NDJSON endpoints (`plan`, `plan/stream`, `validate`, `heal`, `run/stream`, `check`, `advise`, `publish`, `load`, `vote`, `bricks`, `deploy`, `registry/verify`, `artifact/<id>`) | one code path shared by the UI, `curl` agents and the queued endpoints |
| queued | `@app.api()` endpoints: `plan_workflow`, `validate_workflow`, `registry_summary` | callable from `gradio_client`/MCP like any Gradio app — deliberately **token-free**, because queued calls can appear in run history |
| `/canvas/` | the real daggr canvas (node-by-node inspection, re-run individual bricks) | it needs the prefix proxy (below) |
| `/builder/` | the original Gradio Builder | kept as a fallback surface |

### Why the canvas needs a proxy

daggr's prebuilt frontend requests `/assets/…`, `/theme.css`, `/api/…`, `/ws/…` and `/file/…`
as **absolute** paths. `PrefixProxy` (`daggrstudio/web/proxy.py`) strips `/canvas` on the way in
and rewrites those references on the way out — for HTML/JS/CSS only, never JSON or binary.

One thing rewriting **cannot** fix: daggr builds its websocket URL at runtime from
`window.location.host`. So the canvas page also receives a small shim that wraps
`WebSocket`/`fetch`/`EventSource` for same-origin requests. Two bugs lived here, both invisible
server-side and obvious in a browser: the shim compared `origin` (a `wss://` URL has scheme
`wss` on an `https` page, so nothing was ever prefixed), and the back-link was injected twice.

## Also useful without the UI

```bash
curl https://jkorstad-daggr-studio.hf.space/api/bricks?modality=image-edit
curl https://jkorstad-daggr-studio.hf.space/api/leaderboard?modality=image-gen&sort=votes
curl https://jkorstad-daggr-studio.hf.space/api/canvas          # what the canvas shows
curl https://jkorstad-daggr-studio.hf.space/healthz             # registry summary + canvas state

# intent -> validated, healed workflow (agent-native; token is optional and never stored)
curl -X POST https://jkorstad-daggr-studio.hf.space/api/plan \
  -H 'content-type: application/json' \
  -d '{"intent":"3D model of a game character from text","industry":"game-dev","max_steps":3}'

# streaming variant: NDJSON progress (stage events, then the full result)
curl -N -X POST https://jkorstad-daggr-studio.hf.space/api/plan/stream \
  -H 'content-type: application/json' -d '{"intent":"narrate a product blurb"}'

# run a workflow, streaming per-step events as each brick executes
curl -N -X POST https://jkorstad-daggr-studio.hf.space/api/run/stream \
  -H 'content-type: application/json' -d '{"spec":{...},"values":{"prompt":"a red cube"}}'

# validate / heal a spec you already have
curl -X POST https://jkorstad-daggr-studio.hf.space/api/validate \
  -H 'content-type: application/json' -d '{"spec":{...},"live_validation":false}'
curl -X POST https://jkorstad-daggr-studio.hf.space/api/heal \
  -H 'content-type: application/json' -d '{"spec":{...}}'
```

Interactive OpenAPI docs: `/docs`. The queued endpoints are also reachable through
`gradio_client`, e.g. `Client("jkorstad/daggr-studio").predict(intent="...", api_name="/plan_workflow")`.

Artifacts produced by a run are served as `/api/artifact/<opaque-id>` — never by path, so there
is no way to use this Space to read arbitrary files.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q                     # 121 tests, no network
.venv/bin/python scripts/smoke_run.py             # build + validate + RUN a real workflow
.venv/bin/python scripts/acceptance.py            # the full journey, stage by stage
python app.py                                     # /builder + / locally on :7860
```

Repo layout: `daggrstudio/` (spec, registry, codegen, validator, medic, runner, hub, web),
`scripts/` (registry verifier, smoke + acceptance tests), `tests/`, `docs/`.
`MASTER_PLAN.md` is the authoritative roadmap and records the design decisions.

## Honest limitations

* daggr is beta (0.8.0) and its scatter/gather is buggy; batches are looped inside a single
  node instead.
* Registry verification is rate-limited friendly but slow (≈1 call per brick per refresh),
  and a Space can break the minute after it was verified — that is what the Medic is for.
* Votes are last-write-wins (a vote file per workflow), a deliberate trade for zero infrastructure.
* The canvas shows one workflow at a time: the one you last built in the Builder.

MIT licensed. Built as a companion to the `daggr` and `daggr-pipelines` skill packs.
