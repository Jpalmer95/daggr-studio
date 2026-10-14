# AGENTS.md — Daggr Studio

Visual AI-workflow canvas: pick bricks, get a runnable daggr pipeline.
Repo `~/dev/daggr-studio` → GitHub `Jpalmer95/daggr-studio`.

## Deployed surfaces
- Space: `jkorstad/daggr-studio` (docker SDK, `cpu-basic`, no GPU)
- **Dataset: `jkorstad/daggr-studio-workflows` — this IS the database.** No server. Workflows,
  votes, and the leaderboard are commits to the shared dataset. Accepted trade-off: no atomic
  counters — last-write-wins per voter file, which is fine at community scale.

## Authoritative docs
`MASTER_PLAN.md` — vision, architecture, cost model, phases, and recorded design decisions.

## Design rules that must not be violated
- **Bricks, not free-form code.** The planner may only pick brick ids from the catalogue it is
  handed (`allowed_ids`). This kills most hallucination classes and makes every failure a
  *diagnosable* one.
- **Deterministic codegen.** No LLM writes the final Python — only the *spec* is LLM-authored.
  Repair is a spec patch, never free-text code editing.
- **Canvas owns `/`.** Non-negotiable, given daggr's absolute asset paths.
- **In-process execution** via daggr's `SequentialExecutor` — canvas and runner share one engine
  and one session store.
- Avoid "Lego" in the product name (trademark). The metaphor is **bricks**.

## Pitfalls already paid for
- **A verification probe must never guess parameters.** An early probe passed `width=256` to
  FLUX, which rejected it, and the registry condemned a perfectly healthy brick as broken.
  False negatives are worse than no probe: omit anything with a default, supply only the
  *subject*, retry once, and decline when unsure.
- **Tests must not depend on brick health.** Name specific bricks in a test and it fails for
  reasons unrelated to your change the moment one legitimately breaks. Fixtures select healthy
  bricks from the registry at test time.

## Cross-cutting context
`~/dev/knowledge` → `wiki/index.md`. The dataset-as-database choice is recorded as a house
pattern in `wiki/decisions/recurring-architecture-principles.md`.
