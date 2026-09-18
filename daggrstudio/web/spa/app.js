/* Daggr Studio frontend.
 *
 * Vanilla ES modules, no build step: the Space ships exactly these files, so what runs in the
 * browser is what is in the repo (and the Docker image stays simple).
 *
 * Structure:
 *   state      — one small object, persisted to localStorage (token stays in sessionStorage)
 *   api        — fetch wrappers, including NDJSON streaming readers for plan/run
 *   render*    — pure-ish functions that paint a panel from state
 *   handlers   — wired at the bottom
 */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const POLL_KEY = "daggr-studio:state:v1";

export const state = {
  spec: null,
  model: "deepseek-ai/DeepSeek-V4.1-Flash",
  username: "",
  results: {},          // stepId -> {status, seconds, repairs, error, artifacts:[{id,url,kind,name}]}
  health: null,
  running: false,
  upgradedSteps: 0,
};

/* ── token handling ─────────────────────────────────────────────────────────
 * The token never goes to localStorage (persistent) and never appears in a URL.
 * It lives in memory + sessionStorage so a refresh in the same tab keeps it. */
export function token() {
  return sessionStorage.getItem("daggr-studio:token") || "";
}
export function setToken(value, remember = true) {
  if (value) sessionStorage.setItem("daggr-studio:token", value);
  else sessionStorage.removeItem("daggr-studio:token");
  if (remember) void verifyToken(value);
}

export function saveState() {
  const { spec, model } = state;
  try { localStorage.setItem(POLL_KEY, JSON.stringify({ spec, model })); } catch { /* quota */ }
}
export function restoreState() {
  try {
    const raw = localStorage.getItem(POLL_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (parsed.spec) state.spec = parsed.spec;
    if (parsed.model) state.model = parsed.model;
  } catch { /* ignore corrupt state */ }
}

/* ── api ──────────────────────────────────────────────────────────────────── */

async function request(path, { method = "GET", body, timeout = 180000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(path, {
      method,
      headers: body ? { "content-type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch { data = { ok: false, message: text.slice(0, 400) }; }
    if (!res.ok && data && data.detail && !data.message) data.message = data.detail;
    return { status: res.status, ok: res.ok && data.ok !== false, data };
  } catch (err) {
    return { status: 0, ok: false, data: { ok: false, message: `${err.name}: ${err.message}` } };
  } finally {
    clearTimeout(timer);
  }
}

/** POST that yields NDJSON events as they arrive (used by plan and run). */
async function* stream(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    yield { stage: "error", message: text.slice(0, 400) || `HTTP ${res.status}` };
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 1);
      if (!line) continue;
      try { yield JSON.parse(line); } catch { /* skip malformed line */ }
    }
  }
  if (buffer.trim()) {
    try { yield JSON.parse(buffer.trim()); } catch { /* ignore */ }
  }
}

export const api = {
  health: () => request("/api/health"),
  models: () => request("/api/models"),
  plan: (body) => request("/api/plan", { method: "POST", body }),
  planStream: (body) => stream("/api/plan/stream", body),
  validate: (body) => request("/api/validate", { method: "POST", body }),
  heal: (body) => request("/api/heal", { method: "POST", body }),
  check: (body) => request("/api/check", { method: "POST", body }),
  advise: (body) => request("/api/advise", { method: "POST", body }),
  applyUpgrade: (body) => request("/api/apply_upgrade", { method: "POST", body }),
  runStream: (body) => stream("/api/run/stream", body),
  publish: (body) => request("/api/publish", { method: "POST", body }),
  load: (body) => request("/api/load", { method: "POST", body }),
  leaderboard: (params) => request(`/api/leaderboard?${new URLSearchParams(params)}`),
  vote: (body) => request("/api/vote", { method: "POST", body }),
  deploy: (body) => request("/api/deploy", { method: "POST", body, timeout: 300000 }),
  bricks: (params) => request(`/api/bricks?${new URLSearchParams(params)}`),
  verifyBricks: (body) => request("/api/registry/verify", { method: "POST", body, timeout: 300000 }),
  verifyToken: (body) => request("/api/verify_token", { method: "POST", body }),
};

/* ── ui helpers ───────────────────────────────────────────────────────────── */

export function toast(message, kind = "") {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast show ${kind}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.className = "toast"; }, 4200);
}

export function log(el, message, kind = "dim") {
  const target = typeof el === "string" ? $(el) : el;
  if (!target) return;
  const line = document.createElement("div");
  line.className = `l-${kind}`;
  line.textContent = message;
  target.appendChild(line);
  target.scrollTop = target.scrollHeight;
}

export function setLog(el, message, kind = "dim") {
  const target = typeof el === "string" ? $(el) : el;
  if (!target) return;
  target.innerHTML = "";
  log(target, message, kind);
}

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function table(el, rows, placeholder = "—") {
  const body = $(el).querySelector("tbody");
  if (!rows || !rows.length) {
    body.innerHTML = `<tr><td colspan="10" class="mono-sm">${esc(placeholder)}</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((row) => `<tr>${row.map((cell) => `<td>${esc(cell)}</td>`).join("")}</tr>`).join("");
}

/* ── status pill + identity ───────────────────────────────────────────────── */

export function setPill(el, text, kind = "") {
  const node = typeof el === "string" ? $(el) : el;
  if (!node) return;
  node.textContent = text;
  node.className = `pill ${kind}`;
}

async function refreshHealth() {
  const { data } = await api.health();
  if (!data || data.ok === false) return;
  state.health = data;
  setPill("#pool-pill", `pool ${data.pool.remaining}/${data.pool.cap}`,
    data.pool.remaining === 0 ? "warn" : "");
  $("#brand-status").textContent = data.counts ? `${data.counts.proven} proven bricks` : "";
  const canvasState = data.canvas?.ready
    ? `canvas ready${data.canvas.error ? " (error)" : ""}`
    : "canvas idle";
  $("#about-health").textContent =
    `${data.registry} · ${canvasState} · community pool ${data.pool.remaining}/${data.pool.cap} today`;
}

export async function verifyToken(value) {
  const status = $("#token-status");
  if (!value) {
    state.username = "";
    setPill("#user-pill", "not signed in");
    if (status) status.textContent = "No token: planning and healing use the shared pool.";
    return;
  }
  setPill("#user-pill", "checking…");
  const { data } = await api.verifyToken({ token: value });
  if (data.username) {
    state.username = data.username;
    setPill("#user-pill", data.username, "ok");
    if (status) status.textContent = `Signed in as ${data.username} — your requests are billed to you, and published commits are attributed to you.`;
  } else {
    state.username = "";
    setPill("#user-pill", "token rejected", "bad");
    if (status) status.textContent = "Hugging Face rejected that token.";
  }
}

/* ── workflow rendering ───────────────────────────────────────────────────── */

const OUTPUT_ICON = { image: "🖼", audio: "🎧", video: "🎬", model3d: "🧊", text: "📝", json: "{}" };

export function renderSummary() {
  const spec = state.spec;
  const el = $("#summary");
  const hint = $("#workflow-hint");
  if (!spec) {
    el.innerHTML = "";
    hint.textContent = "Nothing built yet — describe a goal on the left.";
    return;
  }
  const steps = spec.steps || [];
  hint.textContent = `${steps.length} bricks · industry ${spec.industry} · licence posture ${spec.license_posture} · planner ${spec.planner_model || "hand-built"}`;
  el.innerHTML = `
    <div style="font-size:19px;font-weight:600;margin-bottom:2px">${esc(spec.name)}</div>
    <div class="hint" style="margin:0 0 10px">${esc(spec.intent || "")}</div>`;
}

export function renderPipeline() {
  const spec = state.spec;
  const el = $("#pipeline");
  if (!spec) { el.innerHTML = ""; return; }
  el.innerHTML = "";
  (spec.steps || []).forEach((step, index) => {
    const run = state.results[step.id] || {};
    const cls = ["node", run.status || ""].join(" ").trim();
    const node = document.createElement("div");
    node.className = cls;
    node.dataset.step = step.id;
    const outKind = Object.values(step.outputs || {})[0]?.component || "textbox";
    const badge = run.status === "ok" ? '<span class="pill ok">ok</span>'
      : run.status === "repaired" ? '<span class="pill warn">self-repaired</span>'
      : run.status === "failed" ? '<span class="pill bad">failed</span>'
      : run.time ? `<span class="pill">${run.time}s</span>` : "";
    node.innerHTML = `
      <h4>${index + 1}. ${esc(step.title || step.id)}</h4>
      <div class="src">${esc(step.source || step.fn || "")}${step.api_name ? esc(step.api_name) : ""}</div>
      ${step.why ? `<div class="why">${esc(step.why)}</div>` : ""}
      <div class="meta">
        ${badge}
        <span class="pill">${OUTPUT_ICON[outKind] || "○"} ${esc(outKind)}</span>
        ${step.brick_id ? `<span class="pill">${esc(step.brick_id)}</span>` : ""}
      </div>
      ${(run.repairs || []).length ? `<div class="repairs">🔧 ${esc(run.repairs.join(" · "))}</div>` : ""}
      ${run.error ? `<div class="repairs" style="color:var(--bad)">${esc(run.error)}</div>` : ""}
      <div class="out" data-out="${esc(step.id)}"></div>`;
    if (index < (spec.steps || []).length - 1) {
      const arrow = document.createElement("div");
      arrow.className = "arrow";
      arrow.textContent = "→";
      node.appendChild(arrow);
    }
    el.appendChild(node);
  });
}

export function renderArtifacts(items, container = "#artifacts") {
  const el = $(container);
  if (!items || !items.length) {
    el.innerHTML = '<span class="mono-sm">Nothing yet.</span>';
    return;
  }
  el.innerHTML = items.map((art) => {
    const src = art.url || art.path;
    if (art.kind === "image") return `<div class="artifact"><img src="${src}" alt="${esc(art.name)}" loading="lazy"><div class="name">${esc(art.name)}</div></div>`;
    if (art.kind === "video") return `<div class="artifact"><video src="${src}" controls></video><div class="name">${esc(art.name)}</div></div>`;
    if (art.kind === "audio") return `<div class="artifact"><audio src="${src}" controls></audio><div class="name">${esc(art.name)}</div></div>`;
    return `<div class="artifact"><a href="${src}" target="_blank" rel="noopener">${esc(art.name)}</a><div class="name">${esc(art.kind)}</div></div>`;
  }).join("");
}

export function renderRunInputs() {
  const el = $("#run-inputs");
  const spec = state.spec;
  el.innerHTML = "";
  if (!spec) {
    el.innerHTML = '<span class="mono-sm">Build a workflow first.</span>';
    return;
  }
  (spec.inputs || []).forEach((port) => {
    const wrap = document.createElement("div");
    const id = `in-${port.port}`;
    const label = port.label || port.port;
    if (port.component === "dropdown") {
      wrap.innerHTML = `<label for="${id}">${esc(label)}</label>
        <select id="${id}" data-port="${esc(port.port)}">${(port.choices || []).map((c) => `<option>${esc(c)}</option>`).join("")}</select>`;
    } else if (port.component === "slider" || port.component === "number") {
      wrap.innerHTML = `<label for="${id}">${esc(label)}</label>
        <input type="number" id="${id}" data-port="${esc(port.port)}" value="${esc(port.default ?? 0)}">`;
    } else {
      wrap.innerHTML = `<label for="${id}">${esc(label)}</label>
        <textarea id="${id}" data-port="${esc(port.port)}" rows="3">${esc(port.default ?? "")}</textarea>`;
    }
    el.appendChild(wrap);
  });
  if (!(spec.inputs || []).length) {
    el.innerHTML = '<span class="mono-sm">This workflow takes no inputs.</span>';
  }
}

export function collectRunValues() {
  const values = {};
  $$("#run-inputs [data-port]").forEach((el) => { values[el.dataset.port] = el.value; });
  return values;
}

export function renderStepTable() {
  const rows = Object.entries(state.results).map(([stepId, run]) => {
    const icon = run.status === "ok" ? "✅" : run.status === "repaired" ? "🔧"
      : run.status === "failed" ? "⛔" : "•";
    return [icon, run.title || stepId, run.time ? `${run.time}s` : "", run.brick || "",
      (run.repairs || []).join(" · ") || run.error || ""];
  });
  table("#step-table", rows);
}

export function renderIssues(issues) {
  const rows = (issues || []).map((issue) => {
    const icon = { blocking: "⛔", warning: "⚠️", info: "ℹ️" }[issue.severity] || "•";
    return [icon, issue.code, issue.step_id || "", issue.message || "", issue.fix_hint || ""];
  });
  table("#issues-table", rows, "No workflows validated yet.");
}

/* ═══ interactions ═══════════════════════════════════════════════════════════ */

/* Daggr Studio frontend — interactions.
 *
 * Everything user-triggered lives here so app.js stays a rendering library and this file
 * stays a list of intents: plan, heal, check, run, publish, load, vote, verify.
 */


const EXAMPLES = [
  "turn a one-line character idea into a game sprite with a transparent background",
  "make a 3D model of a game character from a text description",
  "write a product blurb and narrate it as a voiceover",
  "generate album artwork from a short lyric idea",
  "remove the background from a product photo",
  "write a social caption and generate a matching image",
];

/* ── tabs ─────────────────────────────────────────────────────────────────── */

function showPanel(name) {
  $$(".panel").forEach((panel) => panel.dataset.active = String(panel.dataset.panel === name));
  $$(".tab").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.panel === name)));
  location.hash = name;
}

/* ── build ────────────────────────────────────────────────────────────────── */

function busy(on, label = "working…") {
  state.running = on;
  const button = $("#plan");
  button.disabled = on;
  button.innerHTML = on ? `<span class="spinner"></span> ${label}` : "🩺 Plan, validate & heal";
}

async function plan() {
  const intent = $("#intent").value.trim();
  if (!intent) { toast("Describe what you want first", "bad"); return; }

  busy(true, "planning…");
  setLog("#timeline", "planning…", "info");
  renderIssues([]);

  const body = {
    intent,
    industry: $("#industry").value,
    license_posture: $("#license").value,
    max_steps: Number($("#max-steps").value),
    live_validation: $("#live-check").value === "1",
    model: $("#model").value.trim() || state.model,
    token: token() || null,
  };

  let final = null;
  try {
    for await (const event of api.planStream(body)) {
      if (event.stage === "planning") log("#timeline", "· asking the model to assemble bricks", "info");
      else if (event.stage === "workflow") log("#timeline", `· brick ${event.step}`, "dim");
      else if (event.stage === "healing") {
        log("#timeline", `· validating against live Spaces → ${event.message}`, "info");
        if (event.timeline) setLog("#timeline", event.timeline.replace(/\*\*/g, ""));
        renderIssues(event.issues || []);
      } else if (event.stage === "done") final = event.result;
      else if (event.stage === "error") toast(event.message, "bad");
    }
  } catch (err) {
    log("#timeline", `stream failed: ${err.message}`, "bad");
  }

  busy(false);
  if (!final) { toast("planning failed", "bad"); return; }

  state.spec = final.spec || null;
  if (final.model) state.model = final.model;
  state.results = {};
  renderAll();

  if (final.ok) {
    toast("workflow is valid and ready to run", "ok");
    showPanel("run");
  } else {
    toast(final.message || "the workflow needs attention", "bad");
    log("#timeline", final.message || "blocking findings remain", "bad");
  }
  saveState();
  void refreshCanvasNote(final.canvas);
}

async function refreshCanvasNote(canvas) {
  if (!canvas) return;
  log("#action-log", `canvas: ${canvas}`, "dim");
  $("#action-log").style.display = "block";
  const { data } = await api.health();
  if (data?.canvas) {
    $("#canvas-link").title = data.canvas.ready
      ? `Canvas: ${data.canvas.status}` : "Canvas: nothing pushed yet";
  }
}

async function revalidate() {
  if (!state.spec) { toast("nothing to validate", "bad"); return; }
  setLog("#action-log", "re-validating…", "info");
  $("#action-log").style.display = "block";
  const { data } = await api.validate({
    spec: state.spec, live_validation: $("#live-check").value === "1", token: token() || null,
  });
  renderIssues(data.issues || []);
  log("#action-log", data.message || data.codes?.join(",") || "", data.ok ? "ok" : "bad");
  toast(data.ok ? "all checks passed" : "findings above", data.ok ? "ok" : "bad");
}

async function heal() {
  if (!state.spec) { toast("nothing to heal", "bad"); return; }
  setLog("#action-log", "healing…", "info");
  $("#action-log").style.display = "block";
  const { data } = await api.heal({
    spec: state.spec, token: token() || null, model: $("#model").value.trim() || state.model,
    live_validation: $("#live-check").value === "1",
  });
  if (data.spec) state.spec = data.spec;
  renderIssues(data.issues || []);
  if (data.timeline) setLog("#timeline", data.timeline.replace(/\*\*/g, ""));
  log("#action-log", `status: ${data.status} — ${data.message || ""}`, data.ok ? "ok" : "warn");
  renderAll();
  saveState();
}

async function checkBricks() {
  if (!state.spec) { toast("build a workflow first", "bad"); return; }
  setLog("#action-log", "re-reading the live API of every Space this workflow uses…", "info");
  $("#action-log").style.display = "block";
  const { data } = await api.check({ spec: state.spec, token: token() || null });
  (data.rows || []).forEach((row) => log("#action-log", `${row[0]} ${row[1]} ${row[2]} — ${row[3]} ${row[4] || ""}`,
    row[0] === "✅" ? "ok" : "warn"));
  toast(data.ok ? "nothing changed upstream" : "upstream changed — heal to repair", data.ok ? "ok" : "warn");
}

async function advise() {
  if (!state.spec) { toast("build a workflow first", "bad"); return; }
  toast("asking the advisor…");
  const { data } = await api.advise({
    spec: state.spec, token: token() || null, model: $("#model").value.trim() || state.model,
  });
  const box = $("#suggestions");
  const list = data.suggestions || [];
  if (!list.length) { box.innerHTML = `<div class="hint">${data.message || "nothing to suggest"}</div>`; return; }
  box.innerHTML = list.map((s) => `
    <div class="card" style="padding:11px 13px;margin-bottom:8px">
      <div style="font-size:13.5px"><strong>${s.step}</strong> → <code>${s.brick_id}</code></div>
      <div class="hint" style="margin:4px 0 8px">${s.why || ""}</div>
      <button class="btn" data-apply="${s.step}|${s.brick_id}">Apply this upgrade</button>
    </div>`).join("");
  $$("#suggestions [data-apply]").forEach((button) => {
    button.addEventListener("click", async () => {
      const [step, brickId] = button.dataset.apply.split("|");
      const { data: applied } = await api.applyUpgrade({
        spec: state.spec, step, brick_id: brickId, token: token() || null,
      });
      if (applied.ok) {
        state.spec = applied.spec;
        state.upgradedSteps += 1;
        renderAll(); saveState();
        toast(applied.message || "upgraded", "ok");
        $("#suggestions").innerHTML = "";
      } else {
        toast(applied.message || "that upgrade was refused", "bad");
      }
    });
  });
}

/* ── run ──────────────────────────────────────────────────────────────────── */

async function run() {
  if (!state.spec) { toast("build a workflow first", "bad"); return; }
  if (state.running) return;

  const stop = new AbortController();
  $("#run").disabled = true;
  $("#run-stop").disabled = false;
  setLog("#run-log", "starting…", "info");
  renderArtifacts([]);
  state.results = {};
  renderPipeline();
  renderStepTable();

  let artifacts = [];
  try {
    for await (const event of api.runStream({
      spec: state.spec, values: collectRunValues(), token: token() || null,
    })) {
      if (event.stage === "started") log("#run-log", event.message, "info");
      else if (event.stage === "heartbeat") { /* keeps the connection warm */ }
      else if (event.stage === "step") {
        state.results[event.step_id] = {
          status: event.status, seconds: event.seconds, repairs: event.repairs,
          error: event.error, brick: event.brick_id, title: event.title,
        };
        const icon = event.status === "ok" ? "✅" : event.status === "repaired" ? "🔧" : "⛔";
        log("#run-log", `${icon} ${event.title || event.step_id} (${event.seconds}s)`
          + (event.repairs?.length ? ` — ${event.repairs.join(" · ")}` : "")
          + (event.error ? ` — ${event.error}` : ""),
          event.status === "ok" ? "ok" : event.status === "repaired" ? "warn" : "bad");
        renderPipeline();
        renderStepTable();
      } else if (event.stage === "done") {
        artifacts = event.result?.artifacts || [];
        log("#run-log", event.result?.message || "", event.result?.ok ? "ok" : "bad");
        if (event.result?.note) log("#run-log", event.result.note, "warn");
        renderArtifacts(artifacts);
        Object.entries(state.results).forEach(([stepId, runState]) => {
          const mine = artifacts.filter((a) => (a.name || "").length && runState.specStep === stepId);
          void mine;
        });
        toast(event.result?.ok ? "run finished" : "run finished with failures",
          event.result?.ok ? "ok" : "bad");
      } else if (event.stage === "error") {
        log("#run-log", event.message, "bad");
      }
    }
  } catch (err) {
    log("#run-log", `stream failed: ${err.message}`, "bad");
  } finally {
    $("#run").disabled = false;
    $("#run-stop").disabled = true;
    stop.abort();
    renderPipeline();
  }
}

/* ── code ─────────────────────────────────────────────────────────────────── */

function renderCode(code) {
  $("#code-view").textContent = code || "Build a workflow to see its code.";
}

function download(name, text, type = "text/plain") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

async function deploy() {
  if (!state.spec) { toast("build a workflow first", "bad"); return; }
  if (!token()) { toast("deploying needs your own token (Settings)", "bad"); return; }
  setLog("#deploy-status", "creating your Space…", "info");
  const { data } = await api.deploy({
    spec: state.spec, token: token(), name: $("#deploy-name")?.value || "",
  });
  setLog("#deploy-status", data.message || "", data.ok ? "ok" : "bad");
  if (data.url) {
    $("#deploy-status").innerHTML += ` <a href="${data.url}" target="_blank" rel="noopener">open it ↗</a>`;
  }
}

/* ── share ────────────────────────────────────────────────────────────────── */

async function publish() {
  if (!state.spec) { toast("build a workflow first", "bad"); return; }
  setLog("#share-status", "publishing…", "info");
  const { data } = await api.publish({
    spec: state.spec, token: token() || null, note: $("#publish-note").value,
  });
  setLog("#share-status", data.message || "", data.ok ? "ok" : "warn");
  if (data.url) $("#share-status").innerHTML += ` <a href="${data.url}" target="_blank" rel="noopener">view commit ↗</a>`;
  if (data.ok) { toast("published", "ok"); void refreshLeaderboard(); }
}

async function loadWorkflow() {
  const slug = $("#load-slug").value;
  if (!slug) { toast("pick a workflow", "bad"); return; }
  setLog("#load-status", `loading ${slug}…`, "info");
  const { data } = await api.load({ slug, token: token() || null });
  if (!data.ok) { setLog("#load-status", data.message || "not found", "bad"); return; }
  state.spec = data.spec;
  state.results = {};
  renderAll(); renderCode(data.code); saveState();
  setLog("#load-status", data.message || `loaded ${slug}`, "ok");
  showPanel("build");
}

async function refreshLeaderboard() {
  const params = {
    modality: $("#lb-modality").value || "any",
    industry: $("#lb-industry").value || "any",
    license_filter: $("#lb-license").value,
    compute_tier: $("#lb-tier").value,
    sort: $("#lb-sort").value,
  };
  const { data } = await api.leaderboard(params);
  $("#lb-summary").innerHTML = data.message || "";
  table("#lb-table", (data.rows || []).map((row) => row.slice(0, 8)), "No workflows match those filters yet.");
  const slugs = (data.slugs || []).filter(Boolean);
  const select = $("#load-slug");
  const previous = select.value;
  select.innerHTML = slugs.map((s) => `<option${s === previous ? " selected" : ""}>${s}</option>`).join("")
    || "<option value=''>— nothing published yet —</option>";
}

async function vote(direction) {
  const slug = $("#load-slug").value;
  if (!slug) { toast("pick a workflow", "bad"); return; }
  const { data } = await api.vote({ slug, direction, token: token() || null });
  toast(data.message || "", data.ok ? "ok" : "warn");
  if (data.ok) void refreshLeaderboard();
}

/* ── bricks ───────────────────────────────────────────────────────────────── */

async function refreshBricks() {
  const { data } = await api.bricks({
    modality: $("#b-modality").value || "any",
    commercial_only: $("#b-commercial").value === "1",
    reachable_only: $("#b-reachable").value === "1",
    limit: 80,
  });
  $("#b-summary").textContent = data.summary || "";
  table("#brick-table", data.rows || [], "No bricks match those filters.");
  const select = $("#b-modality");
  if (select.options.length <= 1 && data.modalities) {
    select.innerHTML = data.modalities.map((m) => `<option>${m}</option>`).join("");
    select.value = "any";
  }
}

async function verifyWorkflowBricks() {
  if (!state.spec) { toast("build a workflow first", "bad"); return; }
  const ids = [...new Set((state.spec.steps || []).map((s) => s.brick_id).filter(Boolean))].slice(0, 6);
  if (!ids.length) { toast("this workflow uses no verified bricks", "bad"); return; }
  $("#verify-status").style.display = "block";
  setLog("#verify-status", `making one real call to each of: ${ids.join(", ")} (slow, be patient)…`, "info");
  const { data } = await api.verifyBricks({ ids, spec: state.spec, token: token() || null });
  (data.bricks || []).forEach((row) => {
    log("#verify-status", `${row.ok ? "✅" : "⛔"} ${row.id} ${row.api_name || ""} ${row.error || ""}`,
      row.ok ? "ok" : "bad");
  });
  if (data.findings?.length) {
    renderIssues(data.findings);
    log("#verify-status", `${data.findings.length} finding(s) — see the Build tab`, "warn");
  }
  toast(data.bricks?.every((b) => b.ok) ? "every brick in this workflow still works" : "some bricks need repair",
    data.bricks?.every((b) => b.ok) ? "ok" : "warn");
}

/* ── render everything ────────────────────────────────────────────────────── */

function renderAll() {
  renderSummary();
  renderPipeline();
  renderRunInputs();
  renderStepTable();
}

/* ── boot ─────────────────────────────────────────────────────────────────── */

async function boot() {
  $("#examples").innerHTML = EXAMPLES.map((e) => `<span class="chip">${e}</span>`).join("");
  $$("#examples .chip").forEach((chip) => chip.addEventListener("click", () => {
    $("#intent").value = chip.textContent;
  }));

  $("#max-steps").addEventListener("input", (e) => { $("#max-steps-out").textContent = e.target.value; });
  $("#tabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (tab) showPanel(tab.dataset.panel);
  });

  $("#plan").addEventListener("click", plan);
  $("#revalidate").addEventListener("click", revalidate);
  $("#heal").addEventListener("click", heal);
  $("#check").addEventListener("click", checkBricks);
  $("#advise").addEventListener("click", advise);
  $("#run").addEventListener("click", run);
  $("#run-stop").addEventListener("click", () => location.reload());
  $("#deploy").addEventListener("click", deploy);
  $("#publish").addEventListener("click", publish);
  $("#load").addEventListener("click", loadWorkflow);
  $("#vote-up").addEventListener("click", () => vote(1));
  $("#vote-down").addEventListener("click", () => vote(-1));
  $("#lb-refresh").addEventListener("click", refreshLeaderboard);
  $("#b-refresh").addEventListener("click", refreshBricks);
  $("#verify-bricks").addEventListener("click", verifyWorkflowBricks);
  $("#load-recent").addEventListener("click", refreshLeaderboard);

  $("#settings-toggle").addEventListener("click", () => $("#settings").classList.toggle("hidden"));
  $("#verify-token").addEventListener("click", () => {
    setToken($("#token").value.trim());
    toast("token saved for this tab", "ok");
  });
  $("#clear-token").addEventListener("click", () => {
    setToken(""); $("#token").value = "";
    toast("token forgotten");
  });

  $("#download-code").addEventListener("click", () => {
    download(`${(state.spec?.slug) || "workflow"}_app.py`, $("#code-view").textContent);
  });
  $("#download-spec").addEventListener("click", () => {
    download(`${(state.spec?.slug) || "workflow"}.json`, JSON.stringify(state.spec, null, 2), "application/json");
  });
  $("#copy-code").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("#code-view").textContent);
    toast("copied", "ok");
  });
  $("#new").addEventListener("click", () => {
    state.spec = null; state.results = {};
    $("#intent").value = ""; renderAll(); renderCode(); renderIssues([]);
    setLog("#timeline", "No workflow yet."); saveState();
  });
  $("#import-btn").addEventListener("click", () => {
    const input = document.createElement("input");
    input.type = "file"; input.accept = ".json";
    input.onchange = async () => {
      const text = await input.files[0].text();
      try {
        const parsed = JSON.parse(text);
        state.spec = parsed.spec || parsed;
        renderAll(); saveState();
        toast("imported", "ok");
        showPanel("build");
      } catch { toast("that file is not valid JSON", "bad"); }
    };
    input.click();
  });

  // keyboard: ctrl/cmd+enter plans, alt+enter runs
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); void plan(); }
    if (e.altKey && e.key === "Enter") { e.preventDefault(); void run(); }
  });

  // restore a previous session, then fill the panels from the server
  restoreState();
  if (state.spec) {
    renderAll();
    toast("restored your last workflow from this browser");
  }
  setToken(token(), false);
  $("#token").value = token() ? "•••saved•••" : "";
  $("#model").value = state.model;

  await refreshHealth();
  const { data: models } = await api.models();
  if (models?.models) {
    $("#model-list").innerHTML = models.models.map((m) => `<option value="${m}">`).join("");
    if (models.default && !state.model) { state.model = models.default; $("#model").value = models.default; }
  }

  const hash = (location.hash || "").replace("#", "");
  showPanel(["build", "run", "code", "share", "bricks", "about"].includes(hash) ? hash : "build");

  await Promise.all([refreshBricks(), refreshLeaderboard()]);
  if (state.spec) {
    const { data } = await api.validate({ spec: state.spec, live_validation: false });
    renderCode(state.spec ? "" : "");
    void data;
  }
  window.__daggrStudio = { state, api };   // handy for debugging from the console
  console.info("Daggr Studio ready");
}

boot().catch((err) => {
  document.body.insertAdjacentHTML("beforeend",
    `<div class="toast show bad">startup failed: ${err.message}</div>`);
});
