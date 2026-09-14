"""
The Builder UI (Gradio), mounted at ``/builder``.

Layout follows the actual journey: describe a goal → see the workflow and how it was
repaired → run it → share it. The canvas at ``/`` mirrors whatever is built here.

The dynamic Run tab renders one component per declared input, so a workflow that needs a
prompt and a voice line gets exactly those fields and nothing else.
"""

from __future__ import annotations

import gradio as gr

from daggrstudio.registry.bricks import get_registry
from daggrstudio.web import services
from daggrstudio.web.canvas import show_spec

INDUSTRIES = ["general", "game-dev", "music-production", "art", "3d", "film", "marketing"]
LICENSES = ["commercial-only", "any"]
SORTS = ["votes", "recent", "steps", "name"]

INTRO = """
# 🧱 Daggr Studio

Describe what you want to build and this Space assembles a **[daggr](https://github.com/gradio-app/daggr)**
workflow from *verified bricks* — then validates it against the live Space APIs, **self-heals**
anything that broke, and runs it.

**The canvas lives at [`/`](/) (this page's root)** — open it any time to inspect every step's
output. Sharing, voting and the community leaderboard are on the **Share** tab.

*Nothing here is written by an LLM except the workflow plan: the code you see is generated
deterministically, and every brick in it was introspected and executed before it was offered.*
"""


def _component_for(port: dict):
    """Mirror of codegen.make_input_component, for the dynamic Run tab."""
    label = port.get("label") or port.get("port", "").replace("_", " ").title()
    kind = (port.get("component") or "textbox").lower()
    default = port.get("default", "")
    if kind == "image":
        return gr.Image(label=label)
    if kind == "audio":
        return gr.Audio(label=label)
    if kind == "video":
        return gr.Video(label=label)
    if kind == "model3d":
        return gr.Model3D(label=label)
    if kind == "number":
        return gr.Number(label=label, value=default if default not in ("", None) else 0)
    if kind == "slider":
        return gr.Slider(minimum=port.get("minimum") or 0, maximum=port.get("maximum") or 100,
                         value=default if default not in ("", None) else 0,
                         step=port.get("step") or 1, label=label)
    if kind == "dropdown":
        choices = port.get("choices") or ["auto"]
        return gr.Dropdown(choices=choices, value=default or choices[0], label=label)
    if kind == "checkbox":
        return gr.Checkbox(label=label, value=bool(default))
    if kind == "json":
        return gr.JSON(label=label)
    if kind == "file":
        return gr.File(label=label)
    return gr.Textbox(label=label, value=default if default is not None else "", lines=3)


def _port_names(spec: dict | None) -> list[str]:
    return [p.get("port", "") for p in (spec or {}).get("inputs", [])]


def build_ui() -> gr.Blocks:
    registry = get_registry()
    modalities = ["any"] + registry.modalities_present()

    with gr.Blocks(title="Daggr Studio", fill_height=True) as demo:
        gr.Markdown(INTRO)

        spec_state = gr.State(None)
        token_state = gr.State("")

        with gr.Accordion("⚙️ Settings — your Hugging Face token (used for your own requests)",
                          open=False):
            with gr.Row():
                token_box = gr.Textbox(label="HF token", type="password", scale=4,
                                       info="Session-only, never stored. With it, planning and "
                                            "healing are billed to you (BYOK). Without it you "
                                            "use the shared community pool.")
                verify_btn = gr.Button("Save & verify", size="sm", scale=1)
            token_status = gr.Markdown()
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Planner / Medic model", choices=services.list_models_for_ui(),
                    value=services.default_model(), allow_custom_value=True, scale=2,
                    info="Any model id served by HF Inference Providers")
                pool_md = gr.Markdown()

        with gr.Tabs():
            # ── build ────────────────────────────────────────────────────────────
            with gr.Tab("1 · Build"):
                with gr.Row():
                    intent = gr.Textbox(label="What do you want to build?", lines=4, scale=4,
                                        placeholder="e.g. turn a one-line character idea into a "
                                                    "game sprite with a transparent background")
                    starter = gr.Dropdown(label="…or start from an example",
                                          choices=list(services.STARTERS), value=None, scale=2)
                with gr.Row():
                    industry = gr.Dropdown(choices=INDUSTRIES, value="game-dev", label="Industry")
                    license_posture = gr.Dropdown(choices=LICENSES, value="commercial-only",
                                                  label="Licence posture",
                                                  info="commercial-only blocks NC bricks")
                    max_steps = gr.Slider(2, 8, value=4, step=1, label="Max bricks")
                    live_check = gr.Checkbox(value=True, label="Validate against live Spaces")
                with gr.Row():
                    plan_btn = gr.Button("🩺 Plan, validate & heal", variant="primary")
                    revalidate_btn = gr.Button("Re-validate")
                    heal_btn = gr.Button("Heal again")
                    canvas_btn = gr.Button("↗ Push to canvas")
                plan_status = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=3):
                        spec_md = gr.Markdown(label="Workflow")
                    with gr.Column(scale=4):
                        issues_df = gr.Dataframe(headers=services.ISSUE_HEADERS, wrap=True,
                                                 label="Validation findings", interactive=False)
                timeline_md = gr.Markdown(label="Repair log")
                with gr.Accordion("Upgrade advisor — newer or better bricks", open=False):
                    advise_btn = gr.Button("Suggest upgrades", size="sm")
                    advise_md = gr.Markdown()
                    with gr.Row():
                        upgrade_step = gr.Dropdown(label="Step", choices=[], allow_custom_value=True)
                        upgrade_brick = gr.Dropdown(label="Brick", choices=[], allow_custom_value=True)
                        apply_upgrade_btn = gr.Button("Apply upgrade", size="sm")

            # ── code ─────────────────────────────────────────────────────────────
            with gr.Tab("2 · Code & export"):
                gr.Markdown("This is real daggr code — deterministic codegen, no model in the "
                            "loop. Run it locally with `pip install daggr && daggr app.py`, "
                            "or deploy it as your own Space from the Share tab.")
                code_box = gr.Code(label="app.py", language="python", lines=26)
                with gr.Row():
                    code_btn = gr.Button("Show code for the current workflow")
                    spec_json = gr.Code(label="workflow spec (JSON)", language="json", lines=10)
                with gr.Row():
                    import_box = gr.Textbox(label="Paste a workflow spec or published entry to "
                                                "import", lines=4, scale=4)
                    import_btn = gr.Button("Import", scale=1)

            # ── run ──────────────────────────────────────────────────────────────
            with gr.Tab("3 · Run") as run_tab:
                gr.Markdown("Inputs are rendered from the workflow itself. Running executes the "
                            "real Spaces — a sleeping brick can take a minute on the first call.")
                run_inputs_row = gr.Row()
                run_btn = gr.Button("▶ Run workflow", variant="primary")
                run_status = gr.Markdown()
                run_rows = gr.Dataframe(headers=services.RUN_HEADERS, wrap=True,
                                        label="Step results", interactive=False)
                run_gallery = gr.Gallery(label="Artifacts produced", columns=4, height=320)

            # ── share ────────────────────────────────────────────────────────────
            with gr.Tab("4 · Share & leaderboard"):
                gr.Markdown(
                    "Publishing commits your workflow to a public Hugging Face Dataset — no "
                    "server involved, fully auditable. **You need your own token in Settings "
                    "so the commit is attributed to you.** Workflow metadata is shared under "
                    "CC-BY-4.0; the assets your workflow produces keep the licence you chose.")
                with gr.Row():
                    publish_note = gr.Textbox(label="Note (optional)", scale=4)
                    publish_btn = gr.Button("⬆ Publish to leaderboard", variant="primary")
                share_status = gr.Markdown()
                with gr.Row():
                    publish_url = gr.Markdown()
                    deploy_btn = gr.Button("🚀 Deploy as my own Space (gets the real canvas)")
                deploy_status = gr.Markdown()
                gr.Markdown("---")
                with gr.Row():
                    lb_modality = gr.Dropdown(choices=modalities, value="any", label="Modality")
                    lb_industry = gr.Dropdown(choices=["any"] + INDUSTRIES, value="any",
                                              label="Industry")
                    lb_license = gr.Dropdown(choices=LICENSES + ["any"], value="commercial-only",
                                             label="Licence")
                    lb_tier = gr.Dropdown(choices=["any", "cloud-free", "cloud-paid"], value="any",
                                          label="Compute")
                    lb_sort = gr.Dropdown(choices=SORTS, value="votes", label="Sort")
                lb_btn = gr.Button("🔄 Refresh leaderboard")
                lb_md = gr.Markdown()
                lb_df = gr.Dataframe(headers=services.LEADERBOARD_HEADERS, wrap=True,
                                     label="Community workflows", interactive=False)
                with gr.Row():
                    load_slug = gr.Dropdown(label="Load a workflow", choices=[], allow_custom_value=True)
                    load_btn = gr.Button("Load", size="sm")
                    vote_slug = gr.Dropdown(label="Vote for", choices=[], allow_custom_value=True)
                    vote_up = gr.Button("👍", size="sm")
                    vote_down = gr.Button("👎", size="sm")

            # ── bricks ───────────────────────────────────────────────────────────
            with gr.Tab("5 · Bricks"):
                gr.Markdown("The catalogue this Space is allowed to build with. `status` is live "
                            "reachability; `probe` means the brick was **actually executed** "
                            "during verification.")
                with gr.Row():
                    brick_modality = gr.Dropdown(choices=modalities, value="any", label="Modality")
                    brick_commercial = gr.Checkbox(value=True, label="Commercial only")
                    brick_running = gr.Checkbox(value=False, label="Reachable only")
                    bricks_btn = gr.Button("🔄 Refresh table")
                brick_md = gr.Markdown()
                brick_df = gr.Dataframe(headers=services.BRICK_HEADERS, wrap=True,
                                        label="Verified bricks", interactive=False)
                check_btn = gr.Button("🔍 Re-check the bricks this workflow uses")
                check_md = gr.Markdown()
                check_df = gr.Dataframe(headers=["", "space", "endpoint", "state",
                                                 "live parameters"], wrap=True,
                                        label="Live re-check", interactive=False)

        # ── wiring ───────────────────────────────────────────────────────────────
        def _on_verify(token):
            info = services.settings_info(token)
            if info["token_ok"]:
                msg = (f"✅ token verified for **{info['username']}** — your requests are billed "
                       f"to your account. Pool: {info['pool']['remaining']}/"
                       f"{info['pool']['cap']} left today.")
            elif token:
                msg = "❌ Hugging Face rejected that token."
            else:
                msg = ("No token set. You can still plan and heal using the shared pool "
                       f"({info['pool']['remaining']}/{info['pool']['cap']} calls left today), "
                       "but publishing and voting need your own token.")
            return token or "", msg, f"**Bricks:** {info['registry']}"

        verify_btn.click(_on_verify, inputs=[token_box],
                         outputs=[token_state, token_status, pool_md])
        demo.load(lambda: _on_verify(""), outputs=[token_state, token_status, pool_md])

        def _on_starter(choice):
            return services.STARTERS.get(choice or "", "")

        starter.change(_on_starter, inputs=[starter], outputs=[intent])

        def _plan(intent_text, industry_v, license_v, steps_v, live_v, token_v, model_v):
            out = services.plan_workflow(intent_text, token=token_v or None, model=model_v or None,
                                         industry=industry_v, license_posture=license_v,
                                         max_steps=int(steps_v), live_validation=live_v)
            spec = out.get("spec")
            canvas_note = show_spec(services.as_spec(spec)) if spec else ""
            steps = [s.get("id", "") for s in (spec or {}).get("steps", [])]
            bricks = sorted({s.get("brick_id") for s in (spec or {}).get("steps", [])
                             if s.get("brick_id")})
            status = (f"**{out['status']}** — {out['message']}\n\n"
                      f"`model: {out.get('model') or 'n/a'}`\n\n**canvas:** {canvas_note}")
            if out.get("notes"):
                status += "\n\n" + "\n".join(f"- {n}" for n in out["notes"][:8])
            return (spec, status, out.get("summary", ""), out.get("issues", []),
                    out.get("timeline", ""), out.get("code", ""),
                    gr.update(choices=steps), gr.update(choices=bricks))

        plan_outputs = [spec_state, plan_status, spec_md, issues_df, timeline_md, code_box,
                        upgrade_step, upgrade_brick]
        plan_btn.click(_plan,
                       inputs=[intent, industry, license_posture, max_steps, live_check,
                               token_state, model_dropdown],
                       outputs=plan_outputs)

        def _revalidate(spec, live_v, token_v):
            out = services.validate_spec(spec, live=live_v, token=token_v or None)
            return f"**{out['message']}**" if out.get("message") else "", out.get("issues", [])

        revalidate_btn.click(_revalidate, inputs=[spec_state, live_check, token_state],
                             outputs=[plan_status, issues_df])

        def _heal(spec, token_v, model_v, live_v):
            out = services.heal_spec(spec, token=token_v or None, model=model_v or None,
                                     live=live_v)
            canvas_note = show_spec(services.as_spec(out.get("spec"))) if out.get("spec") else ""
            return (out.get("spec"), f"**{out['status']}** — {out.get('message', '')}\n\n"
                                     f"canvas: {canvas_note}", out.get("issues", []),
                    out.get("timeline", ""), out.get("code", ""), out.get("summary", ""))

        heal_btn.click(_heal, inputs=[spec_state, token_state, model_dropdown, live_check],
                       outputs=[spec_state, plan_status, issues_df, timeline_md, code_box, spec_md])

        def _push_canvas(spec):
            return f"canvas: {show_spec(services.as_spec(spec))}"

        canvas_btn.click(_push_canvas, inputs=[spec_state], outputs=[plan_status])

        def _code(spec):
            return services.render_code(spec), services.export_spec_json(spec)

        code_btn.click(_code, inputs=[spec_state], outputs=[code_box, spec_json])

        def _import(text):
            spec = services.as_spec(services.import_payload(text))
            if spec is None:
                return None, "could not read that as a workflow", "", "", "", ""
            return (spec.to_dict(), f"imported **{spec.name}**", services.spec_summary(spec), "", "",
                    services.render_code(spec))

        import_btn.click(_import, inputs=[import_box],
                         outputs=[spec_state, plan_status, spec_md, issues_df, timeline_md,
                                  code_box])

        def _advise(spec, token_v, model_v):
            out = services.advise_upgrades(spec, token=token_v or None, model=model_v or None)
            choices = [s["brick_id"] for s in out.get("suggestions", [])]
            return out.get("message", ""), gr.update(choices=choices)

        advise_btn.click(_advise, inputs=[spec_state, token_state, model_dropdown],
                         outputs=[advise_md, upgrade_brick])

        def _apply_upgrade(spec, step_id, brick_id, token_v):
            out = services.apply_upgrade(spec, brick_id=brick_id or "", step_id=step_id or "")
            if not out.get("ok"):
                return spec, out.get("message", "nothing applied"), "", "", ""
            show_spec(services.as_spec(out.get("spec")))
            return (out["spec"], out["message"], out.get("summary", ""), out.get("code", ""), "")

        apply_upgrade_btn.click(_apply_upgrade,
                                inputs=[spec_state, upgrade_step, upgrade_brick, token_state],
                                outputs=[spec_state, advise_md, spec_md, code_box, plan_status])

        # dynamic run inputs + run action, re-rendered whenever the workflow changes
        @gr.render(inputs=[spec_state])
        def _render_run_inputs(spec):
            if not spec:
                gr.Markdown("*Build a workflow first — the inputs it needs will appear here.*")
                return
            ports = (spec or {}).get("inputs", [])
            with gr.Row():
                components = [_component_for(port) for port in ports]
            names = [p.get("port") for p in ports]

            def _run(token_v, *values):
                payload = dict(zip(names, values))
                out = services.run_workflow(spec, values=payload, token=token_v or None)
                return out.get("message", ""), out.get("rows", []), out.get("artifacts", [])

            run_btn.click(_run, inputs=[token_state, *components],
                          outputs=[run_status, run_rows, run_gallery])

        # leaderboard
        def _refresh_lb(brick_id, industry_v, license_v, tier_v, sort_v, token_v):
            out = services.leaderboard_view(modality=brick_id, industry=industry_v,
                                           license_filter=license_v, compute_tier=tier_v,
                                           sort=sort_v, token=token_v or None)
            slugs = [s for s in out.get("slugs", []) if s]
            return (out.get("message", ""), out.get("rows", []),
                    gr.update(choices=slugs), gr.update(choices=slugs))

        lb_btn.click(_refresh_lb,
                     inputs=[lb_modality, lb_industry, lb_license, lb_tier, lb_sort, token_state],
                     outputs=[lb_md, lb_df, load_slug, vote_slug])

        def _publish(spec, token_v, note_v):
            out = services.publish_workflow(spec, token=token_v or None, note=note_v or "")
            msg = out.get("message", "")
            url = f"[{out['slug']}]({out['url']})" if out.get("url") else ""
            return msg, url

        publish_btn.click(_publish, inputs=[spec_state, token_state, publish_note],
                          outputs=[share_status, publish_url])

        def _deploy(spec, token_v):
            out = services.deploy_space(spec, token=token_v or None)
            return out.get("message", "")

        deploy_btn.click(_deploy, inputs=[spec_state, token_state], outputs=[deploy_status])

        def _load(slug, token_v):
            out = services.load_workflow(slug or "", token=token_v or None)
            if not out.get("ok"):
                return None, out.get("message", ""), "", "", ""
            show_spec(services.as_spec(out.get("spec")))
            return out["spec"], out["message"], out.get("summary", ""), out.get("code", ""), ""

        load_btn.click(_load, inputs=[load_slug, token_state],
                       outputs=[spec_state, share_status, spec_md, code_box, plan_status])

        def _vote(slug, direction, token_v):
            out = services.cast_vote(slug or "", direction, token=token_v or None)
            return out.get("message", "")

        vote_up.click(lambda slug, tok: _vote(slug, 1, tok), inputs=[vote_slug, token_state],
                      outputs=[share_status])
        vote_down.click(lambda slug, tok: _vote(slug, -1, tok), inputs=[vote_slug, token_state],
                        outputs=[share_status])

        # bricks
        def _bricks(mod, commercial_v, running_v):
            rows = services.bricks_table(modality=mod, only_commercial=commercial_v,
                                        only_running=running_v)
            return get_registry().summary() + f" — showing {len(rows)}", rows

        bricks_btn.click(_bricks, inputs=[brick_modality, brick_commercial, brick_running],
                         outputs=[brick_md, brick_df])

        def _check(spec, token_v):
            out = services.check_workflow_bricks(spec, token=token_v or None)
            return out.get("message", ""), out.get("rows", [])

        check_btn.click(_check, inputs=[spec_state, token_state],
                        outputs=[check_md, check_df])

        demo.load(lambda: _bricks("any", True, False), outputs=[brick_md, brick_df])

    return demo