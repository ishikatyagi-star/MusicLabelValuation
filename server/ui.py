import os
import json
import gradio as gr
from typing import Any, Dict, List

from music_catalog_pe_env.env import MusicCatalogPEEnvironment
from music_catalog_pe_env.tasks import TASKS
from music_catalog_pe_env.models import CatalogAction

# ─── Helper Functions ────────────────────────────────────────────────────────

def get_task_info(task_id: str) -> str:
    task = TASKS.get(task_id)
    if not task:
        return "Task not found."
    
    env = MusicCatalogPEEnvironment()
    env.reset(task_id=task_id)
    gt = env._state.ground_truth

    difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(task.difficulty, "⚪")
    risks = ", ".join(gt.get("must_detect_risks", [])) or "None"
    
    return (
        f"## {difficulty_emoji} {task.description}\n\n"
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| **Difficulty** | {task.difficulty.upper()} |\n"
        f"| **Max Steps** | {task.max_steps} |\n"
        f"| **True TTM Revenue** | ${gt.get('true_normalized_ttm_revenue', 0):,.2f} |\n"
        f"| **Target Valuation** | ${gt.get('true_valuation_base', 0):,.2f} |\n"
        f"| **Required Risks** | {risks} |\n"
        f"| **Correct Call** | `{gt.get('correct_recommendation', 'N/A').upper()}` |\n"
    )

# Global env for the manual console
_env = MusicCatalogPEEnvironment()

def do_reset(task_id: str):
    obs = _env.reset(task_id=task_id)
    payload = json.dumps(obs.result_payload, indent=2, default=str)
    return f"✅ Environment loaded: **{task_id}**", payload, ""

def do_action(action_name: str):
    action = CatalogAction(action_type=action_name, params={})
    obs = _env.step(action)
    payload = json.dumps(obs.result_payload, indent=2, default=str)
    label = f"📦 **{action_name}** executed"
    if obs.warnings:
        label += f" ⚠️ {', '.join(obs.warnings)}"
    return label, payload, ""

def do_submit(ttm, valuation, recommendation, risks_str, task_id):
    risk_list = [r.strip() for r in risks_str.split(",") if r.strip()] if risks_str else []
    params = {
        "catalog_id": task_id,
        "estimated_normalized_ttm_revenue": float(ttm),
        "valuation_low": float(valuation) * 0.85,
        "valuation_base": float(valuation),
        "valuation_high": float(valuation) * 1.15,
        "estimated_top_tracks": [],
        "estimated_key_platforms": [],
        "estimated_risk_flags": risk_list,
        "recommendation": recommendation or "acquire",
        "confidence_score": 0.9,
        "memo": "Submitted via UI."
    }
    action = CatalogAction(action_type="submit_final_valuation", params=params)
    obs = _env.step(action)
    score = max(0.01, min(0.99, float(obs.reward)))
    payload = json.dumps(obs.result_payload, indent=2, default=str)
    
    # Build a prominent grade display
    grade_md = ""
    breakdown = obs.result_payload.get("score_breakdown", {})
    if breakdown:
        grade_md += "### Scoring Breakdown\n| Factor | Accuracy |\n|---|---|\n"
        for k, v in breakdown.items():
            grade_md += f"| **{k}** | {v} |\n"
        grade_md += "\n"

    if score >= 0.7:
        grade_md += f"# 🏆 Final Score: {score:.4f}\nExcellent analysis!"
    elif score >= 0.4:
        grade_md += f"# 📊 Final Score: {score:.4f}\nDecent, but room for improvement."
    else:
        grade_md += f"# ⚠️ Final Score: {score:.4f}\nNeeds significant improvement."
    
    return "✅ Valuation submitted and graded.", payload, grade_md


def run_agent_sync(task_id: str, api_key: str):
    if not api_key:
        return "❌ Please provide an API token first."
    
    os.environ["HF_TOKEN"] = api_key
    
    try:
        from openai import OpenAI
        from inference import get_model_action, MODEL_NAME
    except Exception as e:
        return f"❌ Import error: {e}"
    
    API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    
    env = MusicCatalogPEEnvironment()
    result = env.reset(task_id=task_id)
    last_obs = result.model_dump()
    last_reward = max(0.01, min(0.99, float(result.reward)))
    max_steps = result.max_steps or 25
    
    log_lines = [f"🚀 Starting {MODEL_NAME} on task: {task_id}\n"]
    history_list = []
    
    for step in range(1, max_steps + 1):
        if result.done:
            break
        
        try:
            action = get_model_action(client, step, last_obs, last_reward, history_list)
        except Exception as e:
            log_lines.append(f"Step {step}: ❌ API Error: {e}\n")
            action = CatalogAction(action_type="inspect_catalog_summary", params={})
        
        log_lines.append(f"Step {step}: 🤖 {action.action_type}")
        
        result = env.step(action)
        reward = max(0.01, min(0.99, float(result.reward if result.reward is not None else 0.01)))
        
        log_lines.append(f"         → reward: {reward:.4f}")
        if result.warnings:
            log_lines.append(f"         ⚠️ WARNING: {', '.join(result.warnings)}")
            if "error" in result.result_payload:
                err_msg = str(result.result_payload['error']).split('\n')[0] # First line of error
                log_lines.append(f"         🚨 ERROR: {err_msg}")
        
        history_list.append(f"Step {step}: {action.action_type} -> reward {reward:+.2f}")
        last_obs = result.model_dump()
        last_reward = reward
    
    final = max(0.01, min(0.99, float(last_reward)))
    log_lines.append(f"\n{'='*40}")
    log_lines.append(f"🏁 Final Score: {final:.4f}")
    log_lines.append(f"{'='*40}")
    
    return "\n".join(log_lines)


# ─── Build UI ────────────────────────────────────────────────────────────────

def create_ui():
    
    with gr.Blocks(
        title="Music Catalog PE Dashboard",
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="gray",
            neutral_hue="gray",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        ),
    ) as demo:
        
        gr.Markdown("# 🎵 Music Catalog Valuation Engine\n*AI-powered private equity due diligence for music catalogs*")
        
        with gr.Row():
            task_dropdown = gr.Dropdown(
                choices=list(TASKS.keys()),
                value="easy_stable_evergreen",
                label="📋 Select Catalog to Evaluate",
                scale=2,
            )
            reset_btn = gr.Button("🔄 Load Catalog", variant="primary", scale=1)
        
        task_info = gr.Markdown(get_task_info("easy_stable_evergreen"))
        task_dropdown.change(fn=get_task_info, inputs=[task_dropdown], outputs=[task_info])
        
        with gr.Tabs():
            
            # ── TAB 1: Investigate ────────────────────────────────────
            with gr.TabItem("🔍 Investigate Data"):
                gr.Markdown("Use the buttons below to inspect the catalog data. Each click consumes one analysis step.")
                
                with gr.Row():
                    btn_summary   = gr.Button("📊 Catalog Summary")
                    btn_tracks    = gr.Button("🎤 Top Tracks")
                    btn_platforms = gr.Button("🎧 Platform Mix")
                with gr.Row():
                    btn_territory = gr.Button("🌍 Territory Mix")
                    btn_anomalies = gr.Button("⚠️ Anomaly Scan")
                    btn_ttm       = gr.Button("📈 TTM Revenue")
                
                status_md = gr.Markdown("*Click a button above to start investigating.*")
                data_json = gr.JSON(label="Response Data")
                grade_display = gr.Markdown("")
            
            # ── TAB 2: Submit Valuation ───────────────────────────────
            with gr.TabItem("📄 Submit Valuation"):
                gr.Markdown("Enter your analysis results below and submit to receive a grade from the deterministic scoring engine.")
                
                with gr.Row():
                    val_ttm = gr.Number(label="Estimated TTM Revenue ($)", value=0, precision=2)
                    val_base = gr.Number(label="Estimated Total Valuation ($)", value=0, precision=2)
                with gr.Row():
                    val_rec = gr.Dropdown(
                        choices=["acquire", "acquire_at_discount", "pass"],
                        value="acquire",
                        label="Investment Recommendation",
                    )
                    val_risks = gr.Textbox(
                        label="Detected Risk Flags",
                        placeholder="e.g. HIGH_TRACK_CONCENTRATION, VIRAL_SPIKE_LAST_3M",
                    )
                
                submit_btn = gr.Button("🚀 Grade My Memo", variant="primary", size="lg")
                
                submit_status = gr.Markdown("")
                submit_json = gr.JSON(label="Grader Response")
                submit_grade = gr.Markdown("")
                
                submit_btn.click(
                    fn=do_submit,
                    inputs=[val_ttm, val_base, val_rec, val_risks, task_dropdown],
                    outputs=[submit_status, submit_json, submit_grade],
                )
            
            # ── TAB 3: AI Agent ───────────────────────────────────────
            with gr.TabItem("🤖 AI Agent (Autopilot)"):
                gr.Markdown(
                    "Provide an API token and click **Run** to watch an LLM agent autonomously analyze the catalog. "
                    "The agent will inspect data, identify risks, and submit a valuation."
                )
                with gr.Row():
                    api_key_input = gr.Textbox(label="API Token", type="password", placeholder="hf_...", scale=3)
                    run_btn = gr.Button("▶ Run Agent", variant="primary", scale=1)
                
                agent_log = gr.Textbox(
                    label="Agent Execution Log",
                    lines=20,
                    max_lines=40,
                    interactive=False,
                )
                
                run_btn.click(
                    fn=run_agent_sync,
                    inputs=[task_dropdown, api_key_input],
                    outputs=[agent_log],
                )
        
        # ── Wire investigation buttons ────────────────────────────────
        reset_btn.click(fn=do_reset, inputs=[task_dropdown], outputs=[status_md, data_json, grade_display])
        
        for btn, act in [
            (btn_summary, "inspect_catalog_summary"),
            (btn_tracks, "inspect_top_tracks"),
            (btn_platforms, "inspect_platform_mix"),
            (btn_territory, "inspect_territory_mix"),
            (btn_anomalies, "inspect_anomalies"),
            (btn_ttm, "compute_normalized_ttm"),
        ]:
            btn.click(fn=lambda a=act: do_action(a), outputs=[status_md, data_json, grade_display])
        
        # Load initial data
        demo.load(fn=do_reset, inputs=[task_dropdown], outputs=[status_md, data_json, grade_display])
    
    return demo
