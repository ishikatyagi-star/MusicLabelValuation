import os
import json
import asyncio
import gradio as gr
from typing import Dict, Any, List

from music_catalog_pe_env.env import MusicCatalogPEEnvironment
from music_catalog_pe_env.tasks import TASKS
from music_catalog_pe_env.models import CatalogAction

# --- Helper Functions ---

def get_task_details(task_id: str) -> str:
    """Format task details into markdown."""
    task = TASKS.get(task_id)
    if not task:
        return "**Task not found.**"
    
    # Temporarily init env to load ground truth
    temp_env = MusicCatalogPEEnvironment()
    temp_env.reset(task_id=task_id)
    gt = temp_env._state.ground_truth
    
    md = f"""
### {task.description}

**Evaluation Criteria (What you need to find)**
- **True TTM Revenue:** ${gt.get('true_normalized_ttm_revenue', 0):,.2f}
- **Target Valuation Base:** ${gt.get('true_valuation_base', 0):,.2f}
- **Required Risk Flags:** {', '.join(gt.get('must_detect_risks', [])) or 'None'}
- **Recommendation:** `{gt.get('correct_recommendation', 'N/A').upper()}`
    """
    return md

# Global env instance for manual interaction
manual_env = MusicCatalogPEEnvironment()

def reset_manual_env(task_id: str):
    obs = manual_env.reset(task_id=task_id)
    return (
        f"**System:** Environment reset. Loaded catalog: `{task_id}`", 
        json.dumps(obs.result_payload, indent=2)
    )

def execute_action(action_type: str, action_params_str: str):
    if not action_type:
        return "No action selected.", "{}"
        
    try:
        params = json.loads(action_params_str) if action_params_str.strip() else {}
    except json.JSONDecodeError:
        return "Error: Invalid JSON.", "{}"
        
    action = CatalogAction(action_type=action_type, params=params)
    obs = manual_env.step(action)
    
    history_md = f"**Executed Action:** `{action_type}`\n"
    if obs.warnings:
        history_md += f"**Warnings:** {', '.join(obs.warnings)}\n"
        
    if action_type == "submit_final_valuation":
        score = max(0.01, min(0.99, float(obs.reward)))
        history_md += f"\n### Final Grade: {score:.4f} / 1.0000"
        
    return (
        history_md,
        json.dumps(obs.result_payload, indent=2)
    )

def quick_action(action_type: str):
    """Wrapper for buttons with no params."""
    return execute_action(action_type, "{}")

# --- Baseline Execution Logic ---

async def run_baseline_agent(task_id: str, hf_token: str):
    """Runs the inference logic dynamically and yields output to the chatbot."""
    if not hf_token:
        yield [{"role": "assistant", "content": "Error: HF Token or OpenAI Key required to run the baseline agent. Please enter it below."}]
        return

    os.environ["HF_TOKEN"] = hf_token
    from inference import run_task, MODEL_NAME
    from openai import OpenAI
    
    API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    client = OpenAI(base_url=API_BASE_URL, api_key=hf_token)
    
    # Universal Gradio 4.0 format: list of tuples (user, assistant)
    chat_history = [(None, f"Starting baseline AI analyst ({MODEL_NAME}) on catalog: `{task_id}`...")]
    yield chat_history
    
    from inference import get_model_action, SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS
    
    env_instance = MusicCatalogPEEnvironment()
    result = env_instance.reset(task_id=task_id)
    last_obs = result.model_dump()
    last_reward = max(0.01, min(0.99, float(result.reward)))
    max_steps = result.max_steps or 25
    
    history_list = []
    
    for step in range(1, max_steps + 1):
        if result.done:
            break
            
        chat_history.append((None, f"*Thinking (Step {step})...*"))
        yield chat_history
        
        try:
            action = get_model_action(client, step, last_obs, last_reward, history_list)
        except Exception as e:
             chat_history[-1] = (None, f"API Error: {e}. Defaulting to summary.")
             action = CatalogAction(action_type="inspect_catalog_summary", params={})
             yield chat_history
        
        action_json_str = action.model_dump_json()
        chat_history.append((None, f"**Decision:**\n```json\n{action_json_str}\n```"))
        yield chat_history
        
        result = env_instance.step(action)
        obs_dump = result.model_dump()
        reward = max(0.01, min(0.99, float(result.reward if result.reward is not None else 0.01)))
        
        history_list.append(f"Step {step}: {action.action_type} -> reward {reward:+.2f}")
        last_obs = obs_dump
        last_reward = reward
        
        result_snippet = json.dumps(obs_dump.get("result_payload", {}), indent=2)[:350] + "\n..."
        chat_history.append((f"**Environment Data:**\n```json\n{result_snippet}\n```", None))
        yield chat_history
        
    final_score = max(0.01, min(0.99, float(last_reward)))
    chat_history.append((None, f"### ✅ Episode Finished!\n**Final Grader Score:** {final_score:.4f}"))
    yield chat_history


# --- UI Layout ---

def create_ui():
    css_path = os.path.join(os.path.dirname(__file__), "shadcn-dark.css")
    with open(css_path, "r") as f:
        custom_css = f.read()

    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        primary_hue="zinc",
        secondary_hue="zinc",
        neutral_hue="zinc",
    )

    with gr.Blocks(theme=theme, css=custom_css, title="Music Catalog PE Dashboard") as interface:
        
        with gr.Column(elem_classes=["dashboard-header"]):
            gr.Markdown("# 🎵 Music Catalog Valuation Engine")

        with gr.Row():
            
            # --- LEFT COLUMN: Context & Baseline Agent ---
            with gr.Column(scale=1):
                gr.Markdown("## Task Selection")
                
                task_dropdown = gr.Dropdown(
                    choices=list(TASKS.keys()), 
                    value="easy_stable_evergreen", 
                    label="Active Catalog Profile",
                    interactive=True
                )
                
                task_details_md = gr.Markdown(get_task_details("easy_stable_evergreen"))
                
                task_dropdown.change(
                    fn=get_task_details,
                    inputs=[task_dropdown],
                    outputs=[task_details_md]
                )
                
                gr.Markdown("---")
                gr.Markdown("## Baseline AI Run")
                
                hf_token_input = gr.Textbox(
                    label="API Token (HF / OpenAI)", 
                    type="password", 
                    placeholder="hf_..."
                )
                
                run_agent_btn = gr.Button("▶ Autopilot (Run AI Agent)", variant="primary")
                
                agent_chatbot = gr.Chatbot(
                    label="AI Agent Terminal", 
                    height=500,
                    show_label=True
                )
                
                run_agent_btn.click(
                    fn=run_baseline_agent,
                    inputs=[task_dropdown, hf_token_input],
                    outputs=[agent_chatbot]
                )

            # --- RIGHT COLUMN: Human Manual Play ---
            with gr.Column(scale=2):
                gr.Markdown("## Analyst Workflow Console")
                gr.Markdown("Investigate the data yourself using the quick-actions below.")
                
                reset_env_btn = gr.Button("🔄 Reset Interface to selected Catalog")
                
                # Action Quick Buttons (Dashboard Cards style)
                with gr.Row():
                    btn_summary = gr.Button("📊 View Summary", variant="secondary")
                    btn_platforms = gr.Button("🎧 Platform Mix", variant="secondary")
                    btn_territories = gr.Button("🌍 Territory Mix", variant="secondary")
                with gr.Row():
                    btn_anomalies = gr.Button("⚠️ Run Anomaly Scan", variant="secondary")
                    btn_ttm = gr.Button("📈 Compute TTM Revenue", variant="secondary")

                # Output Viewer
                gr.Markdown("### 📥 Data Terminal")
                action_history_md = gr.Markdown("**System:** Ready for interaction.")
                env_output_json = gr.JSON(label="Database Response")

                # Wiring standard actions
                reset_env_btn.click(
                    fn=reset_manual_env, 
                    inputs=[task_dropdown], 
                    outputs=[action_history_md, env_output_json]
                )
                
                action_map = {
                    btn_summary: "inspect_catalog_summary",
                    btn_platforms: "inspect_platform_mix",
                    btn_territories: "inspect_territory_mix",
                    btn_anomalies: "inspect_anomalies",
                    btn_ttm: "compute_normalized_ttm"
                }
                
                for btn, act_name in action_map.items():
                    btn.click(
                        fn=lambda a=act_name: quick_action(a),
                        outputs=[action_history_md, env_output_json]
                    )
                
                # Final Valuation Form
                with gr.Box() if hasattr(gr, "Box") else gr.Group():
                    gr.Markdown("### 📄 Submit Investment Memo")
                    
                    with gr.Row():
                        val_ttm = gr.Number(label="Final TTM Revenue ($)", value=0)
                        val_base = gr.Number(label="Total Valuation ($)", value=0)
                    
                    with gr.Row():
                        val_rec = gr.Dropdown(choices=["acquire", "acquire_at_discount", "pass"], label="Recommendation", value="acquire")
                        val_risks = gr.Textbox(label="Risk Flags (e.g. VIRAL_SPIKE_LAST_3M)", placeholder="Comma separated strings")
                        
                    btn_submit_final = gr.Button("Grade My Memo 🚀", variant="primary")
                    
                    def submit_final_form(ttm, base, rec, risks, t_id):
                        risk_list = [r.strip() for r in risks.split(",") if r.strip()]
                        params = {
                            "catalog_id": t_id,
                            "estimated_normalized_ttm_revenue": float(ttm),
                            "valuation_low": float(base) * 0.85,
                            "valuation_base": float(base),
                            "valuation_high": float(base) * 1.15,
                            "estimated_top_tracks": [],
                            "estimated_key_platforms": [],
                            "estimated_risk_flags": risk_list,
                            "recommendation": rec,
                            "confidence_score": 0.9,
                            "memo": "Manual Analyst review."
                        }
                        return execute_action("submit_final_valuation", json.dumps(params))
                        
                    btn_submit_final.click(
                        fn=submit_final_form,
                        inputs=[val_ttm, val_base, val_rec, val_risks, task_dropdown],
                        outputs=[action_history_md, env_output_json]
                    )

        # Trigger initial reset MUST be inside Blocks context
        interface.load(fn=reset_manual_env, inputs=[task_dropdown], outputs=[action_history_md, env_output_json])
    
    return interface
