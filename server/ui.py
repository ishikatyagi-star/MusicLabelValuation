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

**Difficulty:** {task.difficulty.upper()}
**Max Steps:** {task.max_steps}

#### Evaluation Criteria (Ground Truth Highlights)
- **True Normalized TTM Revenue:** ${gt.get('true_normalized_ttm_revenue', 0):,.2f}
- **Assumed Base Multiple:** {gt.get('normalized_multiple', 0.0)}x
- **Target Valuation Base:** ${gt.get('true_valuation_base', 0):,.2f}
- **Required Risk Detections:** {', '.join(gt.get('must_detect_risks', [])) or 'None'}
- **Correct Recommendation:** `{gt.get('correct_recommendation', 'N/A').upper()}`
    """
    return md

# Global env instance for manual interaction
manual_env = MusicCatalogPEEnvironment()

def reset_manual_env(task_id: str):
    obs = manual_env.reset(task_id=task_id)
    return (
        f"Environment reset. Task: {task_id}", 
        json.dumps(obs.result_payload, indent=2),
        f"Steps: {obs.step_number} / {obs.max_steps}",
        0.0  # Reset reward
    )

def execute_action(action_type: str, action_params_str: str):
    if not action_type:
        return "No action selected.", "Error", "Error", 0.0
        
    try:
        params = json.loads(action_params_str) if action_params_str.strip() else {}
    except json.JSONDecodeError:
        return "Invalid JSON in parameters.", "Error", "Error", 0.0
        
    action = CatalogAction(action_type=action_type, params=params)
    obs = manual_env.step(action)
    
    history_md = f"**Executed:** `{action_type}`\n"
    if obs.warnings:
        history_md += f"**Warnings:** {', '.join(obs.warnings)}\n"
        
    return (
        history_md,
        json.dumps(obs.result_payload, indent=2),
        f"Steps: {obs.step_number} / {obs.max_steps}",
        obs.reward
    )

def quick_action(action_type: str):
    """Wrapper for buttons with no params."""
    return execute_action(action_type, "{}")

# --- Baseline Execution Logic ---

async def run_baseline_agent(task_id: str, hf_token: str):
    """Runs the inference logic dynamically and yields output to the chatbot."""
    if not hf_token:
        yield [("System", "Error: HF Token or OpenAI Key required to run the baseline agent. Please enter it below.")]
        return

    os.environ["HF_TOKEN"] = hf_token
    # Import here to avoid circular/initialization issues
    from inference import run_task, MODEL_NAME
    from openai import OpenAI
    
    API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    client = OpenAI(base_url=API_BASE_URL, api_key=hf_token)
    
    chat_history = [("System", f"Starting baseline agent ({MODEL_NAME}) on task: {task_id}...")]
    yield chat_history
    
    # We need a way to capture the output of run_task. Since run_task prints to stdout, 
    # we can temporarily redirect stdout or modify a copy of the inference logic.
    # For a clean UI experience without heavy refactoring, we will run the environment loop here.
    
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
            
        chat_history.append(("System", f"Agent is thinking (Step {step})..."))
        yield chat_history
        
        # Determine action
        try:
            # Await the API call if we were using an async client, but OpenAI client here is sync inside async func.
            # We will use the sync call wrapped or just call it (may block event loop slightly, but acceptable for demo).
            action = get_model_action(client, step, last_obs, last_reward, history_list)
        except Exception as e:
             chat_history[-1] = ("System", f"API Error: {e}. Attempting fallback action.")
             action = CatalogAction(action_type="inspect_catalog_summary", params={})
             yield chat_history
        
        action_json_str = action.model_dump_json()
        chat_history.append((f"Agent Action", f"```json\n{action_json_str}\n```"))
        yield chat_history
        
        # Execute in env
        result = env_instance.step(action)
        obs_dump = result.model_dump()
        reward = max(0.01, min(0.99, float(result.reward if result.reward is not None else 0.01)))
        
        history_list.append(f"Step {step}: {action.action_type} -> reward {reward:+.2f}")
        last_obs = obs_dump
        last_reward = reward
        
        # Show partial result snippet
        result_snippet = json.dumps(obs_dump.get("result_payload", {}))[:200] + "..."
        chat_history.append(("Environment Response", f"Reward: {reward:.2f}\n```json\n{result_snippet}\n```"))
        yield chat_history
        
    final_score = max(0.01, min(0.99, float(last_reward)))
    chat_history.append(("System", f"**Episode Finished!**\nFinal Grader Score: **{final_score:.4f}**"))
    yield chat_history


# --- UI Layout ---

def create_ui():
    css_path = os.path.join(os.path.dirname(__file__), "shadcn-dark.css")
    with open(css_path, "r") as f:
        custom_css = f.read()

    # Use a base theme that we heavily override with CSS
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        primary_hue="zinc",
        secondary_hue="zinc",
        neutral_hue="zinc",
    )

    with gr.Blocks(theme=theme, css=custom_css, title="Music Catalog PE Dashboard") as interface:
        
        # Header section
        with gr.Column(elem_classes=["dashboard-header"]):
            gr.Markdown("# Music Catalog PE Environment\nEvaluate private equity music catalogs interactively or using an AI agent.")

        # Main Layout
        with gr.Row():
            
            # --- LEFT COLUMN: Context & Baseline Agent ---
            with gr.Column(scale=1):
                gr.Markdown("## 1. Task Context")
                
                task_dropdown = gr.Dropdown(
                    choices=list(TASKS.keys()), 
                    value="easy_stable_evergreen", 
                    label="Select Diligence Task",
                    interactive=True
                )
                
                task_details_md = gr.Markdown(get_task_details("easy_stable_evergreen"))
                
                task_dropdown.change(
                    fn=get_task_details,
                    inputs=[task_dropdown],
                    outputs=[task_details_md]
                )
                
                gr.Markdown("---")
                gr.Markdown("## 2. Execute Baseline Agent")
                gr.Markdown("Run the LLM (Qwen 72B via API) against the selected environment.")
                
                hf_token_input = gr.Textbox(
                    label="API Token (HF / OpenAI)", 
                    type="password", 
                    placeholder="hf_..."
                )
                
                run_agent_btn = gr.Button("▶ Run Baseline Agent", variant="primary")
                
                agent_chatbot = gr.Chatbot(
                    label="Agent Execution Log", 
                    height=400,
                    show_label=True
                )
                
                run_agent_btn.click(
                    fn=run_baseline_agent,
                    inputs=[task_dropdown, hf_token_input],
                    outputs=[agent_chatbot]
                )

            # --- RIGHT COLUMN: Human Manual Play ---
            with gr.Column(scale=2):
                gr.Markdown("## Manual Analyst Console")
                gr.Markdown("Play the role of the AI agent. Request data views, analyze metrics, and submit a final valuation.")
                
                with gr.Row():
                    reset_env_btn = gr.Button("🔄 Initialize / Reset Environment")
                    step_status = gr.Textbox(label="Step Count", value="Steps: 0 / 25", interactive=False)
                    reward_status = gr.Number(label="Last Reward / Grade", value=0.0, interactive=False)
                
                # Action Quick Buttons (Dashboard Cards style)
                gr.Markdown("### Investigation Actions")
                with gr.Row():
                    btn_summary = gr.Button("View Summary", variant="secondary")
                    btn_platforms = gr.Button("View Platform Mix", variant="secondary")
                    btn_territories = gr.Button("View Territory Mix", variant="secondary")
                    btn_anomalies = gr.Button("Run Anomaly Scan", variant="secondary")
                    btn_ttm = gr.Button("Compute TTM", variant="secondary")

                # Custom Action form
                with gr.Accordion("Advanced / Custom Action", open=False):
                    manual_action_type = gr.Dropdown(
                        choices=[
                            "inspect_catalog_summary", "inspect_top_tracks", "inspect_platform_mix", 
                            "inspect_territory_mix", "inspect_monthly_revenue_trend", "inspect_track_revenue_trend",
                            "inspect_anomalies", "compute_normalized_ttm"
                        ], 
                        label="Action Type"
                    )
                    manual_action_params = gr.Textbox(label="Parameters (JSON)", value="{}")
                    btn_custom_action = gr.Button("Execute Custom Action")

                # Output Viewer
                gr.Markdown("### Environment Response")
                action_history_md = gr.Markdown("**Executed:** `-`")
                env_output_json = gr.JSON(label="Observation Data")

                # Wiring standard actions
                reset_env_btn.click(
                    fn=reset_manual_env, 
                    inputs=[task_dropdown], 
                    outputs=[action_history_md, env_output_json, step_status, reward_status]
                )
                
                # Button bindings
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
                        outputs=[action_history_md, env_output_json, step_status, reward_status]
                    )
                    
                btn_custom_action.click(
                    fn=execute_action,
                    inputs=[manual_action_type, manual_action_params],
                    outputs=[action_history_md, env_output_json, step_status, reward_status]
                )
                
                # Final Valuation Form
                with gr.Box() if hasattr(gr, "Box") else gr.Group(): # Fallback for newer Gradio versions
                    gr.Markdown("### Submit Final Valuation")
                    gr.Markdown("Ready? Submit your final analysis to the deterministic grader.")
                    
                    with gr.Row():
                        val_ttm = gr.Number(label="Estimated TTM Revenue ($)", value=0)
                        val_base = gr.Number(label="Valuation Base ($)", value=0)
                    
                    with gr.Row():
                        val_rec = gr.Dropdown(choices=["acquire", "acquire_at_discount", "pass"], label="Recommendation")
                        val_risks = gr.Textbox(label="Risk Flags (comma separated)", placeholder="e.g. top_track_concentration, viral_spike")
                        
                    btn_submit_final = gr.Button("Submit Final Evaluation", variant="primary")
                    
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
                            "confidence_score": 0.8,
                            "memo": "Manual UI submission."
                        }
                        return execute_action("submit_final_valuation", json.dumps(params))
                        
                    btn_submit_final.click(
                        fn=submit_final_form,
                        inputs=[val_ttm, val_base, val_rec, val_risks, task_dropdown],
                        outputs=[action_history_md, env_output_json, step_status, reward_status]
                    )

    # Trigger initial reset
    interface.load(fn=reset_manual_env, inputs=[task_dropdown], outputs=[action_history_md, env_output_json, step_status, reward_status])
    
    return interface

