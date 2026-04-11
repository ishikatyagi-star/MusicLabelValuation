import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from music_catalog_pe_env.client import MusicCatalogPEEnv
from music_catalog_pe_env.models import CatalogAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

BENCHMARK = "music_catalog_pe_env"
TEMPERATURE = 0.7
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.5  # Need at least 50% score


def _clamp_score(val: float) -> float:
    """Absolute safety net: score is ALWAYS strictly in (0, 1)."""
    return max(0.01, min(0.99, float(val)))


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an elite music rights analyst evaluating a catalog for acquisition.
    You will inspect the catalog with the available actions until you have sufficient confidence,
    then you MUST use 'submit_final_valuation' action to produce your investment memo.
    
    The environment provides JSON responses.
    Reply with EXACTLY ONE valid JSON command using this schema:
    {
      "action_type": "<action_name>",
      "params": { ... }
    }
    No markdown, no conversation, ONLY JSON.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(task: str, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Format action string to fit safely in one line
    action_str = action.replace('\n', ' ').replace('\r', '')
    # Clamp reward in every step log too
    clamped_reward = _clamp_score(reward)
    print(
        f"[STEP] task={task} step={step} action={action_str} reward={clamped_reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # CRITICAL: clamp score RIGHT HERE at the print boundary -- impossible to bypass
    final_score = _clamp_score(score)
    clamped_rewards = [_clamp_score(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped_rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={final_score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, last_obs: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    obs_str = json.dumps(last_obs, indent=2)
    return textwrap.dedent(
        f"""
        Step: {step}
        Last observation:
        {obs_str}
        Last reward: {last_reward:.2f}
        Previous steps history:
        {history_block}
        
        Output your next action JSON.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, last_obs: dict, last_reward: float, history: List[str]) -> CatalogAction:
    user_prompt = build_user_prompt(step, last_obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Cleanup markdown brackets if model added them
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        data = json.loads(text.strip())
        return CatalogAction(**data)
    except Exception as exc:
        # Fallback to inspect summary if something goes wrong
        return CatalogAction(action_type="inspect_catalog_summary", params={})


async def run_task(task_id: str, client: OpenAI) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    # CRITICAL: Initialize score to a VALID value, never 0.0
    score = 0.01
    success = False
    
    try:
        # We use direct instantiation here so inference script works without standalone server
        from music_catalog_pe_env.env import MusicCatalogPEEnvironment
        env_instance = MusicCatalogPEEnvironment()
        
        result = env_instance.reset(task_id=task_id)
        last_obs = result.model_dump()
        last_reward = _clamp_score(result.reward)
        
        max_steps = result.max_steps or 25
        
        for step in range(1, max_steps + 1):
            if result.done:
                break
                
            action = get_model_action(client, step, last_obs, last_reward, history)
            action_json_str = action.model_dump_json()
            
            result = env_instance.step(action)
            obs_dump = result.model_dump()
            
            # CRITICAL: Clamp reward immediately, never allow raw 0.0 or 1.0
            reward = _clamp_score(result.reward if result.reward is not None else 0.01)
            done = result.done
            # Detect error from observation
            error = None
            if hasattr(result, 'warnings') and result.warnings:
                error = result.warnings[0]
            if "error" in obs_dump.get("result_payload", {}):
                error = obs_dump["result_payload"]["error"]
            
            rewards.append(reward)
            steps_taken = step
            last_obs = obs_dump
            last_reward = reward
            
            log_step(task=task_id, step=step, action=action_json_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action.action_type} -> reward {reward:+.2f}")
            
            if done:
                # If finished, the final reward is the score
                score = reward
                break
        
        # Clamp score one more time for safety
        score = _clamp_score(score)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] env error: {e}", flush=True)
        # Even on crash, score stays at safe default 0.01
    finally:
        # log_end does its own internal clamping too -- belt AND suspenders
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Generate data locally in case it doesn't exist yet
    try:
        from music_catalog_pe_env.generators import generate_all
        generate_all()
    except Exception:
        pass
        
    tasks = ["easy_stable_evergreen", "medium_concentration_risk", "hard_viral_spike_noisy"]
    for task_id in tasks:
        await run_task(task_id, client)
        
if __name__ == "__main__":
    asyncio.run(main())
