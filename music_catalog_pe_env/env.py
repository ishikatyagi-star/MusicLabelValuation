import os
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

from .data_loader import CatalogDataLoader
from .models import CatalogAction, CatalogObservation, CatalogState, FinalSubmission
from .rewards import compute_final_reward, compute_step_reward
from .tasks import get_task


class MusicCatalogPEEnvironment(Environment):
    """
    OpenEnv Environment for Music Catalog Private Equity Valuation.
    """
    
    def __init__(self):
        super().__init__()
        self._state = CatalogState()
        # Ensure we set base_dir properly relying on repo structure
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_loader = CatalogDataLoader(base_dir=base_dir)

    def close(self):
        pass

    @property
    def state(self) -> CatalogState:
        return self._state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> CatalogObservation:
        """
        Resets the environment. Expects 'task_id' in kwargs.
        """
        task_id = kwargs.get("task_id", "easy_stable_evergreen")
        task_config = get_task(task_id)
        if not task_config:
            raise ValueError(f"Unknown task_id: {task_id}")
            
        self.data_loader.load(task_config.catalog_dir)
        
        self._state = CatalogState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            remaining_budget=task_config.max_steps,
            ground_truth=self.data_loader.ground_truth,
        )
        
        return self._build_observation(
            result_payload={"message": f"Task '{task_id}' started.", "description": task_config.description},
            reward=0.0,
            done=False
        )

    def step(self, action: CatalogAction, **kwargs) -> CatalogObservation:
        """Processes an action."""
        if self._state.remaining_budget <= 0:
            return self._build_observation(
                result_payload={"error": "Budget exhausted"},
                reward=compute_final_reward({}, self._state.ground_truth, self._state.step_count, get_task(self._state.task_id).max_steps),
                done=True,
                warnings=["Max steps exhausted."]
            )

        self._state.step_count += 1
        self._state.remaining_budget -= 1
        
        act = action.action_type
        result = {}
        done = False
        is_invalid = False
        
        try:
            if act == "inspect_catalog_summary":
                result = self.data_loader.query_catalog_summary()
                
            elif act == "inspect_top_tracks":
                result = self.data_loader.query_top_tracks(limit=action.params.get("limit", 5))
                
            elif act == "inspect_platform_mix":
                result = {"platform_mix": self.data_loader.query_platform_mix()}
                
            elif act == "inspect_territory_mix":
                result = {"territory_mix": self.data_loader.query_territory_mix()}
                
            elif act == "inspect_monthly_revenue_trend":
                result = {"trend": self.data_loader.query_revenue_trend(limit_months=action.params.get("limit_months", 12))}
                
            elif act == "inspect_anomalies":
                result = self.data_loader.query_anomalies()
                
            elif act == "compute_normalized_ttm":
                # Mock a calculator providing TTM
                result = {"raw_ttm": self.data_loader.query_revenue_trend(12)}
                
            elif act == "submit_final_valuation":
                submission = FinalSubmission(**action.params)
                self._state.final_submission = submission.model_dump()
                result = {"message": "Final valuation submitted successfully."}
                done = True
                
            else:
                result = {"error": f"Action '{act}' not implemented or invalid."}
                is_invalid = True
                
        except Exception as e:
            result = {"error": str(e)}
            is_invalid = True

        # Rewards
        if done and act == "submit_final_valuation":
            step_reward = compute_final_reward(
                self._state.final_submission,
                self._state.ground_truth,
                self._state.step_count,
                get_task(self._state.task_id).max_steps
            )
        else:
            step_reward = compute_step_reward(act, self._state.action_history, is_invalid)

        self._state.action_history.append({
            "action_type": act,
            "params": action.params,
            "result_summary": "success" if not is_invalid else "failed"
        })

        return self._build_observation(
            result_payload=result,
            reward=step_reward,
            done=done,
            warnings=["Invalid action"] if is_invalid else []
        )

    def _build_observation(self, result_payload: Dict[str, Any], reward: float, done: bool, warnings: list = None) -> CatalogObservation:
        task_config = get_task(self._state.task_id)
        return CatalogObservation(
            done=done,
            reward=reward if reward is not None else 0.001,
            task_id=self._state.task_id,
            step_number=self._state.step_count,
            max_steps=task_config.max_steps if task_config else 0,
            remaining_budget=self._state.remaining_budget,
            available_actions=[
                "inspect_catalog_summary", "inspect_top_tracks", "inspect_platform_mix", 
                "inspect_territory_mix", "inspect_monthly_revenue_trend", "inspect_anomalies", 
                "submit_final_valuation"
            ],
            result_payload=result_payload,
            cumulative_reward=0.0, # Handled implicitly by grader
            discovered_features=self._state.discovered_insights,
            warnings=warnings or [],
            can_submit=self._state.step_count >= 1
        )
