import sys
from typing import Any, Dict, List, Literal, Optional

# Compatibility for typing
if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State

class CatalogAction(Action):
    """
    Action interface for the Music Catalog PE Environment.
    The agent can choose an action type and supply appropriate parameters.
    """
    action_type: Literal[
        "inspect_catalog_summary",
        "inspect_track",
        "inspect_top_tracks",
        "inspect_bottom_tracks",
        "inspect_platform_mix",
        "inspect_territory_mix",
        "inspect_monthly_revenue_trend",
        "inspect_track_revenue_trend",
        "inspect_revenue_source_breakdown",
        "inspect_rights_split",
        "inspect_concentration_metrics",
        "inspect_anomalies",
        "compute_normalized_ttm",
        "estimate_growth_decay",
        "estimate_risk_flags",
        "draft_investment_memo",
        "submit_final_valuation",
    ] = Field(description="The type of action to perform.")
    
    entity_id: Optional[str] = Field(default=None, description="Optional ID for specific inspections (e.g., track_id).")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional structured parameters for the action.")


class CatalogObservation(Observation):
    """
    Observation returned after each step.
    Must contain what the agent requires to proceed without seeing hidden ground truth.
    """
    task_id: str = Field(description="The ID of the current task.")
    step_number: int = Field(description="Current step number.")
    max_steps: int = Field(description="Maximum allowed steps.")
    remaining_budget: int = Field(description="Steps remaining before forced termination.")
    available_actions: List[str] = Field(default_factory=list, description="List of valid action types.")
    
    result_payload: Dict[str, Any] = Field(default_factory=dict, description="The returned data from the requested inspection.")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated so far.")
    discovered_features: List[str] = Field(default_factory=list, description="Insights or keys discovered.")
    warnings: List[str] = Field(default_factory=list, description="Issues such as invalid actions or constraints.")
    can_submit: bool = Field(default=False, description="True if final submission is allowed.")


class FinalSubmission(BaseModel):
    """
    Structure required for the final valuation action.
    """
    catalog_id: str
    estimated_normalized_ttm_revenue: float
    estimated_top_tracks: List[str]
    estimated_key_platforms: List[str]
    estimated_risk_flags: List[str]
    valuation_low: float
    valuation_base: float
    valuation_high: float
    recommendation: Literal["acquire", "acquire_at_discount", "pass"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    memo: str


class CatalogState(State):
    """
    Internal environment episode state.
    """
    task_id: Optional[str] = None
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    discovered_insights: List[str] = Field(default_factory=list)
    remaining_budget: int = 0
    final_submission: Optional[Dict[str, Any]] = None
    ground_truth: Dict[str, Any] = Field(default_factory=dict, description="Hidden ground truth for the grader.")


class TaskConfig(BaseModel):
    """
    Configuration for a specific benchmark task.
    """
    task_id: str
    difficulty: str
    catalog_dir: str
    max_steps: int
    description: str
