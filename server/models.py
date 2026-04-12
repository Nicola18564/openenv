"""OpenEnv schema models for the placement-readiness environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class PlacementAction(Action):
    """Action payload for the placement environment."""

    action: str = Field(..., description="Selected placement action token.")


class PlacementObservation(Observation):
    """Observation returned after reset or step."""

    scenario_name: Optional[str] = Field(default=None, description="Current scenario name.")
    scenario_id: Optional[str] = Field(default=None, description="Stable scenario identifier.")
    case_id: Optional[str] = Field(default=None, description="Stable case identifier.")
    title: Optional[str] = Field(default=None, description="Human-readable case title.")
    name: Optional[str] = Field(default=None, description="Scenario name.")
    summary: Optional[str] = Field(default=None, description="Short scenario summary.")
    target_company: Optional[str] = Field(default=None, description="Target company type or name.")
    company_type: Optional[str] = Field(default=None, description="Company category.")
    role: Optional[str] = Field(default=None, description="Target role.")
    stage: Optional[str] = Field(default=None, description="Current readiness stage.")
    focus_modules: List[str] = Field(default_factory=list, description="Primary learning modules.")
    required_skills: List[str] = Field(default_factory=list, description="Required skills for the role.")
    skill_gaps: List[str] = Field(default_factory=list, description="Skills still missing.")
    initial_state: Dict[str, int] = Field(default_factory=dict, description="Initial skill state.")
    proof_targets: Dict[str, int] = Field(default_factory=dict, description="Eligibility proof thresholds.")
    skill_levels: Dict[str, int] = Field(default_factory=dict, description="Current skill levels.")
    projects: List[str] = Field(default_factory=list, description="Current portfolio projects.")
    project_count: int = Field(default=0, description="Number of projects built.")
    company_analysis_score: int = Field(default=0, description="Company-analysis progress.")
    testing_score: int = Field(default=0, description="Testing discipline score.")
    progress_score: int = Field(default=0, description="Weekly tracking / momentum score.")
    brand_score: int = Field(default=0, description="Personal branding score.")
    resume_score: int = Field(default=0, description="Resume readiness score.")
    interview_score: int = Field(default=0, description="Interview practice score.")
    application_score: int = Field(default=0, description="Application readiness score.")
    proof_score: int = Field(default=0, description="Proof / eligibility accumulation score.")
    applications_submitted: int = Field(default=0, description="Applications submitted so far.")
    feedback_pending: bool = Field(default=False, description="Whether failure feedback is pending.")
    step_count: int = Field(default=0, description="Number of steps taken in the episode.")
    total_reward: float = Field(default=0.0, description="Cumulative reward so far.")
    skill_average: float = Field(default=0.0, description="Average skill level across tracked skills.")
    company_match_score: int = Field(default=0, description="Heuristic company match score.")
    readiness_score: float = Field(default=0.0, description="Computed readiness score.")
    readiness_state: str = Field(default="needs_work", description="Readiness band.")
    proof_ready: bool = Field(default=False, description="Whether proof gates have been satisfied.")
    recommended_action: Optional[str] = Field(default=None, description="Recommended action from the environment.")
    correct_action: Optional[str] = Field(default=None, description="Oracle action for the scenario.")
    current_action_suggestion: Optional[str] = Field(default=None, description="Best next action suggestion.")
    reward_breakdown: Dict[str, float] = Field(default_factory=dict, description="Component reward breakdown.")
    resolution_quality: str = Field(default="pending", description="Resolution quality label.")
    growth_plan: str = Field(default="undetermined", description="Suggested growth plan.")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Step-by-step action history.")


class PlacementState(State):
    """Persistent environment state exposed by OpenEnv."""

    scenario_name: Optional[str] = Field(default=None, description="Current scenario name.")
    scenario_id: Optional[str] = Field(default=None, description="Stable scenario identifier.")
    case_id: Optional[str] = Field(default=None, description="Stable case identifier.")
    title: Optional[str] = Field(default=None, description="Human-readable case title.")
    summary: Optional[str] = Field(default=None, description="Short scenario summary.")
    target_company: Optional[str] = Field(default=None, description="Target company type or name.")
    company_type: Optional[str] = Field(default=None, description="Company category.")
    role: Optional[str] = Field(default=None, description="Target role.")
    stage: Optional[str] = Field(default=None, description="Current readiness stage.")
    focus_modules: List[str] = Field(default_factory=list, description="Primary learning modules.")
    required_skills: List[str] = Field(default_factory=list, description="Required skills for the role.")
    skill_gaps: List[str] = Field(default_factory=list, description="Skills still missing.")
    initial_state: Dict[str, int] = Field(default_factory=dict, description="Initial skill state.")
    proof_targets: Dict[str, int] = Field(default_factory=dict, description="Eligibility proof thresholds.")
    skill_levels: Dict[str, int] = Field(default_factory=dict, description="Current skill levels.")
    projects: List[str] = Field(default_factory=list, description="Current portfolio projects.")
    project_count: int = Field(default=0, description="Number of projects built.")
    company_analysis_score: int = Field(default=0, description="Company-analysis progress.")
    testing_score: int = Field(default=0, description="Testing discipline score.")
    progress_score: int = Field(default=0, description="Weekly tracking / momentum score.")
    brand_score: int = Field(default=0, description="Personal branding score.")
    resume_score: int = Field(default=0, description="Resume readiness score.")
    interview_score: int = Field(default=0, description="Interview practice score.")
    application_score: int = Field(default=0, description="Application readiness score.")
    proof_score: int = Field(default=0, description="Proof / eligibility accumulation score.")
    applications_submitted: int = Field(default=0, description="Applications submitted so far.")
    feedback_pending: bool = Field(default=False, description="Whether failure feedback is pending.")
    total_reward: float = Field(default=0.0, description="Cumulative reward so far.")
    skill_average: float = Field(default=0.0, description="Average skill level across tracked skills.")
    company_match_score: int = Field(default=0, description="Heuristic company match score.")
    readiness_score: float = Field(default=0.0, description="Computed readiness score.")
    readiness_state: str = Field(default="needs_work", description="Readiness band.")
    proof_ready: bool = Field(default=False, description="Whether proof gates have been satisfied.")
    recommended_action: Optional[str] = Field(default=None, description="Recommended action from the environment.")
    correct_action: Optional[str] = Field(default=None, description="Oracle action for the scenario.")
    current_action_suggestion: Optional[str] = Field(default=None, description="Best next action suggestion.")
    reward_breakdown: Dict[str, float] = Field(default_factory=dict, description="Component reward breakdown.")
    resolution_quality: str = Field(default="pending", description="Resolution quality label.")
    growth_plan: str = Field(default="undetermined", description="Suggested growth plan.")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Step-by-step action history.")
    last_action: Optional[str] = Field(default=None, description="Most recent action token.")
    last_reward: Optional[float] = Field(default=None, description="Reward from the most recent step.")
    done: bool = Field(default=False, description="Whether the episode has terminated.")
__all__ = [
    "PlacementAction",
    "PlacementObservation",
    "PlacementState",
]
