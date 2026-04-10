"""OpenEnv schema models for the MediAssist triage environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class MediAssistAction(Action):
    """Action payload for the triage environment."""

    action: str = Field(..., description="Selected triage action token.")


class MediAssistObservation(Observation):
    """Observation returned after reset or step."""

    scenario_name: Optional[str] = Field(default=None, description="Current scenario name.")
    case_id: Optional[str] = Field(default=None, description="Stable case identifier.")
    title: Optional[str] = Field(default=None, description="Human-readable case title.")
    name: Optional[str] = Field(default=None, description="Scenario name.")
    summary: Optional[str] = Field(default=None, description="Short case summary.")
    symptoms: List[str] = Field(default_factory=list, description="Presenting symptoms.")
    duration_days: int = Field(default=0, description="Symptom duration in days.")
    age_group: str = Field(default="adult", description="Age group for the patient.")
    severity: str = Field(default="low", description="Scenario severity level.")
    fall_flag: bool = Field(default=False, description="Whether the case includes a fall.")
    epidemic_flag: bool = Field(default=False, description="Whether the case occurs in an outbreak context.")
    rural_access: bool = Field(default=False, description="Whether the patient has limited rural access.")
    mobility_issues: bool = Field(default=False, description="Whether mobility is limited.")
    language_barrier: bool = Field(default=False, description="Whether a language barrier is present.")
    insurance_risk: bool = Field(default=False, description="Whether access to care is financially constrained.")
    chronic_conditions: List[str] = Field(default_factory=list, description="Relevant chronic conditions.")
    red_flags: List[str] = Field(default_factory=list, description="Red-flag symptoms or risks.")
    mental_state: str = Field(default="neutral", description="Observed patient emotional state.")
    step_count: int = Field(default=0, description="Number of steps taken in the episode.")
    total_reward: float = Field(default=0.0, description="Cumulative reward so far.")
    risk_score: int = Field(default=0, description="Computed risk score.")
    urgency: str = Field(default="low", description="Urgency band.")
    access_risk: int = Field(default=0, description="Access-to-care risk score.")
    social_risk: int = Field(default=0, description="Social vulnerability score.")
    recommended_action: Optional[str] = Field(
        default=None, description="Recommended action from the environment."
    )
    correct_action: Optional[str] = Field(
        default=None, description="Oracle action for the case."
    )
    context_collected: bool = Field(
        default=False, description="Whether follow-up data has been collected."
    )
    support_provided: bool = Field(
        default=False, description="Whether empathy/support has been provided."
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Component reward breakdown."
    )
    resolution_quality: str = Field(
        default="pending", description="Resolution quality label."
    )
    care_plan: str = Field(
        default="undetermined", description="Suggested care plan."
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Step-by-step action history."
    )


class MediAssistState(State):
    """Persistent environment state exposed by OpenEnv."""

    scenario_name: Optional[str] = Field(default=None, description="Current scenario name.")
    case_id: Optional[str] = Field(default=None, description="Stable case identifier.")
    title: Optional[str] = Field(default=None, description="Human-readable case title.")
    summary: Optional[str] = Field(default=None, description="Short case summary.")
    symptoms: List[str] = Field(default_factory=list, description="Presenting symptoms.")
    duration_days: int = Field(default=0, description="Symptom duration in days.")
    age_group: str = Field(default="adult", description="Age group for the patient.")
    severity: str = Field(default="low", description="Scenario severity level.")
    fall_flag: bool = Field(default=False, description="Whether the case includes a fall.")
    epidemic_flag: bool = Field(default=False, description="Whether the case occurs in an outbreak context.")
    rural_access: bool = Field(default=False, description="Whether the patient has limited rural access.")
    mobility_issues: bool = Field(default=False, description="Whether mobility is limited.")
    language_barrier: bool = Field(default=False, description="Whether a language barrier is present.")
    insurance_risk: bool = Field(default=False, description="Whether access to care is financially constrained.")
    chronic_conditions: List[str] = Field(default_factory=list, description="Relevant chronic conditions.")
    red_flags: List[str] = Field(default_factory=list, description="Red-flag symptoms or risks.")
    mental_state: str = Field(default="neutral", description="Current patient emotional state.")
    total_reward: float = Field(default=0.0, description="Cumulative reward so far.")
    risk_score: int = Field(default=0, description="Computed risk score.")
    urgency: str = Field(default="low", description="Urgency band.")
    access_risk: int = Field(default=0, description="Access-to-care risk score.")
    social_risk: int = Field(default=0, description="Social vulnerability score.")
    recommended_action: Optional[str] = Field(
        default=None, description="Recommended action from the environment."
    )
    correct_action: Optional[str] = Field(
        default=None, description="Oracle action for the case."
    )
    context_collected: bool = Field(
        default=False, description="Whether follow-up data has been collected."
    )
    support_provided: bool = Field(
        default=False, description="Whether empathy/support has been provided."
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Component reward breakdown."
    )
    resolution_quality: str = Field(
        default="pending", description="Resolution quality label."
    )
    care_plan: str = Field(
        default="undetermined", description="Suggested care plan."
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Step-by-step action history."
    )
    last_action: Optional[str] = Field(default=None, description="Most recent action token.")
    last_reward: Optional[float] = Field(default=None, description="Reward from the most recent step.")
    done: bool = Field(default=False, description="Whether the episode has terminated.")


__all__ = [
    "MediAssistAction",
    "MediAssistObservation",
    "MediAssistState",
]
