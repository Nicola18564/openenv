"""OpenEnv wrapper around the custom MediAssist triage environment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from medienv.environment import HealthTriageEnvironment, load_scenarios

from server.models import MediAssistAction, MediAssistObservation, MediAssistState


SCENARIOS = {item["name"]: item for item in load_scenarios()}
README_PATH = Path(__file__).resolve().parents[1] / "README.md"


class MediAssistTriageEnvironment(
    Environment[MediAssistAction, MediAssistObservation, MediAssistState]
):
    """Thin OpenEnv-compliant wrapper for the triage engine."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(
        self,
        scenario_name: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ):
        super().__init__()
        self._seed = seed
        self._episode_id = episode_id
        self._scenario_name = scenario_name
        self._engine = None
        self._state = MediAssistState(
            episode_id=episode_id,
            scenario_name=scenario_name,
        )
        self.reset(seed=seed, episode_id=self._episode_id, scenario_name=scenario_name)

    @staticmethod
    def _normalize_scenario_name(scenario_name: Optional[str]) -> Optional[str]:
        if scenario_name is None:
            return None
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario name: {scenario_name}")
        return scenario_name

    def _create_engine(self, scenario_name: Optional[str], seed: Optional[int]):
        if scenario_name is None:
            return HealthTriageEnvironment(seed=seed)
        return HealthTriageEnvironment(copy.deepcopy(SCENARIOS[scenario_name]), seed=seed)

    def _case_fields(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        scenario = getattr(self._engine, "scenario", {}) or {}

        symptoms = state_dict.get("symptoms", [])
        if isinstance(symptoms, str):
            symptoms = [symptoms]

        chronic_conditions = state_dict.get("chronic_conditions", [])
        if isinstance(chronic_conditions, str):
            chronic_conditions = [chronic_conditions]

        red_flags = state_dict.get("red_flags", [])
        if isinstance(red_flags, str):
            red_flags = [red_flags]

        reward_breakdown = state_dict.get("reward_breakdown", {})
        if reward_breakdown is None:
            reward_breakdown = {}

        return {
            "scenario_name": state_dict.get("name") or self._scenario_name,
            "case_id": state_dict.get("case_id") or state_dict.get("name"),
            "title": state_dict.get("title") or state_dict.get("name"),
            "summary": state_dict.get("summary"),
            "symptoms": list(symptoms),
            "duration_days": int(state_dict.get("duration_days", 0)),
            "age_group": state_dict.get("age_group", "adult"),
            "severity": state_dict.get("severity", "low"),
            "fall_flag": bool(scenario.get("fall_flag", False)),
            "epidemic_flag": bool(scenario.get("epidemic_flag", False)),
            "rural_access": bool(state_dict.get("rural_access", False)),
            "mobility_issues": bool(state_dict.get("mobility_issues", False)),
            "language_barrier": bool(state_dict.get("language_barrier", False)),
            "insurance_risk": bool(state_dict.get("insurance_risk", False)),
            "chronic_conditions": list(chronic_conditions),
            "red_flags": list(red_flags),
            "mental_state": state_dict.get("mental_state", "neutral"),
            "step_count": int(state_dict.get("step_count", 0)),
            "total_reward": float(state_dict.get("total_reward", 0.0)),
            "risk_score": int(state_dict.get("risk_score", 0)),
            "urgency": state_dict.get("urgency", "low"),
            "access_risk": int(state_dict.get("access_risk", 0)),
            "social_risk": int(state_dict.get("social_risk", 0)),
            "recommended_action": state_dict.get("recommended_action"),
            "correct_action": state_dict.get("correct_action"),
            "context_collected": bool(state_dict.get("context_collected", False)),
            "support_provided": bool(state_dict.get("support_provided", False)),
            "reward_breakdown": dict(reward_breakdown),
            "resolution_quality": state_dict.get("resolution_quality", "pending"),
            "care_plan": state_dict.get("care_plan", "undetermined"),
            "history": copy.deepcopy(state_dict.get("history", [])),
        }

    def _build_state(
        self,
        state_dict: Dict[str, Any],
        *,
        last_action: Optional[str],
        last_reward: Optional[float],
        done: bool,
    ) -> MediAssistState:
        payload = self._case_fields(state_dict)
        return MediAssistState(
            episode_id=self._episode_id,
            last_action=last_action,
            last_reward=last_reward,
            done=done,
            **payload,
        )

    def _build_observation(
        self,
        state_dict: Dict[str, Any],
        *,
        reward: Optional[float],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> MediAssistObservation:
        payload = self._case_fields(state_dict)
        metadata = {
            "state": copy.deepcopy(state_dict),
            "info": copy.deepcopy(info or {}),
        }
        return MediAssistObservation(
            reward=reward,
            done=done,
            metadata=metadata,
            **payload,
        )

    def available_actions(self):
        return list(HealthTriageEnvironment.ACTIONS)

    def expert_policy(self, state: Optional[Dict[str, Any]] = None):
        return self._engine.expert_policy(state)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_name: Optional[str] = None,
        **kwargs: Any,
    ) -> MediAssistObservation:
        if seed is not None:
            self._seed = seed
        self._episode_id = episode_id or str(uuid4())
        if scenario_name is not None:
            self._scenario_name = self._normalize_scenario_name(scenario_name)

        self._engine = self._create_engine(self._scenario_name, self._seed)
        state_dict = self._engine.reset(
            scenario=copy.deepcopy(SCENARIOS[self._scenario_name])
            if self._scenario_name in SCENARIOS
            else None
        )
        self._scenario_name = self._engine.scenario.get("name", self._scenario_name)
        self._state = self._build_state(
            state_dict,
            last_action=None,
            last_reward=None,
            done=False,
        )
        return self._build_observation(
            state_dict,
            reward=None,
            done=False,
            info={"message": "Environment reset"},
        )

    def step(
        self,
        action: MediAssistAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MediAssistObservation:
        action_token = action.action if hasattr(action, "action") else str(action)
        state_dict, reward, done, info = self._engine.step(action_token)
        self._state = self._build_state(
            state_dict,
            last_action=action_token,
            last_reward=reward,
            done=done,
        )
        return self._build_observation(
            state_dict,
            reward=reward,
            done=done,
            info=info,
        )

    @property
    def state(self) -> MediAssistState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        readme_content = None
        if README_PATH.exists():
            readme_content = README_PATH.read_text(encoding="utf-8")

        return EnvironmentMetadata(
            name="MediAssist Triage Arena",
            description="Explainable and equity-aware health triage simulation environment.",
            readme_content=readme_content,
            version="0.1.0",
            author="Shivakumar",
            documentation_url="https://github.com/Nicola18564/openenv",
        )

    def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()


__all__ = ["MediAssistTriageEnvironment"]
