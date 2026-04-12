"""OpenEnv wrapper around the placement-readiness environment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from medienv.environment import PlacementIntelligenceEnv, load_scenarios

from server.models import (
    PlacementAction,
    PlacementObservation,
    PlacementState,
)


SCENARIOS = {item["name"]: item for item in load_scenarios()}
README_PATH = Path(__file__).resolve().parents[1] / "README.md"


class PlacementIntelligenceEnvironment(
    Environment[PlacementAction, PlacementObservation, PlacementState]
):
    """Thin OpenEnv-compliant wrapper for the placement engine."""

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
        self._state = PlacementState(
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
            return PlacementIntelligenceEnv(seed=seed)
        return PlacementIntelligenceEnv(copy.deepcopy(SCENARIOS[scenario_name]), seed=seed)

    def _case_fields(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        reward_breakdown = state_dict.get("reward_breakdown", {}) or {}
        return {
            "scenario_name": state_dict.get("scenario_name") or self._scenario_name,
            "scenario_id": state_dict.get("scenario_id") or state_dict.get("case_id"),
            "case_id": state_dict.get("case_id") or state_dict.get("scenario_id"),
            "title": state_dict.get("title") or state_dict.get("name"),
            "name": state_dict.get("name") or self._scenario_name,
            "summary": state_dict.get("summary"),
            "target_company": state_dict.get("target_company"),
            "company_type": state_dict.get("company_type"),
            "role": state_dict.get("role"),
            "stage": state_dict.get("stage"),
            "focus_modules": list(state_dict.get("focus_modules", [])),
            "required_skills": list(state_dict.get("required_skills", [])),
            "skill_gaps": list(state_dict.get("skill_gaps", [])),
            "initial_state": dict(state_dict.get("initial_state", {})),
            "proof_targets": dict(state_dict.get("proof_targets", {})),
            "skill_levels": dict(state_dict.get("skill_levels", {})),
            "projects": list(state_dict.get("projects", [])),
            "project_count": int(state_dict.get("project_count", 0)),
            "company_analysis_score": int(state_dict.get("company_analysis_score", 0)),
            "testing_score": int(state_dict.get("testing_score", 0)),
            "progress_score": int(state_dict.get("progress_score", 0)),
            "brand_score": int(state_dict.get("brand_score", 0)),
            "resume_score": int(state_dict.get("resume_score", 0)),
            "interview_score": int(state_dict.get("interview_score", 0)),
            "application_score": int(state_dict.get("application_score", 0)),
            "proof_score": int(state_dict.get("proof_score", 0)),
            "applications_submitted": int(state_dict.get("applications_submitted", 0)),
            "feedback_pending": bool(state_dict.get("feedback_pending", False)),
            "step_count": int(state_dict.get("step_count", 0)),
            "total_reward": float(state_dict.get("total_reward", 0.0)),
            "skill_average": float(state_dict.get("skill_average", 0.0)),
            "company_match_score": int(state_dict.get("company_match_score", 0)),
            "readiness_score": float(state_dict.get("readiness_score", 0.0)),
            "readiness_state": state_dict.get("readiness_state", "needs_work"),
            "proof_ready": bool(state_dict.get("proof_ready", False)),
            "recommended_action": state_dict.get("recommended_action"),
            "correct_action": state_dict.get("correct_action"),
            "reward_breakdown": dict(reward_breakdown),
            "resolution_quality": state_dict.get("resolution_quality", "pending"),
            "growth_plan": state_dict.get("growth_plan", "undetermined"),
            "history": copy.deepcopy(state_dict.get("history", [])),
        }

    def _build_state(
        self,
        state_dict: Dict[str, Any],
        *,
        last_action: Optional[str],
        last_reward: Optional[float],
        done: bool,
    ) -> PlacementState:
        payload = self._case_fields(state_dict)
        return PlacementState(
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
    ) -> PlacementObservation:
        payload = self._case_fields(state_dict)
        metadata = {
            "state": copy.deepcopy(state_dict),
            "info": copy.deepcopy(info or {}),
        }
        return PlacementObservation(
            reward=reward,
            done=done,
            metadata=metadata,
            **payload,
        )

    def available_actions(self):
        return list(PlacementIntelligenceEnv.ACTIONS)

    def expert_policy(self, state: Optional[Dict[str, Any]] = None):
        return self._engine.expert_policy(state)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_name: Optional[str] = None,
        **kwargs: Any,
    ) -> PlacementObservation:
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
        action: PlacementAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PlacementObservation:
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
    def state(self) -> PlacementState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        readme_content = None
        if README_PATH.exists():
            readme_content = README_PATH.read_text(encoding="utf-8")

        return EnvironmentMetadata(
            name="Placement Intelligence Environment",
            description="Explainable and proof-based placement readiness simulation environment.",
            readme_content=readme_content,
            version="0.1.0",
            author="Shivakumar",
            documentation_url="https://github.com/Nicola18564/openenv",
        )

    def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()


MediAssistTriageEnvironment = PlacementIntelligenceEnvironment


__all__ = ["PlacementIntelligenceEnvironment", "MediAssistTriageEnvironment"]
