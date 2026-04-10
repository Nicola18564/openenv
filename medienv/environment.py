from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from medienv.grader import (
    TERMINAL_ACTIONS,
    assess_case,
    compute_reward,
    recommend_action,
    score_action,
)


class HealthTriageEnvironment:
    ACTIONS = [
        "ASK_FOLLOWUP",
        "REQUEST_VITALS",
        "CHECK_MEDICATION_HISTORY",
        "PROVIDE_SUPPORT_MESSAGE",
        "SCHEDULE_TELEMEDICINE",
        "RECOMMEND_CLINIC",
        "RECOMMEND_DOCTOR_VISIT",
        "RECOMMEND_SELF_CARE",
        "ESCALATE_EMERGENCY",
    ]

    def __init__(self, scenario: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._scenarios = load_scenarios()
        self.initial_scenario = self._resolve_scenario(scenario)
        self.scenario = copy.deepcopy(self.initial_scenario)
        self.step_count = 0
        self.done = False
        self.history: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self.last_result: Dict[str, Any] = {}
        self.context_collected = False
        self.support_provided = False

    def _resolve_scenario(self, scenario: Optional[Dict[str, Any]]):
        if scenario is None:
            return copy.deepcopy(self.rng.choice(self._scenarios))
        if isinstance(scenario, str):
            matches = [item for item in self._scenarios if item["name"] == scenario]
            if not matches:
                raise ValueError(f"Unknown scenario name: {scenario}")
            return copy.deepcopy(matches[0])
        return copy.deepcopy(scenario)

    def available_actions(self):
        return list(self.ACTIONS)

    def expert_policy(self, state: Optional[Dict[str, Any]] = None):
        current_case = self.scenario if state is None else self.scenario
        assessment = assess_case(current_case)
        if assessment["urgency"] == "critical" or current_case.get("fall_flag"):
            return "ESCALATE_EMERGENCY"
        if current_case.get("severity") == "high":
            return "RECOMMEND_CLINIC" if current_case.get("rural_access") else "RECOMMEND_DOCTOR_VISIT"
        if current_case.get("severity") == "moderate":
            if current_case.get("rural_access") or current_case.get("mobility_issues"):
                return "SCHEDULE_TELEMEDICINE"
            return "RECOMMEND_DOCTOR_VISIT"
        if current_case.get("mental_state") in {"panicked", "distressed", "worried"}:
            return "PROVIDE_SUPPORT_MESSAGE"
        return "RECOMMEND_SELF_CARE"

    def reset(self, scenario: Optional[Dict[str, Any]] = None):
        if scenario is not None:
            self.initial_scenario = self._resolve_scenario(scenario)
        self.scenario = copy.deepcopy(self.initial_scenario)
        self.step_count = 0
        self.done = False
        self.history = []
        self.total_reward = 0.0
        self.last_result = {}
        self.context_collected = False
        self.support_provided = False
        return self._build_state()

    def step(self, action: str):
        if self.done:
            info = {
                "message": "Episode already finished.",
                "recommended_action": recommend_action(self.scenario),
                "reward_breakdown": {"safety": 0, "sequence": 0, "access": 0, "empathy": 0, "efficiency": 0},
                "resolution_quality": "finished",
                "care_plan": self.last_result.get("care_plan", "undetermined"),
            }
            return self._build_state(), 0.0, True, info

        if action not in self.ACTIONS:
            info = {
                "error": f"Invalid action: {action}",
                "recommended_action": recommend_action(self.scenario),
                "reward_breakdown": {"safety": -1, "sequence": 0, "access": 0, "empathy": 0, "efficiency": 0},
                "resolution_quality": "invalid_action",
                "care_plan": "undetermined",
            }
            return self._build_state(), -2.0, False, info

        result = score_action(self.scenario, action)
        reward = result["reward"]
        self.total_reward += reward
        self.step_count += 1

        if action == "ASK_FOLLOWUP":
            self.context_collected = True
            self.scenario["mental_state"] = "more informed"
        elif action == "PROVIDE_SUPPORT_MESSAGE":
            self.support_provided = True
            self.scenario["mental_state"] = "supported"
        elif action == "REQUEST_VITALS":
            self.context_collected = True
            self.scenario["mental_state"] = "vitals collected"
        elif action == "CHECK_MEDICATION_HISTORY":
            self.context_collected = True
            self.scenario["mental_state"] = "medication history reviewed"
        elif action == "SCHEDULE_TELEMEDICINE":
            self.scenario["mental_state"] = "telemedicine scheduled"
        elif action == "RECOMMEND_CLINIC":
            self.scenario["mental_state"] = "clinic recommended"
        elif action == "RECOMMEND_DOCTOR_VISIT":
            self.scenario["mental_state"] = "doctor visit recommended"
        elif action == "RECOMMEND_SELF_CARE":
            self.scenario["mental_state"] = "self-care advised"
        elif action == "ESCALATE_EMERGENCY":
            self.scenario["mental_state"] = "urgent escalation"

        if action in TERMINAL_ACTIONS or self.step_count >= 3:
            self.done = True

        history_entry = {
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "resolution_quality": result["resolution_quality"],
        }
        self.history.append(history_entry)
        self.last_result = result

        info = {
            "step_count": self.step_count,
            "action_taken": action,
            "recommended_action": result["recommended_action"],
            "correct_action": result["correct_action"],
            "assessment": result["assessment"],
            "verdict": result["verdict"],
            "resolution_quality": result["resolution_quality"],
            "reward_breakdown": result["reward_breakdown"],
            "rationale": result["rationale"],
            "care_plan": result["care_plan"],
            "history": self.history,
        }
        return self._build_state(), reward, self.done, info

    def benchmark(self, episodes: int = 20):
        total_reward = 0.0
        solved = 0
        urgency_breakdown = {"critical": 0, "high": 0, "moderate": 0, "low": 0}

        for _ in range(episodes):
            scenario = copy.deepcopy(self.rng.choice(self._scenarios))
            state = self.reset(scenario)
            urgency_breakdown[assess_case(scenario)["urgency"]] += 1
            steps = 0

            while not self.done and steps < 3:
                action = self.expert_policy(state)
                state, reward, done, info = self.step(action)
                total_reward += reward
                steps += 1
                if done:
                    if action == scenario.get("correct_action", recommend_action(scenario)):
                        solved += 1
                    break

        return {
            "episodes": episodes,
            "average_reward": round(total_reward / episodes, 2) if episodes else 0.0,
            "successful_triage_rate": round((solved / episodes) * 100, 1) if episodes else 0.0,
            "urgency_breakdown": urgency_breakdown,
        }

    def close(self):
        return None

    def _build_state(self):
        assessment = assess_case(self.scenario)
        scenario_name = self.scenario.get("name", self.scenario.get("title", "unknown"))
        summary = self.scenario.get("summary") or f"Patient case: {scenario_name}"
        return {
            "case_id": self.scenario.get("id", scenario_name),
            "title": self.scenario.get("title", scenario_name),
            "name": scenario_name,
            "summary": summary,
            "symptoms": self.scenario.get("symptoms", []),
            "duration_days": self.scenario.get("duration_days", 0),
            "age_group": self.scenario.get("age_group", "adult"),
            "severity": self.scenario.get("severity", "low"),
            "rural_access": self.scenario.get("rural_access", False),
            "mobility_issues": self.scenario.get("mobility_issues", False),
            "language_barrier": self.scenario.get("language_barrier", False),
            "insurance_risk": self.scenario.get("insurance_risk", False),
            "chronic_conditions": self.scenario.get("chronic_conditions", []),
            "red_flags": self.scenario.get("red_flags", []),
            "mental_state": self.scenario.get("mental_state", "neutral"),
            "step_count": self.step_count,
            "done": self.done,
            "history": list(self.history),
            "total_reward": self.total_reward,
            "risk_score": assessment["risk_score"],
            "urgency": assessment["urgency"],
            "access_risk": assessment["access_risk"],
            "social_risk": assessment["social_risk"],
            "recommended_action": recommend_action(self.scenario),
            "correct_action": self.scenario.get("correct_action", recommend_action(self.scenario)),
            "context_collected": self.context_collected,
            "support_provided": self.support_provided,
            "reward_breakdown": self.last_result.get(
                "reward_breakdown",
                {"safety": 0, "sequence": 0, "access": 0, "empathy": 0, "efficiency": 0},
            ),
            "resolution_quality": self.last_result.get("resolution_quality", "pending"),
            "care_plan": self.last_result.get("care_plan", "undetermined"),
        }


def load_scenarios():
    """Load scenarios from scenarios.json file."""
    scenario_path = Path(__file__).parent / "scenarios.json"
    with open(scenario_path, "r", encoding="utf-8") as f:
        return json.load(f)


HealthTriageEnv = HealthTriageEnvironment
