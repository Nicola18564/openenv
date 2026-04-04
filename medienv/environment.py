import json
import logging
import random
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def load_scenarios():
    scenario_path = Path(__file__).with_name("scenarios.json")
    with scenario_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class HealthTriageEnv:
    ACTIONS = [
        "ASK_FOLLOWUP",
        "ESCALATE_EMERGENCY",
        "RECOMMEND_CLINIC",
        "RECOMMEND_DOCTOR_VISIT",
        "RECOMMEND_SELF_CARE",
        "PROVIDE_SUPPORT_MESSAGE",
    ]

    DEFAULT_SCENARIOS = load_scenarios()

    def __init__(self, scenario=None, seed=None):
        self.rng = random.Random(seed)
        self.initial_scenario = (scenario or self.rng.choice(self.DEFAULT_SCENARIOS)).copy()
        self.scenario = self.initial_scenario.copy()
        self.step_count = 0
        self.done = False
        self.history = []
        LOGGER.info("Environment initialized with severity=%s", self.scenario["severity"])

    def reset(self):
        self.scenario = self.initial_scenario.copy()
        self.step_count = 0
        self.done = False
        self.history = []
        LOGGER.info("Environment reset")
        return self._build_state()

    def step(self, action):
        if self.done:
            LOGGER.warning("Step called after episode completion")
            return self._build_state(), 0.0, True, {"message": "Episode already finished."}

        if action not in self.ACTIONS:
            LOGGER.error("Invalid action received: %s", action)
            return self._build_state(), -1.0, False, {"error": "Invalid action."}

        self.step_count += 1
        reward, rationale = self._compute_reward(action)
        self._apply_transition(action)

        self.history.append({"step": self.step_count, "action": action, "reward": reward})
        LOGGER.info("Step %s action=%s reward=%s", self.step_count, action, reward)

        if action in {
            "ESCALATE_EMERGENCY",
            "RECOMMEND_CLINIC",
            "RECOMMEND_DOCTOR_VISIT",
            "RECOMMEND_SELF_CARE",
        } or self.step_count >= 3:
            self.done = True

        return self._build_state(), reward, self.done, {
            "step_count": self.step_count,
            "action_taken": action,
            "rationale": rationale,
            "history": self.history,
            "correct_action": self.expert_policy(),
        }

    def expert_policy(self, state=None):
        current = state or self._build_state()
        if current["fall_flag"] or current["severity"] == "high":
            return "ESCALATE_EMERGENCY"
        if current["severity"] == "medium" and current["rural_access"]:
            return "RECOMMEND_CLINIC"
        if current["severity"] == "medium":
            return "RECOMMEND_DOCTOR_VISIT"
        return "RECOMMEND_SELF_CARE"

    def available_actions(self):
        return list(self.ACTIONS)

    def benchmark(self, episodes=20):
        rewards = []
        solved = 0
        urgency_breakdown = {"high": 0, "medium": 0, "low": 0}

        for _ in range(episodes):
            self.initial_scenario = self.rng.choice(self.DEFAULT_SCENARIOS).copy()
            state = self.reset()
            urgency_breakdown[state["severity"]] += 1
            action = self.expert_policy(state)
            _, reward, done, info = self.step(action)
            rewards.append(reward)
            solved += int(done and action == info["correct_action"])

        return {
            "episodes": episodes,
            "average_reward": round(sum(rewards) / len(rewards), 2),
            "successful_triage_rate": round((solved / episodes) * 100, 1),
            "urgency_breakdown": urgency_breakdown,
        }

    def _compute_reward(self, action):
        severity = self.scenario["severity"]
        fall_flag = self.scenario["fall_flag"]
        epidemic_flag = self.scenario["epidemic_flag"]
        age_group = self.scenario["age_group"]

        if fall_flag or severity == "high":
            if action == "ESCALATE_EMERGENCY":
                return 3.0, "Emergency escalation is correct for high-risk or fall-related cases."
            if action in ["ASK_FOLLOWUP", "RECOMMEND_SELF_CARE"]:
                return -3.0, "This is unsafe for a high-risk situation."
            return -1.0, "A stronger response is needed for this case."

        if severity == "medium":
            if action in ["RECOMMEND_CLINIC", "RECOMMEND_DOCTOR_VISIT"]:
                return 2.0, "Appropriate escalation for a medium-risk case."
            if action == "ASK_FOLLOWUP":
                return 1.0, "Useful for gathering more context."
            if action == "RECOMMEND_SELF_CARE":
                return -1.0, "Too mild for a medium-risk case."
            return 0.0, "Neutral action."

        if severity == "low":
            if action in ["RECOMMEND_SELF_CARE", "PROVIDE_SUPPORT_MESSAGE", "ASK_FOLLOWUP"]:
                return 2.0, "Reasonable response for low-risk symptoms."
            return -1.0, "Over-escalation is unnecessary for a low-risk case."

        if epidemic_flag and age_group == "elderly":
            if action in ["RECOMMEND_DOCTOR_VISIT", "RECOMMEND_CLINIC"]:
                return 2.0, "Good precaution for a vulnerable patient."
            return 0.0, "No strong penalty, but better care is possible."

        return 0.0, "No specific reward."

    def _apply_transition(self, action):
        if action == "ASK_FOLLOWUP":
            self.scenario["mental_state"] = "more informed"
        elif action == "ESCALATE_EMERGENCY":
            self.scenario["mental_state"] = "urgent attention needed"
        elif action == "RECOMMEND_CLINIC":
            self.scenario["mental_state"] = "referred to clinic"
        elif action == "RECOMMEND_DOCTOR_VISIT":
            self.scenario["mental_state"] = "reassured and guided"
        elif action == "RECOMMEND_SELF_CARE":
            self.scenario["mental_state"] = "relieved"
        elif action == "PROVIDE_SUPPORT_MESSAGE":
            self.scenario["mental_state"] = "supported"

    def _build_state(self):
        return {
            "symptoms": self.scenario["symptoms"],
            "age_group": self.scenario["age_group"],
            "severity": self.scenario["severity"],
            "rural_access": self.scenario["rural_access"],
            "mental_state": self.scenario["mental_state"],
            "fall_flag": self.scenario["fall_flag"],
            "epidemic_flag": self.scenario["epidemic_flag"],
            "step_count": self.step_count,
            "done": self.done,
        }


HealthTriageEnvironment = HealthTriageEnv
