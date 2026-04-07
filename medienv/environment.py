class HealthTriageEnvironment:
    ACTIONS = [
        "ASK_FOLLOWUP",
        "ESCALATE_EMERGENCY",
        "RECOMMEND_CLINIC",
        "RECOMMEND_DOCTOR_VISIT",
        "RECOMMEND_SELF_CARE",
        "PROVIDE_SUPPORT_MESSAGE",
    ]

    def __init__(self, scenario):
        self.initial_scenario = scenario.copy()
        self.scenario = scenario.copy()
        self.step_count = 0
        self.done = False
        self.history = []

    def reset(self):
        self.scenario = self.initial_scenario.copy()
        self.step_count = 0
        self.done = False
        self.history = []
        return self._build_state()

    def step(self, action):
        if self.done:
            return self._build_state(), 0.0, True, {"message": "Episode already finished."}

        if action not in self.ACTIONS:
            return self._build_state(), -1.0, False, {"error": "Invalid action."}

        self.step_count += 1
        reward, rationale = self._compute_reward(action)
        self._apply_transition(action)

        self.history.append(
            {"step": self.step_count, "action": action, "reward": reward}
        )

        if self.step_count >= 3:
            self.done = True

        return self._build_state(), reward, self.done, {
            "step_count": self.step_count,
            "action_taken": action,
            "rationale": rationale,
            "history": self.history,
        }

    def _compute_reward(self, action):
        severity = self.scenario["severity"]
        fall_flag = self.scenario["fall_flag"]
        epidemic_flag = self.scenario["epidemic_flag"]
        age_group = self.scenario["age_group"]

        if fall_flag or severity == "high":
            if action == "ESCALATE_EMERGENCY":
                return 2.0, "Emergency escalation is correct for high-risk/fall cases."
            if action in ["ASK_FOLLOWUP", "RECOMMEND_SELF_CARE"]:
                return -2.0, "This is too weak for a high-risk situation."
            return -1.0, "A stronger response is needed."

        if severity == "medium":
            if action in ["RECOMMEND_CLINIC", "RECOMMEND_DOCTOR_VISIT"]:
                return 2.0, "Appropriate escalation for a medium-risk case."
            if action == "ASK_FOLLOWUP":
                return 0.5, "Useful for gathering more context."
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


import json
from pathlib import Path


def load_scenarios():
    """Load scenarios from scenarios.json file."""
    scenario_path = Path(__file__).parent / "scenarios.json"
    with open(scenario_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Alias for backward compatibility
HealthTriageEnv = HealthTriageEnvironment
