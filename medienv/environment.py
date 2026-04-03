import random

from medienv.grader import TERMINAL_ACTIONS, assess_case, compute_reward, explain_action
from medienv.tasks import ACTION_CATALOG, SCENARIOS


class HealthTriageEnv:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.current_case = None
        self.done = False
        self.history = []
        self.total_reward = 0
        self.last_explanation = {}

    def reset(self, case_id=None):
        if case_id:
            matches = [case for case in SCENARIOS if case["id"] == case_id]
            if not matches:
                raise ValueError(f"Unknown case_id: {case_id}")
            self.current_case = matches[0]
        else:
            self.current_case = self.rng.choice(SCENARIOS)

        self.done = False
        self.history = []
        self.total_reward = 0
        self.last_explanation = {}
        return self.state()

    def available_actions(self):
        return list(ACTION_CATALOG)

    def state(self):
        if self.current_case is None:
            return {}

        assessment = assess_case(self.current_case)
        return {
            "case_id": self.current_case["id"],
            "title": self.current_case["title"],
            "summary": self.current_case["summary"],
            "age_group": self.current_case["age_group"],
            "symptoms": self.current_case["symptoms"],
            "duration_days": self.current_case["duration_days"],
            "severity": self.current_case["severity"],
            "rural_access": self.current_case["rural_access"],
            "mobility_issues": self.current_case["mobility_issues"],
            "language_barrier": self.current_case["language_barrier"],
            "insurance_risk": self.current_case["insurance_risk"],
            "chronic_conditions": self.current_case["chronic_conditions"],
            "red_flags": self.current_case["red_flags"],
            "history": list(self.history),
            "done": self.done,
            "total_reward": self.total_reward,
            "risk_score": assessment["risk_score"],
            "urgency": assessment["urgency"],
            "access_risk": assessment["access_risk"],
            "social_risk": assessment["social_risk"],
        }

    def expert_policy(self, state=None):
        current_state = state or self.state()
        if current_state.get("urgency") == "critical":
            return "ESCALATE_EMERGENCY"
        if current_state.get("severity") == "high" and current_state.get("rural_access"):
            return "RECOMMEND_CLINIC"
        if current_state.get("severity") == "high":
            return "RECOMMEND_DOCTOR_VISIT"
        if current_state.get("severity") == "moderate":
            return "SCHEDULE_TELEMEDICINE"
        return "RECOMMEND_SELF_CARE"

    def step(self, action):
        if self.current_case is None:
            raise ValueError("Environment not reset. Call reset() first.")
        if self.done:
            raise ValueError("Episode already completed. Reset the environment for a new case.")
        if action not in ACTION_CATALOG:
            raise ValueError(f"Unsupported action '{action}'. Choose from: {', '.join(ACTION_CATALOG)}")

        reward = compute_reward(self.current_case, action)
        self.total_reward += reward
        self.history.append(action)
        self.last_explanation = explain_action(self.current_case, action)

        if action in TERMINAL_ACTIONS or len(self.history) >= 4:
            self.done = True

        info = {
            "correct_action": self.current_case["correct_action"],
            "recommended_path": self.current_case["recommended_path"],
            "explanation": self.last_explanation,
            "expert_action": self.expert_policy(),
        }
        return self.state(), reward, self.done, info

    def benchmark(self, episodes=25):
        total_reward = 0
        solved = 0
        urgency_breakdown = {"critical": 0, "high": 0, "moderate": 0, "low": 0}

        for _ in range(episodes):
            state = self.reset()
            urgency_breakdown[state["urgency"]] += 1
            action = self.expert_policy(state)
            _, reward, done, info = self.step(action)
            total_reward += reward
            solved += int(done and action == info["correct_action"])

        return {
            "episodes": episodes,
            "average_reward": round(total_reward / episodes, 2),
            "successful_triage_rate": round((solved / episodes) * 100, 1),
            "urgency_breakdown": urgency_breakdown,
        }


Env = HealthTriageEnv
