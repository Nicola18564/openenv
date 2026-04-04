import copy
import json
import logging
import random
from pathlib import Path


LOGGER = logging.getLogger(__name__)
TERMINAL_ACTIONS = {
    "ESCALATE_EMERGENCY",
    "RECOMMEND_CLINIC",
    "RECOMMEND_DOCTOR_VISIT",
    "RECOMMEND_SELF_CARE",
}
DISTRESS_STATES = {"distressed", "panicked", "worried", "anxious", "overwhelmed"}


def load_scenarios():
    scenario_path = Path(__file__).with_name("scenarios.json")
    with scenario_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class HealthTriageEnv:
    ACTIONS = [
        "ASK_FOLLOWUP",
        "PROVIDE_SUPPORT_MESSAGE",
        "RECOMMEND_SELF_CARE",
        "RECOMMEND_CLINIC",
        "RECOMMEND_DOCTOR_VISIT",
        "ESCALATE_EMERGENCY",
    ]

    DEFAULT_SCENARIOS = load_scenarios()

    def __init__(self, scenario=None, seed=None, max_steps=4):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        selected = scenario or self.rng.choice(self.DEFAULT_SCENARIOS)
        self.initial_scenario = self._normalize_scenario(selected)
        self.scenario = {}
        self.step_count = 0
        self.done = False
        self.history = []
        self.total_reward = 0.0
        self.flags = {}
        LOGGER.info("Environment initialized for case '%s'", self.initial_scenario["name"])
        self.reset()

    def reset(self):
        self.scenario = copy.deepcopy(self.initial_scenario)
        self.step_count = 0
        self.done = False
        self.history = []
        self.total_reward = 0.0
        self.flags = {
            "followup_done": False,
            "support_done": False,
            "last_action": None,
        }
        LOGGER.info("Environment reset for case '%s'", self.scenario["name"])
        return self._build_state()

    def step(self, action):
        if self.done:
            LOGGER.warning("Step called after episode completion")
            return self._build_state(), 0.0, True, {
                "message": "Episode already finished.",
                "history": list(self.history),
                "resolution_quality": "completed",
            }

        if action not in self.ACTIONS:
            LOGGER.error("Invalid action received: %s", action)
            return self._build_state(), -2.0, False, {
                "error": "Invalid action.",
                "available_actions": self.available_actions(),
            }

        assessment = self._assess_case()
        reward, reward_breakdown = self._compute_reward(action, assessment)

        self.step_count += 1
        self.total_reward = round(self.total_reward + reward, 1)
        self._apply_transition(action)

        if action in TERMINAL_ACTIONS or self.step_count >= self.max_steps:
            self.done = True

        alignment = self._action_alignment(action, assessment)
        resolution_quality = self._resolution_quality(action, assessment)
        verdict = self._verdict_from_reward(reward, resolution_quality)
        care_plan = self._care_plan(assessment)
        rationale = self._build_rationale(action, assessment, reward_breakdown, resolution_quality)

        history_entry = {
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "alignment": alignment,
        }
        self.history.append(history_entry)
        LOGGER.info(
            "Step=%s action=%s reward=%s verdict=%s",
            self.step_count,
            action,
            reward,
            verdict,
        )

        current_assessment = self._assess_case()
        info = {
            "step_count": self.step_count,
            "action_taken": action,
            "action_alignment": alignment,
            "verdict": verdict,
            "resolution_quality": resolution_quality,
            "reward_breakdown": reward_breakdown,
            "rationale": rationale,
            "care_plan": care_plan,
            "correct_action": assessment["recommended_action"],
            "suggested_next_action": None if self.done else self.expert_policy(),
            "risk_score": current_assessment["risk_score"],
            "urgency": current_assessment["urgency"],
            "total_reward": self.total_reward,
            "history": list(self.history),
        }
        return self._build_state(), reward, self.done, info

    def expert_policy(self, state=None):
        assessment = self._assess_case()
        return assessment["preferred_action"]

    def available_actions(self):
        return list(self.ACTIONS)

    def benchmark(self, episodes=20):
        rewards = []
        solved = 0
        urgency_breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for _ in range(episodes):
            self.initial_scenario = self._normalize_scenario(self.rng.choice(self.DEFAULT_SCENARIOS))
            state = self.reset()
            urgency_breakdown[state["urgency"]] += 1
            episode_reward = 0.0
            info = {}

            while True:
                action = self.expert_policy(state)
                state, reward, done, info = self.step(action)
                episode_reward += reward
                if done:
                    break

            rewards.append(round(episode_reward, 1))
            solved += int(info.get("resolution_quality") == "safe_final")

        return {
            "episodes": episodes,
            "average_reward": round(sum(rewards) / len(rewards), 2),
            "successful_triage_rate": round((solved / episodes) * 100, 1),
            "urgency_breakdown": urgency_breakdown,
        }

    def _normalize_scenario(self, scenario):
        normalized = copy.deepcopy(scenario)
        normalized.setdefault("name", "Unnamed case")
        normalized.setdefault("summary", normalized.get("symptoms", "No summary available."))
        normalized.setdefault("symptoms", "No symptoms recorded")
        normalized.setdefault("age_group", "adult")
        normalized.setdefault("severity", "low")
        normalized.setdefault("rural_access", False)
        normalized.setdefault("mental_state", "neutral")
        normalized.setdefault("fall_flag", False)
        normalized.setdefault("epidemic_flag", False)
        normalized.setdefault("chronic_conditions", [])
        normalized.setdefault("red_flags", [])
        normalized["rural_access"] = bool(normalized["rural_access"])
        normalized["fall_flag"] = bool(normalized["fall_flag"])
        normalized["epidemic_flag"] = bool(normalized["epidemic_flag"])
        return normalized

    def _assess_case(self):
        severity_base = {
            "low": 18,
            "medium": 44,
            "high": 72,
        }
        risk_score = severity_base.get(self.scenario["severity"], 18)
        risk_score += 20 if self.scenario["fall_flag"] else 0
        risk_score += 8 if self.scenario["rural_access"] else 0
        risk_score += 7 if self.scenario["epidemic_flag"] else 0
        risk_score += 8 if self.scenario["age_group"] == "elderly" else 4 if self.scenario["age_group"] == "child" else 0
        risk_score += 5 if self.scenario["mental_state"] in DISTRESS_STATES else 0
        risk_score += min(8, len(self.scenario.get("chronic_conditions", [])) * 3)
        risk_score += min(12, len(self.scenario.get("red_flags", [])) * 4)
        risk_score = min(100, risk_score)

        if risk_score >= 85:
            urgency = "critical"
        elif risk_score >= 60:
            urgency = "high"
        elif risk_score >= 35:
            urgency = "medium"
        else:
            urgency = "low"

        recommended_action = self._recommended_action(risk_score)
        support_relevant = (
            self.scenario["mental_state"] in DISTRESS_STATES
            or self.scenario["age_group"] in {"child", "elderly"}
        )
        followup_helpful = (
            urgency in {"medium", "high"}
            and not self.scenario["fall_flag"]
            and not self.flags["followup_done"]
        )

        preferred_action = recommended_action
        if recommended_action != "ESCALATE_EMERGENCY":
            if support_relevant and urgency == "low" and not self.flags["support_done"]:
                preferred_action = "PROVIDE_SUPPORT_MESSAGE"
            elif followup_helpful:
                preferred_action = "ASK_FOLLOWUP"

        return {
            "risk_score": risk_score,
            "urgency": urgency,
            "recommended_action": recommended_action,
            "preferred_action": preferred_action,
            "support_relevant": support_relevant,
            "followup_helpful": followup_helpful,
            "rural_access": self.scenario["rural_access"],
        }

    def _recommended_action(self, risk_score):
        if self.scenario["fall_flag"] or risk_score >= 85:
            return "ESCALATE_EMERGENCY"
        if self.scenario["severity"] == "high":
            return "RECOMMEND_CLINIC" if self.scenario["rural_access"] else "RECOMMEND_DOCTOR_VISIT"
        if self.scenario["severity"] == "medium":
            if self.scenario["rural_access"] or self.scenario["epidemic_flag"]:
                return "RECOMMEND_CLINIC"
            return "RECOMMEND_DOCTOR_VISIT"
        return "RECOMMEND_SELF_CARE"

    def _compute_reward(self, action, assessment):
        breakdown = {
            "safety": 0.0,
            "sequence": 0.0,
            "access": 0.0,
            "empathy": 0.0,
            "efficiency": 0.0,
        }
        repeated_action = action == self.flags["last_action"]
        recommended_action = assessment["recommended_action"]
        preferred_action = assessment["preferred_action"]
        urgency = assessment["urgency"]

        if recommended_action == "ESCALATE_EMERGENCY":
            if action == "ESCALATE_EMERGENCY":
                breakdown["safety"] = 6.0
            elif action in {"ASK_FOLLOWUP", "PROVIDE_SUPPORT_MESSAGE"}:
                breakdown["safety"] = -3.5
            else:
                breakdown["safety"] = -6.5
        elif urgency in {"high", "medium"}:
            if action == recommended_action:
                breakdown["safety"] = 4.8 if urgency == "high" else 4.0
            elif action in {"RECOMMEND_CLINIC", "RECOMMEND_DOCTOR_VISIT"}:
                breakdown["safety"] = 1.0 if urgency == "medium" else -0.8
            elif action == "ASK_FOLLOWUP":
                breakdown["safety"] = 1.3 if assessment["followup_helpful"] else -0.5
            elif action == "PROVIDE_SUPPORT_MESSAGE":
                breakdown["safety"] = 0.4 if assessment["support_relevant"] else -0.5
            elif action == "RECOMMEND_SELF_CARE":
                breakdown["safety"] = -4.5
        else:
            if action == "RECOMMEND_SELF_CARE":
                breakdown["safety"] = 3.5
            elif action == "PROVIDE_SUPPORT_MESSAGE":
                breakdown["safety"] = 1.4
            elif action == "ASK_FOLLOWUP":
                breakdown["safety"] = 1.0
            elif action == "ESCALATE_EMERGENCY":
                breakdown["safety"] = -4.0
            else:
                breakdown["safety"] = -1.5

        if action == preferred_action:
            breakdown["sequence"] = 2.0 if action in TERMINAL_ACTIONS else 1.8
        elif action == recommended_action:
            breakdown["sequence"] = 1.2
        elif action in {"ASK_FOLLOWUP", "PROVIDE_SUPPORT_MESSAGE"}:
            breakdown["sequence"] = 0.3
        else:
            breakdown["sequence"] = -1.2

        if assessment["rural_access"]:
            if action == "RECOMMEND_CLINIC":
                breakdown["access"] = 1.5
            elif action == "RECOMMEND_DOCTOR_VISIT":
                breakdown["access"] = 0.5
            elif action == "RECOMMEND_SELF_CARE" and urgency != "low":
                breakdown["access"] = -1.0
        elif action == "RECOMMEND_CLINIC" and urgency == "low":
            breakdown["access"] = -0.4

        if assessment["support_relevant"]:
            if action == "PROVIDE_SUPPORT_MESSAGE":
                breakdown["empathy"] = 1.5
            elif action == "ASK_FOLLOWUP":
                breakdown["empathy"] = 0.5
            elif action in TERMINAL_ACTIONS and urgency == "low" and not self.flags["support_done"]:
                breakdown["empathy"] = -0.4
        elif action == "PROVIDE_SUPPORT_MESSAGE":
            breakdown["empathy"] = 0.2

        if repeated_action:
            breakdown["efficiency"] = -1.5
        elif action in {"ASK_FOLLOWUP", "PROVIDE_SUPPORT_MESSAGE"} and self.step_count >= 1:
            breakdown["efficiency"] = -0.4
        elif action in TERMINAL_ACTIONS and self.step_count <= 1:
            breakdown["efficiency"] = 0.8

        total_reward = round(sum(breakdown.values()), 1)
        total_reward = max(-8.0, min(10.0, total_reward))
        return total_reward, {name: round(value, 1) for name, value in breakdown.items()}

    def _action_alignment(self, action, assessment):
        if action == assessment["preferred_action"] == assessment["recommended_action"]:
            return "on_path_final"
        if action == assessment["preferred_action"] and action not in TERMINAL_ACTIONS:
            return "on_path_progress"
        if action == assessment["recommended_action"]:
            return "safe_but_direct"
        if assessment["recommended_action"] == "ESCALATE_EMERGENCY" and action != "ESCALATE_EMERGENCY":
            return "under_triage"
        if assessment["urgency"] == "low" and action in {"RECOMMEND_CLINIC", "RECOMMEND_DOCTOR_VISIT", "ESCALATE_EMERGENCY"}:
            return "over_triage"
        if action == "PROVIDE_SUPPORT_MESSAGE" and assessment["support_relevant"]:
            return "supportive_but_incomplete"
        return "off_path"

    def _resolution_quality(self, action, assessment):
        if action in TERMINAL_ACTIONS:
            if action == assessment["recommended_action"]:
                return "safe_final"
            if assessment["recommended_action"] == "ESCALATE_EMERGENCY" and action != "ESCALATE_EMERGENCY":
                return "unsafe_final"
            if assessment["urgency"] == "low" and action in {"RECOMMEND_CLINIC", "RECOMMEND_DOCTOR_VISIT", "ESCALATE_EMERGENCY"}:
                return "over_escalated"
            return "mismatched_final"
        if self.done:
            return "incomplete"
        return "ongoing"

    def _care_plan(self, assessment):
        ordered = [assessment["preferred_action"], assessment["recommended_action"]]
        deduped = []
        for action in ordered:
            if action not in deduped:
                deduped.append(action)
        return deduped

    def _build_rationale(self, action, assessment, reward_breakdown, resolution_quality):
        parts = []

        if assessment["recommended_action"] == "ESCALATE_EMERGENCY":
            parts.append("This case carries emergency-level risk and should not be delayed.")
        elif assessment["urgency"] == "high":
            parts.append("This case needs prompt in-person escalation.")
        elif assessment["urgency"] == "medium":
            parts.append("This case benefits from added context and clinician follow-up.")
        else:
            parts.append("This is a lower-risk case where reassurance and self-care can be appropriate.")

        if self.scenario["rural_access"]:
            parts.append("Rural access barriers matter, so reachable in-person care gets extra credit.")

        if action == "ASK_FOLLOWUP":
            parts.append("Follow-up questions improve decision confidence before final routing.")
        elif action == "PROVIDE_SUPPORT_MESSAGE":
            parts.append("Supportive language improves trust and adherence, especially in anxious cases.")
        else:
            parts.append(f"{action.replace('_', ' ').title()} was evaluated as a final routing decision.")

        if resolution_quality == "safe_final":
            parts.append("The final disposition matches the safest route for this case.")
        elif resolution_quality == "unsafe_final":
            parts.append("The final disposition under-triaged a case that needed emergency care.")
        elif resolution_quality == "over_escalated":
            parts.append("The final disposition escalated more than the case required.")
        elif resolution_quality == "incomplete":
            parts.append("The episode ended before a safe final routing decision was made.")

        dominant_component = max(reward_breakdown, key=lambda name: abs(reward_breakdown[name]))
        parts.append(
            f"The largest reward signal came from {dominant_component}, scored at {reward_breakdown[dominant_component]:+.1f}."
        )
        return " ".join(parts)

    def _verdict_from_reward(self, reward, resolution_quality):
        if resolution_quality == "unsafe_final":
            return "unsafe"
        if reward >= 7:
            return "excellent"
        if reward >= 4:
            return "strong"
        if reward > 0:
            return "reasonable"
        if reward > -3:
            return "weak"
        return "unsafe"

    def _apply_transition(self, action):
        if action == "ASK_FOLLOWUP":
            self.flags["followup_done"] = True
            self.scenario["mental_state"] = "better described"
        elif action == "PROVIDE_SUPPORT_MESSAGE":
            self.flags["support_done"] = True
            self.scenario["mental_state"] = "calmer"
        elif action == "ESCALATE_EMERGENCY":
            self.scenario["mental_state"] = "urgent attention needed"
        elif action == "RECOMMEND_CLINIC":
            self.scenario["mental_state"] = "referred for urgent clinic review"
        elif action == "RECOMMEND_DOCTOR_VISIT":
            self.scenario["mental_state"] = "guided toward medical review"
        elif action == "RECOMMEND_SELF_CARE":
            self.scenario["mental_state"] = "reassured with home care guidance"

        self.flags["last_action"] = action

    def _build_state(self):
        assessment = self._assess_case()
        return {
            "name": self.scenario["name"],
            "summary": self.scenario["summary"],
            "symptoms": self.scenario["symptoms"],
            "age_group": self.scenario["age_group"],
            "severity": self.scenario["severity"],
            "rural_access": self.scenario["rural_access"],
            "mental_state": self.scenario["mental_state"],
            "fall_flag": self.scenario["fall_flag"],
            "epidemic_flag": self.scenario["epidemic_flag"],
            "chronic_conditions": list(self.scenario.get("chronic_conditions", [])),
            "red_flags": list(self.scenario.get("red_flags", [])),
            "risk_score": assessment["risk_score"],
            "urgency": assessment["urgency"],
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "context_collected": self.flags["followup_done"],
            "support_provided": self.flags["support_done"],
            "history": list(self.history),
            "done": self.done,
        }


HealthTriageEnvironment = HealthTriageEnv
