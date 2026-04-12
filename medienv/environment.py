from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from medienv.grader import assess_case, recommend_action, score_action
from medienv.tasks import ACTION_CATALOG, SCENARIOS


PROOF_SKILLS = {
    "python": "LEARN_PYTHON",
    "dsa": "LEARN_DSA",
    "ai": "LEARN_AI",
    "backend": "LEARN_BACKEND",
    "web": "BUILD_FULLSTACK_PROJECT",
    "github": "PUBLISH_GITHUB",
    "testing": "WRITE_TESTS",
    "deployment": "BUILD_BACKEND_PROJECT",
    "communication": "PRACTICE_INTERVIEW",
    "consistency": "TRACK_PROGRESS",
    "interviewing": "PRACTICE_INTERVIEW",
    "resume": "OPTIMIZE_RESUME",
    "branding": "PUBLISH_GITHUB",
}


def load_scenarios() -> List[Dict[str, Any]]:
    return copy.deepcopy(SCENARIOS)


def _deepcopy_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return copy.deepcopy(value or {})


class PlacementIntelligenceEnv:
    ACTIONS = tuple(ACTION_CATALOG)

    def __init__(self, scenario: Optional[Dict[str, Any] | str] = None, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._scenarios = load_scenarios()
        self.initial_scenario = self._resolve_scenario(scenario)
        self.scenario: Dict[str, Any] = {}
        self.skill_levels: Dict[str, int] = {}
        self.projects: List[str] = []
        self.company_analysis_score = 0
        self.testing_score = 0
        self.progress_score = 0
        self.brand_score = 0
        self.resume_score = 0
        self.interview_score = 0
        self.application_score = 0
        self.proof_score = 0
        self.applications_submitted = 0
        self.feedback_pending = False
        self.step_count = 0
        self.done = False
        self.history: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self.last_result: Dict[str, Any] = {}
        self._state_cache: Dict[str, Any] = {}
        self.reset(self.initial_scenario)

    def _resolve_scenario(self, scenario: Optional[Dict[str, Any] | str]) -> Dict[str, Any]:
        if scenario is None:
            return copy.deepcopy(self.rng.choice(self._scenarios))
        if isinstance(scenario, str):
            matches = [item for item in self._scenarios if item["name"] == scenario]
            if not matches:
                raise ValueError(f"Unknown scenario name: {scenario}")
            return copy.deepcopy(matches[0])
        return copy.deepcopy(scenario)

    def available_actions(self) -> List[str]:
        return list(self.ACTIONS)

    def _reset_runtime(self) -> None:
        self.skill_levels = copy.deepcopy(self.scenario.get("initial_state", {}))
        self.projects = list(self.scenario.get("projects", []))
        self.company_analysis_score = 0
        self.testing_score = 0
        self.progress_score = 0
        self.brand_score = 0
        self.resume_score = 0
        self.interview_score = 0
        self.application_score = 0
        self.proof_score = 0
        self.applications_submitted = 0
        self.feedback_pending = False
        self.step_count = 0
        self.done = False
        self.history = []
        self.total_reward = 0.0
        self.last_result = {}
        self._state_cache = {}

    def _normalized_skill(self, skill: str) -> str:
        return skill.strip().lower().replace("-", "_").replace(" ", "_")

    def _history_actions(self, state: Optional[Dict[str, Any]] = None) -> List[str]:
        snapshot = state or self._build_state()
        return [item.get("action") for item in snapshot.get("history", []) if item.get("action")]

    def _proof_targets(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        snapshot = state or self._build_state()
        thresholds = snapshot.get("proof_targets", {})
        return {
            "projects": int(thresholds.get("projects", 2)),
            "resume": int(thresholds.get("resume", 25)),
            "interview": int(thresholds.get("interview", 20)),
            "branding": int(thresholds.get("branding", 20)),
            "testing": int(thresholds.get("testing", 25)),
            "readiness": float(thresholds.get("readiness", 60)),
        }

    def _skill_average(self) -> float:
        if not self.skill_levels:
            return 0.0
        return sum(self.skill_levels.values()) / max(len(self.skill_levels), 1)

    def _proof_ready(self, state: Optional[Dict[str, Any]] = None) -> bool:
        snapshot = state or self._build_state()
        thresholds = self._proof_targets(snapshot)
        return (
            snapshot.get("project_count", 0) >= thresholds["projects"]
            and snapshot.get("resume_score", 0) >= thresholds["resume"]
            and snapshot.get("interview_score", 0) >= thresholds["interview"]
            and snapshot.get("brand_score", 0) >= thresholds["branding"]
            and snapshot.get("testing_score", 0) >= thresholds["testing"]
            and snapshot.get("readiness_score", 0) >= thresholds["readiness"]
        )

    def _pick_learning_action(self, current_state: Dict[str, Any]) -> Optional[str]:
        skill_levels = current_state.get("skill_levels", {})
        required_skills = [self._normalized_skill(skill) for skill in current_state.get("required_skills", [])]
        learned_actions = {action for action in self._history_actions(current_state) if action.startswith("LEARN_")}
        candidates: List[tuple[int, str]] = []

        for token in required_skills:
            action = PROOF_SKILLS.get(token)
            if action and action.startswith("LEARN_") and action not in learned_actions:
                candidates.append((int(skill_levels.get(token, 0)), action))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            if candidates[0][0] < 60:
                return candidates[0][1]

        company_type = current_state.get("company_type")
        fallback = {
            "product": "LEARN_AI",
            "service": "LEARN_BACKEND",
            "mixed": "LEARN_DSA",
            "exploratory": "LEARN_PYTHON",
        }.get(company_type, "LEARN_DSA")

        if fallback not in learned_actions and skill_levels.get(fallback.split("_", 1)[1].lower(), 0) < 60:
            return fallback
        return None

    def _apply_action_effects(self, action: str) -> None:
        if action == "ANALYZE_COMPANY":
            self.company_analysis_score += 20
            self.progress_score += 3
        elif action == "EXTRACT_SKILLS":
            self.company_analysis_score += 14
            self.progress_score += 4
        elif action == "UPDATE_SKILL_MAP":
            self.company_analysis_score += 10
            self.progress_score += 5
        elif action in {"LEARN_PYTHON", "LEARN_DSA", "LEARN_AI", "LEARN_BACKEND"}:
            key = action.split("_", 1)[1].lower()
            self.skill_levels[key] = min(100, self.skill_levels.get(key, 0) + 15)
            self.progress_score += 6
            if action == "LEARN_AI":
                self.brand_score += 1
        elif action == "BUILD_AI_PROJECT":
            if "AI project" not in self.projects:
                self.projects.append("AI project")
            self.skill_levels["python"] = min(100, self.skill_levels.get("python", 0) + 6)
            self.skill_levels["ai"] = min(100, self.skill_levels.get("ai", 0) + 8)
            self.testing_score += 8
            self.progress_score += 10
            self.brand_score += 4
        elif action == "BUILD_BACKEND_PROJECT":
            if "Backend project" not in self.projects:
                self.projects.append("Backend project")
            self.skill_levels["backend"] = min(100, self.skill_levels.get("backend", 0) + 8)
            self.testing_score += 8
            self.progress_score += 10
            self.brand_score += 3
        elif action == "BUILD_FULLSTACK_PROJECT":
            if "Full stack project" not in self.projects:
                self.projects.append("Full stack project")
            self.skill_levels["web"] = min(100, self.skill_levels.get("web", 0) + 8)
            self.skill_levels["backend"] = min(100, self.skill_levels.get("backend", 0) + 5)
            self.brand_score += 5
            self.testing_score += 5
            self.progress_score += 10
        elif action == "WRITE_TESTS":
            self.testing_score += 20
            self.progress_score += 5
        elif action == "TRACK_PROGRESS":
            self.progress_score += 18
            self.proof_score += 5
        elif action == "PUBLISH_GITHUB":
            self.brand_score += 20
            self.resume_score += 5
        elif action == "OPTIMIZE_RESUME":
            self.resume_score += 20
            self.brand_score += 5
        elif action == "APPLY_JOB":
            self.application_score += 25
            self.applications_submitted += 1
            self.proof_score += 8
        elif action == "PRACTICE_INTERVIEW":
            self.interview_score += 20
            self.progress_score += 5
        elif action == "REVIEW_FAILURE":
            self.feedback_pending = False
            self.progress_score += 10
            self.resume_score += 4
            self.brand_score += 4
            self.proof_score += 6
        elif action == "EXPLORE_HACKATHON":
            self.brand_score += 8
            self.progress_score += 5
            self.company_analysis_score += 4
        elif action == "GATHER_CERTIFICATION":
            self.brand_score += 10
            self.proof_score += 12
        elif action == "VALIDATE_READINESS":
            self.proof_score += 15
            self.progress_score += 3

    def _build_state(self) -> Dict[str, Any]:
        state = {
            "scenario_id": self.scenario.get("id"),
            "scenario_name": self.scenario.get("name"),
            "case_id": self.scenario.get("id"),
            "title": self.scenario.get("title"),
            "name": self.scenario.get("name"),
            "summary": self.scenario.get("summary"),
            "target_company": self.scenario.get("target_company"),
            "company_type": self.scenario.get("company_type"),
            "role": self.scenario.get("role"),
            "stage": self.scenario.get("stage"),
            "focus_modules": list(self.scenario.get("focus_modules", [])),
            "required_skills": list(self.scenario.get("required_skills", [])),
            "skill_gaps": list(self.scenario.get("skill_gaps", [])),
            "initial_state": _deepcopy_dict(self.scenario.get("initial_state")),
            "proof_targets": _deepcopy_dict(self.scenario.get("proof_targets")),
            "skill_levels": copy.deepcopy(self.skill_levels),
            "projects": list(self.projects),
            "project_count": len(self.projects),
            "company_analysis_score": self.company_analysis_score,
            "testing_score": self.testing_score,
            "progress_score": self.progress_score,
            "brand_score": self.brand_score,
            "resume_score": self.resume_score,
            "interview_score": self.interview_score,
            "application_score": self.application_score,
            "proof_score": self.proof_score,
            "applications_submitted": self.applications_submitted,
            "feedback_pending": self.feedback_pending,
            "step_count": self.step_count,
            "done": self.done,
            "history": copy.deepcopy(self.history),
            "total_reward": self.total_reward,
            "skill_average": round(self._skill_average(), 1),
            "company_match_score": min(100, self.company_analysis_score + int(self._skill_average() / 2)),
            "readiness_score": 0.0,
            "readiness_state": "needs_work",
            "proof_ready": False,
            "recommended_action": None,
            "correct_action": self.scenario.get("correct_action", "APPLY_JOB"),
            "current_action_suggestion": None,
            "reward_breakdown": self.last_result.get(
                "reward_breakdown",
                {
                    "analysis": 0,
                    "skill": 0,
                    "project": 0,
                    "ai": 0,
                    "testing": 0,
                    "tracking": 0,
                    "branding": 0,
                    "resume": 0,
                    "application": 0,
                    "interview": 0,
                    "proof": 0,
                    "feedback": 0,
                },
            ),
            "resolution_quality": self.last_result.get("resolution_quality", "pending"),
            "growth_plan": self.last_result.get("care_plan", "undetermined"),
        }

        assessment = assess_case(state)
        state["readiness_score"] = assessment["readiness_score"]
        state["readiness_state"] = assessment["readiness_state"]
        state["proof_ready"] = self._proof_ready(state)
        state["recommended_action"] = recommend_action(state)
        state["current_action_suggestion"] = state["recommended_action"]
        return state

    def _next_skill_action(self, current_state: Dict[str, Any]) -> str:
        history_actions = [item.get("action") for item in current_state.get("history", [])]
        proof_targets = self._proof_targets(current_state)

        if "ANALYZE_COMPANY" not in history_actions:
            return "ANALYZE_COMPANY"
        if "EXTRACT_SKILLS" not in history_actions:
            return "EXTRACT_SKILLS"
        if "UPDATE_SKILL_MAP" not in history_actions:
            return "UPDATE_SKILL_MAP"

        learning_action = self._pick_learning_action(current_state)
        if learning_action is not None:
            return learning_action

        if current_state.get("project_count", 0) < proof_targets["projects"]:
            if current_state.get("company_type") == "service":
                return "BUILD_BACKEND_PROJECT"
            if current_state.get("company_type") == "product":
                return "BUILD_AI_PROJECT"
            return "BUILD_FULLSTACK_PROJECT"
        if current_state.get("testing_score", 0) < proof_targets["testing"]:
            return "WRITE_TESTS"
        if current_state.get("progress_score", 0) < 30:
            return "TRACK_PROGRESS"
        if current_state.get("brand_score", 0) < proof_targets["branding"]:
            return "PUBLISH_GITHUB"
        if current_state.get("resume_score", 0) < proof_targets["resume"]:
            return "OPTIMIZE_RESUME"
        if current_state.get("interview_score", 0) < proof_targets["interview"]:
            return "PRACTICE_INTERVIEW"
        if current_state.get("stage") == "feedback":
            return "REVIEW_FAILURE"
        if current_state.get("stage") == "opportunity":
            return "EXPLORE_HACKATHON"
        if not self._proof_ready(current_state):
            return "VALIDATE_READINESS"
        return "APPLY_JOB"

    def expert_policy(self, state: Optional[Dict[str, Any]] = None) -> str:
        current_state = state or self._build_state()
        if current_state.get("feedback_pending"):
            return "REVIEW_FAILURE"
        if current_state.get("proof_ready"):
            return "APPLY_JOB"
        return self._next_skill_action(current_state)

    def reset(self, scenario: Optional[Dict[str, Any] | str] = None):
        if scenario is not None:
            self.initial_scenario = self._resolve_scenario(scenario)
        self.scenario = copy.deepcopy(self.initial_scenario)
        self._reset_runtime()
        return self._build_state()

    def step(self, action: str):
        current_state = self._build_state()
        if self.done:
            info = {
                "message": "Episode already finished. Reset for a new scenario.",
                "recommended_action": current_state["recommended_action"],
                "reward_breakdown": current_state["reward_breakdown"],
                "resolution_quality": "finished",
                "growth_plan": current_state["growth_plan"],
                "proof_ready": current_state["proof_ready"],
            }
            return current_state, 0.0, True, info

        if action not in self.ACTIONS:
            info = {
                "error": f"Invalid action: {action}",
                "recommended_action": current_state["recommended_action"],
                "reward_breakdown": {
                    "analysis": -1,
                    "skill": 0,
                    "project": 0,
                    "ai": 0,
                    "testing": 0,
                    "tracking": 0,
                    "branding": 0,
                    "resume": 0,
                    "application": 0,
                    "interview": 0,
                    "proof": 0,
                    "feedback": 0,
                },
                "resolution_quality": "invalid_action",
                "growth_plan": "undetermined",
                "proof_ready": current_state["proof_ready"],
            }
            return current_state, -3.0, False, info

        result = score_action(current_state, action)
        reward = float(result["reward"])
        self.total_reward += reward
        self.step_count += 1
        self._apply_action_effects(action)

        if action == "APPLY_JOB":
            if result["proof_ready"]:
                self.done = True
            else:
                self.feedback_pending = True
                self.done = True
        elif self.step_count >= 12:
            self.done = True

        history_entry = {
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "resolution_quality": result["resolution_quality"],
            "recommended_action": result["recommended_action"],
        }
        self.history.append(history_entry)
        self.last_result = result

        updated_state = self._build_state()
        info = {
            "step_count": self.step_count,
            "action_taken": action,
            "recommended_action": result["recommended_action"],
            "correct_action": result["correct_action"],
            "assessment": result["assessment"],
            "readiness_score": updated_state["readiness_score"],
            "readiness_state": updated_state["readiness_state"],
            "proof_ready": updated_state["proof_ready"],
            "resolution_quality": result["resolution_quality"],
            "reward_breakdown": result["reward_breakdown"],
            "rationale": result["rationale"],
            "growth_plan": result["care_plan"],
            "history": copy.deepcopy(self.history),
        }
        return updated_state, reward, self.done, info

    def benchmark(self, episodes: int = 20):
        total_reward = 0.0
        solved = 0
        ready_count = 0
        application_count = 0

        for _ in range(episodes):
            scenario = copy.deepcopy(self.rng.choice(self._scenarios))
            state = self.reset(scenario)
            steps = 0
            episode_reward = 0.0
            final_quality = None

            while not self.done and steps < 12:
                action = self.expert_policy(state)
                state, reward, done, info = self.step(action)
                episode_reward += reward
                steps += 1
                final_quality = info.get("resolution_quality", final_quality)
                if done:
                    break

            total_reward += episode_reward
            if state.get("proof_ready"):
                ready_count += 1
            if any(item["action"] == "APPLY_JOB" for item in self.history):
                application_count += 1
            if final_quality in {"submission_ready", "proof_ready"}:
                solved += 1

        return {
            "episodes": episodes,
            "average_reward": round(total_reward / episodes, 2) if episodes else 0.0,
            "successful_readiness_rate": round((solved / episodes) * 100, 1) if episodes else 0.0,
            "successful_triage_rate": round((solved / episodes) * 100, 1) if episodes else 0.0,
            "proof_ready_rate": round((ready_count / episodes) * 100, 1) if episodes else 0.0,
            "application_rate": round((application_count / episodes) * 100, 1) if episodes else 0.0,
        }

    def close(self):
        return None


HealthTriageEnv = PlacementIntelligenceEnv
HealthTriageEnvironment = PlacementIntelligenceEnv
