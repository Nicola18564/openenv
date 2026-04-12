from __future__ import annotations

from typing import Dict, Iterable

TERMINAL_ACTIONS = {
    "APPLY_JOB",
}

INFO_ACTIONS = {
    "ANALYZE_COMPANY",
    "EXTRACT_SKILLS",
    "UPDATE_SKILL_MAP",
    "TRACK_PROGRESS",
    "GATHER_CERTIFICATION",
    "REVIEW_FAILURE",
    "EXPLORE_HACKATHON",
    "VALIDATE_READINESS",
}


def _thresholds(case: Dict) -> Dict:
    return case.get(
        "proof_targets",
        {
            "projects": 2,
            "dsa": 60,
            "resume": 25,
            "interview": 20,
            "branding": 20,
            "testing": 25,
            "readiness": 60,
        },
    )


def _project_count(case: Dict) -> int:
    return len(case.get("projects", []))


def _skill_average(case: Dict) -> float:
    skills = case.get("skill_levels") or case.get("initial_state") or {}
    if not skills:
        return 0.0
    return sum(skills.values()) / max(len(skills), 1)


def assess_case(case: Dict) -> Dict:
    thresholds = _thresholds(case)
    skills = case.get("skill_levels") or case.get("initial_state") or {}
    project_count = _project_count(case)
    brand_score = case.get("brand_score", 0)
    resume_score = case.get("resume_score", 0)
    interview_score = case.get("interview_score", 0)
    testing_score = case.get("testing_score", 0)
    progress_score = case.get("progress_score", 0)

    skill_gap = max(0.0, 100 - _skill_average(case))
    project_gap = max(0, thresholds["projects"] - project_count)
    resume_gap = max(0, thresholds["resume"] - resume_score)
    interview_gap = max(0, thresholds["interview"] - interview_score)
    brand_gap = max(0, thresholds["branding"] - brand_score)

    readiness = max(
        0.0,
        min(
            100.0,
            (sum(skills.values()) / max(len(skills), 1))
            * 0.35
            + project_count * 15
            + testing_score * 0.1
            + progress_score * 0.1
            + brand_score * 0.1
            + resume_score * 0.15
            + interview_score * 0.15,
        ),
    )

    if readiness >= 65:
        readiness_state = "ready"
    elif readiness >= 50:
        readiness_state = "almost_ready"
    else:
        readiness_state = "needs_work"

    return {
        "readiness_score": round(readiness, 1),
        "readiness_state": readiness_state,
        "skill_gap": round(skill_gap, 1),
        "project_gap": project_gap,
        "resume_gap": resume_gap,
        "interview_gap": interview_gap,
        "brand_gap": brand_gap,
        "projects": project_count,
    }


def recommend_action(case: Dict) -> str:
    assessment = assess_case(case)
    focus = case.get("focus_modules", [])
    stage = case.get("stage", "")
    thresholds = _thresholds(case)
    history = [item.get("action") for item in case.get("history", []) if item.get("action")]

    def action_taken(prefix: str) -> bool:
        return any(action == prefix for action in history)

    if case.get("feedback_pending"):
        return "REVIEW_FAILURE"
    if assessment["readiness_state"] == "ready":
        return "APPLY_JOB"
    if "ANALYZE_COMPANY" not in history:
        return "ANALYZE_COMPANY"
    if "EXTRACT_SKILLS" not in history:
        return "EXTRACT_SKILLS"
    if "UPDATE_SKILL_MAP" not in history:
        return "UPDATE_SKILL_MAP"
    if not any(action.startswith("LEARN_") for action in history):
        if "LEARN_AI" in focus or case.get("company_type") == "product":
            return "LEARN_AI"
        if "LEARN_BACKEND" in focus or case.get("company_type") == "service":
            return "LEARN_BACKEND"
        return "LEARN_DSA"
    if len(case.get("projects", [])) < thresholds["projects"]:
        if "BUILD_AI_PROJECT" in focus or case.get("company_type") == "product":
            return "BUILD_AI_PROJECT"
        if "BUILD_BACKEND_PROJECT" in focus or case.get("company_type") == "service":
            return "BUILD_BACKEND_PROJECT"
        return "BUILD_FULLSTACK_PROJECT"
    if case.get("testing_score", 0) < thresholds["testing"]:
        return "WRITE_TESTS"
    if case.get("progress_score", 0) < 30:
        return "TRACK_PROGRESS"
    if case.get("brand_score", 0) < thresholds["branding"]:
        return "PUBLISH_GITHUB"
    if case.get("resume_score", 0) < thresholds["resume"]:
        return "OPTIMIZE_RESUME"
    if case.get("interview_score", 0) < thresholds["interview"]:
        return "PRACTICE_INTERVIEW"
    if stage == "feedback":
        return "REVIEW_FAILURE"
    if stage == "opportunity":
        return "EXPLORE_HACKATHON"
    return "VALIDATE_READINESS" if not case.get("proof_ready") else "APPLY_JOB"


def _module_breakdown(action: str, case: Dict, assessment: Dict, recommended: str) -> Dict[str, int]:
    breakdown = {
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
    }

    if action == "ANALYZE_COMPANY":
        breakdown["analysis"] = 5
    elif action in {"EXTRACT_SKILLS", "UPDATE_SKILL_MAP"}:
        breakdown["analysis"] = 2
        breakdown["skill"] = 3
    elif action in {"LEARN_PYTHON", "LEARN_DSA", "LEARN_AI", "LEARN_BACKEND"}:
        breakdown["skill"] = 6
        if action == "LEARN_AI":
            breakdown["ai"] = 2
        if action == "LEARN_BACKEND":
            breakdown["project"] = 1
    elif action in {"BUILD_AI_PROJECT", "BUILD_BACKEND_PROJECT", "BUILD_FULLSTACK_PROJECT"}:
        breakdown["project"] = 7
        if action == "BUILD_AI_PROJECT":
            breakdown["ai"] = 3
        if action == "BUILD_BACKEND_PROJECT":
            breakdown["skill"] = 2
        if action == "BUILD_FULLSTACK_PROJECT":
            breakdown["branding"] = 1
    elif action == "WRITE_TESTS":
        breakdown["testing"] = 7
    elif action == "TRACK_PROGRESS":
        breakdown["tracking"] = 6
    elif action == "PUBLISH_GITHUB":
        breakdown["branding"] = 7
    elif action == "OPTIMIZE_RESUME":
        breakdown["resume"] = 8
    elif action == "PRACTICE_INTERVIEW":
        breakdown["interview"] = 8
    elif action == "GATHER_CERTIFICATION":
        breakdown["branding"] = 2
        breakdown["proof"] = 2
    elif action == "REVIEW_FAILURE":
        breakdown["feedback"] = 7
        breakdown["tracking"] = 2
    elif action == "EXPLORE_HACKATHON":
        breakdown["branding"] = 4
        breakdown["analysis"] = 2
    elif action == "VALIDATE_READINESS":
        breakdown["proof"] = 10
        breakdown["tracking"] = 1
    elif action == "APPLY_JOB":
        breakdown["application"] = 10

    if action == recommended:
        breakdown["proof"] += 2
        breakdown["tracking"] += 1

    if assessment["readiness_state"] == "ready" and action == "APPLY_JOB":
        breakdown["application"] += 4
        breakdown["proof"] += 4

    return breakdown


def score_action(case: Dict, action: str) -> Dict:
    assessment = assess_case(case)
    recommended = recommend_action(case)
    thresholds = _thresholds(case)
    proof_ready = (
        _project_count(case) >= thresholds["projects"]
        and case.get("resume_score", 0) >= thresholds["resume"]
        and case.get("interview_score", 0) >= thresholds["interview"]
        and case.get("brand_score", 0) >= thresholds["branding"]
        and case.get("testing_score", 0) >= thresholds["testing"]
        and assessment["readiness_score"] >= thresholds["readiness"]
    )

    breakdown = _module_breakdown(action, case, assessment, recommended)
    rationale = []
    verdict = "partial"
    resolution_quality = "partial"

    if action == recommended:
        verdict = "optimal"
        resolution_quality = "safe_progress"
        rationale.append("The action matches the next highest-value growth step.")
    elif action in INFO_ACTIONS:
        verdict = "reasonable"
        resolution_quality = "information_gain"
        rationale.append("This action collects information or tracks progress.")

    if action == "APPLY_JOB" and not proof_ready:
        breakdown["application"] = -12
        breakdown["proof"] = -2
        verdict = "unsafe"
        resolution_quality = "applied_too_early"
        rationale.append("Applying before the proof gates are met is unsafe.")
    elif action == "APPLY_JOB" and proof_ready:
        resolution_quality = "submission_ready"
        verdict = "optimal"
        rationale.append("Proof-based eligibility is satisfied, so applying is appropriate.")
    elif action == "VALIDATE_READINESS":
        rationale.append("Readiness validation helps decide whether to apply now or keep building.")
        if proof_ready:
            resolution_quality = "proof_ready"
            verdict = "optimal"

    if action == "REVIEW_FAILURE":
        rationale.append("Reflection turns a setback into a measurable improvement loop.")
    if action == "TRACK_PROGRESS":
        rationale.append("Weekly tracking helps make growth visible.")
    if action == "PUBLISH_GITHUB":
        rationale.append("Public proof strengthens personal branding.")
    if action == "PRACTICE_INTERVIEW":
        rationale.append("Interview drills improve communication and confidence.")

    reward = sum(breakdown.values())
    if action == recommended and action != "APPLY_JOB":
        reward = max(reward, 6)
    if action == recommended and action == "APPLY_JOB" and proof_ready:
        reward = max(reward, 12)

    care_plan = {
        "ANALYZE_COMPANY": "research_market",
        "EXTRACT_SKILLS": "map_requirements",
        "UPDATE_SKILL_MAP": "refresh_plan",
        "LEARN_PYTHON": "skill_building",
        "LEARN_DSA": "problem_solving",
        "LEARN_AI": "ai_depth",
        "LEARN_BACKEND": "backend_depth",
        "BUILD_AI_PROJECT": "portfolio_proof",
        "BUILD_BACKEND_PROJECT": "system_proof",
        "BUILD_FULLSTACK_PROJECT": "delivery_proof",
        "WRITE_TESTS": "quality_proof",
        "TRACK_PROGRESS": "weekly_tracking",
        "PUBLISH_GITHUB": "public_brand",
        "OPTIMIZE_RESUME": "resume_polish",
        "APPLY_JOB": "strategic_application",
        "PRACTICE_INTERVIEW": "interview_readiness",
        "REVIEW_FAILURE": "feedback_loop",
        "EXPLORE_HACKATHON": "opportunity_search",
        "GATHER_CERTIFICATION": "optional_supporting_proof",
        "VALIDATE_READINESS": "proof_gate",
    }.get(action, "undetermined")

    return {
        "assessment": assessment,
        "recommended_action": recommended,
        "correct_action": case.get("correct_action", recommended),
        "reward": float(reward),
        "reward_breakdown": breakdown,
        "verdict": verdict,
        "resolution_quality": resolution_quality,
        "care_plan": care_plan,
        "proof_ready": proof_ready,
        "rationale": " ".join(rationale) if rationale else "No additional rationale.",
    }


def explain_action(case: Dict, action: str) -> Dict:
    return score_action(case, action)


def compute_reward(case: Dict, action: str) -> float:
    return score_action(case, action)["reward"]
