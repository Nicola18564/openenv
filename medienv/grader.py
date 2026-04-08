from __future__ import annotations

from typing import Dict

TERMINAL_ACTIONS = {
    "RECOMMEND_SELF_CARE",
    "RECOMMEND_DOCTOR_VISIT",
    "RECOMMEND_CLINIC",
    "SCHEDULE_TELEMEDICINE",
    "ESCALATE_EMERGENCY",
}

INFO_ACTIONS = {
    "ASK_FOLLOWUP",
    "REQUEST_VITALS",
    "CHECK_MEDICATION_HISTORY",
    "PROVIDE_SUPPORT_MESSAGE",
}

SUPPORT_ACTIONS = {
    "PROVIDE_SUPPORT_MESSAGE",
    "ASK_FOLLOWUP",
}


def _severity_score(severity: str) -> int:
    return {
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4,
    }.get(severity, 1)


def assess_case(case: Dict) -> Dict:
    severity = _severity_score(case["severity"])
    access_risk = int(case.get("rural_access", False)) + int(case.get("mobility_issues", False))
    social_risk = int(case.get("language_barrier", False)) + int(case.get("insurance_risk", False))
    red_flag_count = len(case.get("red_flags", []))
    risk_score = min(100, severity * 20 + access_risk * 10 + social_risk * 6 + red_flag_count * 8)

    if risk_score >= 80:
        urgency = "critical"
    elif risk_score >= 60:
        urgency = "high"
    elif risk_score >= 35:
        urgency = "moderate"
    else:
        urgency = "low"

    return {
        "risk_score": risk_score,
        "urgency": urgency,
        "access_risk": access_risk,
        "social_risk": social_risk,
        "red_flag_count": red_flag_count,
        "severity_score": severity,
    }


def recommend_action(case: Dict) -> str:
    if case.get("correct_action"):
        return case["correct_action"]

    assessment = assess_case(case)
    urgency = assessment["urgency"]
    severity = case.get("severity", "low")
    mental_state = case.get("mental_state", "")

    if urgency == "critical" or case.get("fall_flag"):
        return "ESCALATE_EMERGENCY"
    if severity == "high":
        return "RECOMMEND_CLINIC" if case.get("rural_access") else "RECOMMEND_DOCTOR_VISIT"
    if severity == "moderate":
        if case.get("rural_access") or case.get("mobility_issues"):
            return "SCHEDULE_TELEMEDICINE"
        return "RECOMMEND_DOCTOR_VISIT"
    if mental_state in {"panicked", "distressed", "worried"}:
        return "PROVIDE_SUPPORT_MESSAGE"
    return "RECOMMEND_SELF_CARE"


def _care_plan_for_action(action: str) -> str:
    return {
        "ASK_FOLLOWUP": "clarify_next_step",
        "REQUEST_VITALS": "collect_vitals",
        "CHECK_MEDICATION_HISTORY": "review_medication_history",
        "PROVIDE_SUPPORT_MESSAGE": "support_and_monitor",
        "RECOMMEND_SELF_CARE": "self_care",
        "RECOMMEND_CLINIC": "clinic_visit",
        "RECOMMEND_DOCTOR_VISIT": "doctor_visit",
        "SCHEDULE_TELEMEDICINE": "telemedicine",
        "ESCALATE_EMERGENCY": "emergency_escalation",
    }.get(action, "undetermined")


def score_action(case: Dict, action: str) -> Dict:
    assessment = assess_case(case)
    recommended = recommend_action(case)
    correct = case.get("correct_action", recommended)
    mental_state = case.get("mental_state", "")
    access_sensitive = bool(case.get("rural_access") or case.get("mobility_issues"))
    distress_present = mental_state in {"panicked", "distressed", "worried", "fatigued"}

    breakdown = {
        "safety": 0,
        "sequence": 0,
        "access": 0,
        "empathy": 0,
        "efficiency": 0,
    }
    rationale = []

    if case.get("red_flags"):
        rationale.append(f"Red flags: {', '.join(case['red_flags'])}.")
    if access_sensitive:
        rationale.append("Access barriers make follow-through more fragile.")

    if action == correct:
        breakdown["safety"] = 6 if action in TERMINAL_ACTIONS else 4
        breakdown["sequence"] = 2
        breakdown["access"] = 2 if access_sensitive else 1
        breakdown["empathy"] = 1 if distress_present or action in SUPPORT_ACTIONS else 0
        breakdown["efficiency"] = 1
        verdict = "optimal"
        resolution_quality = "safe_final" if action in TERMINAL_ACTIONS else "useful_information"
        rationale.append("The action matches the recommended care pathway.")
    elif action in INFO_ACTIONS:
        breakdown["safety"] = 1 if assessment["urgency"] != "critical" else -1
        breakdown["sequence"] = 1
        breakdown["access"] = 1 if access_sensitive else 0
        breakdown["empathy"] = 2 if action == "PROVIDE_SUPPORT_MESSAGE" and distress_present else 1
        breakdown["efficiency"] = 0
        verdict = "reasonable"
        resolution_quality = "info_gathering"
        rationale.append("Useful information gathering or supportive acknowledgement.")
    elif correct == "ESCALATE_EMERGENCY" and action != "ESCALATE_EMERGENCY":
        breakdown["safety"] = -8
        breakdown["sequence"] = -2
        breakdown["access"] = -1
        breakdown["empathy"] = 0
        breakdown["efficiency"] = -1
        verdict = "unsafe"
        resolution_quality = "unsafe_under_triage"
        rationale.append("Emergency escalation is required; anything slower is unsafe.")
    elif action in TERMINAL_ACTIONS:
        breakdown["safety"] = -4
        breakdown["sequence"] = -1
        breakdown["access"] = -1 if access_sensitive else 0
        breakdown["empathy"] = 0
        breakdown["efficiency"] = -1
        verdict = "suboptimal"
        resolution_quality = "suboptimal_final"
        rationale.append("A final recommendation was made, but it does not match the ideal pathway.")
    else:
        breakdown["safety"] = -1
        breakdown["sequence"] = 0
        breakdown["access"] = 0
        breakdown["empathy"] = 0
        breakdown["efficiency"] = 0
        verdict = "partial"
        resolution_quality = "partial"
        rationale.append("The action adds context but does not complete routing.")

    if action == "PROVIDE_SUPPORT_MESSAGE" and distress_present:
        rationale.append("Support is appropriate for the patient's emotional state.")
    elif action == "ASK_FOLLOWUP":
        rationale.append("Clarifying questions can reduce uncertainty.")

    reward = sum(breakdown.values())
    if action == correct and action in TERMINAL_ACTIONS:
        reward = max(reward, 9 + assessment["risk_score"] // 20)
    elif action == correct and action not in TERMINAL_ACTIONS:
        reward = max(reward, 3)

    return {
        "assessment": assessment,
        "recommended_action": recommended,
        "correct_action": correct,
        "reward": float(reward),
        "reward_breakdown": breakdown,
        "verdict": verdict,
        "resolution_quality": resolution_quality,
        "care_plan": _care_plan_for_action(action),
        "rationale": " ".join(rationale) if rationale else "No additional rationale.",
    }


def explain_action(case: Dict, action: str) -> Dict:
    return score_action(case, action)


def compute_reward(case: Dict, action: str) -> float:
    return score_action(case, action)["reward"]
