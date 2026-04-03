TERMINAL_ACTIONS = {
    "RECOMMEND_SELF_CARE",
    "RECOMMEND_DOCTOR_VISIT",
    "RECOMMEND_CLINIC",
    "SCHEDULE_TELEMEDICINE",
    "ESCALATE_EMERGENCY",
}


def _severity_score(severity):
    return {
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4,
    }.get(severity, 1)


def assess_case(case):
    severity = _severity_score(case["severity"])
    access_risk = int(case.get("rural_access", False)) + int(case.get("mobility_issues", False))
    social_risk = int(case.get("language_barrier", False)) + int(case.get("insurance_risk", False))
    red_flag_bonus = len(case.get("red_flags", []))
    risk_score = min(100, severity * 20 + access_risk * 10 + social_risk * 5 + red_flag_bonus * 8)

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
        "red_flag_count": red_flag_bonus,
    }


def explain_action(case, action):
    assessment = assess_case(case)
    correct = case["correct_action"]
    rationale = []

    if case.get("red_flags"):
        rationale.append(f"Red flags present: {', '.join(case['red_flags'])}.")

    if case.get("rural_access") or case.get("mobility_issues"):
        rationale.append("Access barriers increase follow-through risk.")

    if action == correct:
        rationale.append("The action matches the recommended care pathway.")
        verdict = "optimal"
    elif action == "ASK_FOLLOWUP":
        rationale.append("Follow-up questioning is safe, but may delay the final decision.")
        verdict = "reasonable"
    elif correct == "ESCALATE_EMERGENCY" and action != "ESCALATE_EMERGENCY":
        rationale.append("This case needs emergency escalation. Anything slower is unsafe.")
        verdict = "unsafe"
    elif action in TERMINAL_ACTIONS:
        rationale.append("A terminal recommendation was made, but it does not match the ideal triage path.")
        verdict = "suboptimal"
    else:
        rationale.append("The action adds some information but does not complete care routing.")
        verdict = "partial"

    return {
        "assessment": assessment,
        "verdict": verdict,
        "rationale": " ".join(rationale),
    }


def compute_reward(case, action):
    correct = case["correct_action"]
    assessment = assess_case(case)
    risk_score = assessment["risk_score"]

    if action == correct:
        return 10 + risk_score // 20

    if action in {"ASK_FOLLOWUP", "REQUEST_VITALS", "CHECK_MEDICATION_HISTORY"}:
        return 3 if correct != "ESCALATE_EMERGENCY" else 1

    if correct == "ESCALATE_EMERGENCY" and action != "ESCALATE_EMERGENCY":
        return -12

    if action in TERMINAL_ACTIONS:
        return -6

    return -2
