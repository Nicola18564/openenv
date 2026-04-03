import gradio as gr
import json

from medienv.environment import HealthTriageEnv
from medienv.tasks import ACTION_CATALOG, SCENARIOS


env = HealthTriageEnv(seed=7)
state = env.reset(case_id=SCENARIOS[0]["id"])


CSS = """
:root {
  --bg: #f3efe7;
  --paper: rgba(255, 253, 249, 0.92);
  --ink: #152536;
  --muted: #5f6f7f;
  --primary: #0b6e69;
  --accent: #c67a12;
  --danger: #b42318;
  --border: rgba(11, 110, 105, 0.14);
  --shadow: 0 20px 70px rgba(21, 37, 54, 0.12);
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(198, 122, 18, 0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(11, 110, 105, 0.17), transparent 26%),
    linear-gradient(135deg, #f7f2ea 0%, #eef7f6 100%);
  color: var(--ink);
  font-family: "Trebuchet MS", "Segoe UI", sans-serif;
}

.hero-card, .panel-card, .metric-card {
  background: var(--paper);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow: var(--shadow);
}

.hero-card {
  padding: 24px 28px;
  margin-bottom: 18px;
}

.hero-title {
  font-size: 2.35rem;
  line-height: 1.05;
  margin: 0;
  color: #16324a;
}

.hero-subtitle {
  margin-top: 10px;
  color: var(--muted);
  font-size: 1.02rem;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.metric-card {
  padding: 14px 16px;
}

.metric-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.metric-value {
  font-size: 1.7rem;
  font-weight: 700;
  margin-top: 8px;
  color: #0b3b36;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.badge {
  background: rgba(15, 118, 110, 0.1);
  border: 1px solid rgba(15, 118, 110, 0.16);
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 0.85rem;
}

.status-strip {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 14px;
}

.status-pill {
  border-radius: 999px;
  padding: 8px 14px;
  font-size: 0.84rem;
  border: 1px solid rgba(11, 110, 105, 0.14);
  background: rgba(11, 110, 105, 0.08);
}

.quality-banner {
  border-radius: 20px;
  padding: 16px 18px;
  border: 1px solid rgba(21, 37, 54, 0.08);
  margin-bottom: 14px;
}

.quality-title {
  font-size: 1.15rem;
  font-weight: 700;
  margin: 0 0 8px 0;
}

.danger {
  color: var(--danger);
  font-weight: 700;
}

@media (max-width: 900px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
"""


def _history_markdown(history):
    if not history:
        return "No actions recorded yet."
    return "\n".join(f"{idx}. `{item}`" for idx, item in enumerate(history, start=1))


def _scenario_choices():
    return [f"{case['id']} | {case['title']}" for case in SCENARIOS]


def _extract_case_id(choice):
    return choice.split("|", 1)[0].strip()


def _build_metrics_html(current_state):
    urgency_color = {
        "critical": "#b91c1c",
        "high": "#c2410c",
        "moderate": "#0f766e",
        "low": "#2563eb",
    }.get(current_state.get("urgency"), "#334155")
    return f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Urgency</div>
        <div class="metric-value" style="color:{urgency_color}">{current_state.get("urgency", "-").title()}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Risk Score</div>
        <div class="metric-value">{current_state.get("risk_score", 0)}/100</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Episode Reward</div>
        <div class="metric-value">{current_state.get("total_reward", 0)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Actions Taken</div>
        <div class="metric-value">{len(current_state.get("history", []))}</div>
      </div>
    </div>
    """


def _build_case_brief_html(current_state):
    symptoms = ", ".join(current_state.get("symptoms", []))
    red_flags = current_state.get("red_flags", [])
    chronic_conditions = current_state.get("chronic_conditions", [])

    red_flag_html = "".join(
        f'<span class="badge danger">{flag.replace("_", " ").title()}</span>' for flag in red_flags
    ) or '<span class="badge">No critical red flags</span>'

    chronic_html = "".join(
        f'<span class="badge">{item.replace("_", " ").title()}</span>' for item in chronic_conditions
    ) or '<span class="badge">No chronic condition reported</span>'

    return f"""
    <div class="panel-card" style="padding:18px 20px;">
      <h3 style="margin:0; color:#102a43;">{current_state.get("title", "Case Brief")}</h3>
      <p style="margin:8px 0 0 0; color:#475569;">{current_state.get("summary", "")}</p>
      <div class="badge-row" style="margin-top:14px;">
        <span class="badge">Age: {current_state.get("age_group", "-").title()}</span>
        <span class="badge">Severity: {current_state.get("severity", "-").title()}</span>
        <span class="badge">Duration: {current_state.get("duration_days", "-")} day(s)</span>
        <span class="badge">Symptoms: {symptoms}</span>
      </div>
      <div class="badge-row">{red_flag_html}</div>
      <div class="badge-row">{chronic_html}</div>
    </div>
    """


def _build_equity_html(current_state):
    chips = []
    if current_state.get("rural_access"):
        chips.append("Rural access barrier")
    if current_state.get("mobility_issues"):
        chips.append("Mobility constraints")
    if current_state.get("language_barrier"):
        chips.append("Language support needed")
    if current_state.get("insurance_risk"):
        chips.append("Financial follow-up risk")
    if not chips:
        chips.append("No major access barriers flagged")

    chip_html = "".join(f'<span class="badge">{chip}</span>' for chip in chips)
    return f"""
    <div class="panel-card" style="padding:18px 20px;">
      <h3 style="margin:0 0 10px 0; color:#102a43;">Care Equity Lens</h3>
      <p style="margin:0; color:#475569;">
        The simulator highlights social and access barriers so judges can see this project is not only clinically aware,
        but also designed for real-world care delivery.
      </p>
      <div class="badge-row">{chip_html}</div>
    </div>
    """


def _build_explanation_html(info):
    explanation = info.get("explanation", {})
    assessment = explanation.get("assessment", {})
    path = info.get("recommended_path", [])
    path_text = " -> ".join(path) if path else "Not available"

    return f"""
    <div class="panel-card" style="padding:18px 20px;">
      <h3 style="margin:0 0 10px 0; color:#102a43;">Decision Intelligence</h3>
      <p style="margin:0 0 8px 0;"><strong>Expert action:</strong> {info.get("expert_action", "-")}</p>
      <p style="margin:0 0 8px 0;"><strong>Recommended path:</strong> {path_text}</p>
      <p style="margin:0 0 8px 0;"><strong>Verdict:</strong> {explanation.get("verdict", "-").title()}</p>
      <p style="margin:0 0 8px 0;"><strong>Rationale:</strong> {explanation.get("rationale", "No explanation yet.")}</p>
      <p style="margin:0;"><strong>Signals:</strong> risk={assessment.get("risk_score", "-")}, urgency={assessment.get("urgency", "-")}, access={assessment.get("access_risk", "-")}, social={assessment.get("social_risk", "-")}</p>
    </div>
    """


def _build_benchmark_html():
    benchmark_env = HealthTriageEnv(seed=21)
    report = benchmark_env.benchmark(episodes=30)
    breakdown = report["urgency_breakdown"]
    return f"""
    <div class="panel-card" style="padding:18px 20px;">
      <h3 style="margin:0 0 10px 0; color:#102a43;">Benchmark Snapshot</h3>
      <p style="margin:0 0 8px 0;"><strong>Episodes:</strong> {report['episodes']}</p>
      <p style="margin:0 0 8px 0;"><strong>Average reward:</strong> {report['average_reward']}</p>
      <p style="margin:0 0 8px 0;"><strong>Successful triage rate:</strong> {report['successful_triage_rate']}%</p>
      <p style="margin:0;"><strong>Urgency mix:</strong> critical {breakdown['critical']}, high {breakdown['high']}, moderate {breakdown['moderate']}, low {breakdown['low']}</p>
    </div>
    """


def _reward_detail(last_reward, current_state):
    status = "Episode Complete" if current_state.get("done") else "Active"
    total_reward = current_state.get("total_reward", 0)

    if last_reward is None:
        impact = "Awaiting first action"
    elif last_reward >= 12:
        impact = "Critical-safe decision"
    elif last_reward >= 10:
        impact = "Correct final disposition"
    elif last_reward >= 3:
        impact = "Useful information gathering"
    elif last_reward >= 0:
        impact = "Low-impact safe step"
    elif last_reward <= -10:
        impact = "Unsafe under-triage"
    else:
        impact = "Incorrect final recommendation"

    return f"Step Reward: {last_reward if last_reward is not None else 0} | Total Reward: {total_reward} | Assessment: {impact} | Status: {status}"


def _reward_breakdown_html(last_reward, current_state, info=None):
    info = info or {}
    explanation = info.get("explanation", {})
    verdict = explanation.get("verdict", "ready").title()
    score = current_state.get("risk_score", 0)
    urgency = current_state.get("urgency", "-").title()
    step_reward = 0 if last_reward is None else last_reward

    return f"""
    <div class="panel-card" style="padding:18px 20px;">
      <h3 style="margin:0 0 10px 0; color:#16324a;">Reward Analysis</h3>
      <div class="status-strip">
        <span class="status-pill">Step Reward: {step_reward}</span>
        <span class="status-pill">Total Reward: {current_state.get("total_reward", 0)}</span>
        <span class="status-pill">Risk Score: {score}</span>
        <span class="status-pill">Urgency: {urgency}</span>
        <span class="status-pill">Verdict: {verdict}</span>
      </div>
      <p style="margin:14px 0 0 0; color:#475569;">
        Score meaning: information gathering earns a small positive score, correct non-emergency routing earns a medium score,
        correct emergency escalation earns the highest positive score, and unsafe under-triage receives a strong penalty.
      </p>
    </div>
    """


def _quality_label(info, current_state):
    verdict = info.get("explanation", {}).get("verdict", "ready")
    done = current_state.get("done", False)

    mapping = {
        "ready": ("System Ready", "#eff6ff", "#1d4ed8"),
        "optimal": ("High-Confidence Triage", "#ecfdf3", "#027a48"),
        "reasonable": ("Safe Investigative Step", "#fffaeb", "#b54708"),
        "partial": ("Partial Evidence", "#fff7ed", "#c2410c"),
        "suboptimal": ("Needs Better Routing", "#fef3f2", "#b42318"),
        "unsafe": ("Unsafe Under-Triage", "#fef2f2", "#b42318"),
    }
    label, bg, ink = mapping.get(verdict, ("System Ready", "#eff6ff", "#1d4ed8"))
    if done and verdict == "optimal":
        label = "Episode Closed Safely"
    return label, bg, ink


def _quality_banner_html(info, current_state):
    label, bg, ink = _quality_label(info, current_state)
    rationale = info.get("explanation", {}).get(
        "rationale",
        "Select an action to generate a triage quality assessment."
    )
    return f"""
    <div class="quality-banner" style="background:{bg}; color:{ink};">
      <p class="quality-title">{label}</p>
      <p style="margin:0;">{rationale}</p>
    </div>
    """


def _episode_log_text(current_state, info=None):
    info = info or {}
    payload = {
        "case_id": current_state.get("case_id"),
        "title": current_state.get("title"),
        "urgency": current_state.get("urgency"),
        "risk_score": current_state.get("risk_score"),
        "total_reward": current_state.get("total_reward"),
        "done": current_state.get("done"),
        "history": current_state.get("history", []),
        "expert_action": info.get("expert_action"),
        "recommended_path": info.get("recommended_path", []),
        "explanation": info.get("explanation", {}),
    }
    return json.dumps(payload, indent=2)


def export_episode_log():
    info = {
        "expert_action": env.expert_policy(state),
        "recommended_path": [],
        "explanation": env.last_explanation or {
            "verdict": "ready",
            "rationale": "No completed action has been recorded yet.",
        },
    }
    return _episode_log_text(state, info)


def run_benchmark():
    benchmark_env = HealthTriageEnv(seed=21)
    report = benchmark_env.benchmark(episodes=50)
    breakdown = report["urgency_breakdown"]
    summary = (
        f"Episodes: {report['episodes']} | "
        f"Average Reward: {report['average_reward']} | "
        f"Success Rate: {report['successful_triage_rate']}% | "
        f"Urgency Mix: critical {breakdown['critical']}, high {breakdown['high']}, "
        f"moderate {breakdown['moderate']}, low {breakdown['low']}"
    )
    return _build_benchmark_html(), summary


def _render(current_state, message, info=None):
    info = info or {}
    reward_text = _reward_detail(info.get("step_reward"), current_state)
    return (
        current_state,
        message,
        reward_text,
        _quality_banner_html(info, current_state),
        _reward_breakdown_html(info.get("step_reward"), current_state, info),
        _episode_log_text(current_state, info),
        _build_metrics_html(current_state),
        _build_case_brief_html(current_state),
        _build_equity_html(current_state),
        _build_explanation_html(info),
        _history_markdown(current_state.get("history", [])),
        _build_benchmark_html(),
    )


def reset_env(case_choice):
    global state
    case_id = _extract_case_id(case_choice) if case_choice else None
    state = env.reset(case_id=case_id)
    message = f"Loaded case {state['case_id']}. Review the case summary and choose the next triage action."
    info = {
        "step_reward": None,
        "recommended_path": [],
        "expert_action": env.expert_policy(state),
        "explanation": {
            "verdict": "ready",
            "rationale": "Case initialized. Use the action selector to gather information or issue a care decision.",
            "assessment": {
                "risk_score": state["risk_score"],
                "urgency": state["urgency"],
                "access_risk": state["access_risk"],
                "social_risk": state["social_risk"],
            },
        },
    }
    return _render(state, message, info)


def step_env(action):
    global state
    if not action:
        return _render(state, "Choose an action before stepping the environment.")

    try:
        state, reward, done, info = env.step(action)
        info["step_reward"] = reward
        if done:
            message = f"Action recorded. Step reward {reward}. Episode closed with total score {state['total_reward']}."
        else:
            message = f"Action recorded. Step reward {reward}. Continue triage or finalize the care route."
        return _render(state, message, info)
    except Exception as exc:
        return _render(state, f"Error: {exc}")


with gr.Blocks(title="MediAssist Triage Arena") as demo:
    gr.HTML(
        """
        <div class="hero-card">
          <p style="margin:0; letter-spacing:0.18em; text-transform:uppercase; color:#c67a12; font-weight:700;">AI-Assisted Clinical Triage</p>
          <h1 class="hero-title">MediAssist Clinical Triage System</h1>
          <p class="hero-subtitle">
            A structured health-triage system with explainable scoring, safe decision routing,
            and equity-aware patient assessment.
          </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=7):
            metrics_html = gr.HTML(value=_build_metrics_html(state))
        with gr.Column(scale=5):
            benchmark_html = gr.HTML(value=_build_benchmark_html())

    with gr.Row():
        with gr.Column(scale=7):
            case_brief_html = gr.HTML(value=_build_case_brief_html(state))
        with gr.Column(scale=5):
            equity_html = gr.HTML(value=_build_equity_html(state))

    with gr.Row():
        with gr.Column(scale=4):
            case_selector = gr.Dropdown(
                choices=_scenario_choices(),
                value=_scenario_choices()[0],
                label="Clinical Case",
                info="Select a patient scenario.",
            )
            action_input = gr.Dropdown(
                choices=ACTION_CATALOG,
                value="ASK_FOLLOWUP",
                label="Triage Action",
                info="Choose the next structured clinical action.",
            )
            with gr.Row():
                step_btn = gr.Button("Step", variant="primary")
                reset_btn = gr.Button("Reset Case")
                benchmark_btn = gr.Button("Run Benchmark")
            message_output = gr.Textbox(label="System Output", interactive=False, value="System ready.")
            reward_output = gr.Textbox(label="Reward Summary", interactive=False, value=_reward_detail(None, state))
            log_output = gr.Code(label="Episode Log", value=_episode_log_text(state, {}), language="json")
            export_btn = gr.Button("Export Episode Log")
            benchmark_output = gr.Textbox(label="Benchmark Output", interactive=False, value="Benchmark ready.")
            history_output = gr.Markdown(value=_history_markdown(state.get("history", [])), label="Episode Timeline")
        with gr.Column(scale=8):
            quality_banner = gr.HTML(value=_quality_banner_html({}, state))
            reward_panel = gr.HTML(value=_reward_breakdown_html(None, state, {}))
            explanation_html = gr.HTML(
                value=_build_explanation_html(
                    {
                        "recommended_path": [],
                        "expert_action": env.expert_policy(state),
                        "explanation": {
                            "verdict": "ready",
                            "rationale": "Start the simulation to see reasoning, expert policy, and care path guidance.",
                            "assessment": {
                                "risk_score": state["risk_score"],
                                "urgency": state["urgency"],
                                "access_risk": state["access_risk"],
                                "social_risk": state["social_risk"],
                            },
                        },
                    }
                )
            )
            state_output = gr.JSON(label="OpenEnv State", value=state)

    step_btn.click(
        fn=step_env,
        inputs=action_input,
        outputs=[
            state_output,
            message_output,
            reward_output,
            quality_banner,
            reward_panel,
            log_output,
            metrics_html,
            case_brief_html,
            equity_html,
            explanation_html,
            history_output,
            benchmark_html,
        ],
    )

    reset_btn.click(
        fn=reset_env,
        inputs=case_selector,
        outputs=[
            state_output,
            message_output,
            reward_output,
            quality_banner,
            reward_panel,
            log_output,
            metrics_html,
            case_brief_html,
            equity_html,
            explanation_html,
            history_output,
            benchmark_html,
        ],
    )

    benchmark_btn.click(
        fn=run_benchmark,
        outputs=[benchmark_html, benchmark_output],
    )

    export_btn.click(
        fn=export_episode_log,
        outputs=log_output,
    )


if __name__ == "__main__":
    demo.launch(css=CSS)
