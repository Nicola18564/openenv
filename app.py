import gradio as gr

from medienv.environment import HealthTriageEnv
from medienv.tasks import ACTION_CATALOG, SCENARIOS


env = HealthTriageEnv(seed=7)
state = env.reset(case_id=SCENARIOS[0]["id"])


CSS = """
:root {
  --bg: #f6efe4;
  --paper: rgba(255, 252, 247, 0.88);
  --ink: #1f2937;
  --muted: #6b7280;
  --primary: #0f766e;
  --accent: #d97706;
  --danger: #b91c1c;
  --border: rgba(15, 118, 110, 0.16);
  --shadow: 0 18px 60px rgba(31, 41, 55, 0.12);
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(217, 119, 6, 0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(15, 118, 110, 0.18), transparent 26%),
    linear-gradient(135deg, #f7f1e7 0%, #eef6f5 100%);
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
  font-size: 2.2rem;
  line-height: 1.05;
  margin: 0;
  color: #102a43;
}

.hero-subtitle {
  margin-top: 10px;
  color: var(--muted);
  font-size: 1rem;
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
        return "No actions yet. Start by collecting signals or making a triage decision."
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


def _render(current_state, message, info=None):
    info = info or {}
    reward_text = f"Total Reward: {current_state.get('total_reward', 0)}"
    if current_state.get("done"):
        reward_text += " | Episode Complete"
    return (
        current_state,
        message,
        reward_text,
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
    message = f"Loaded case {state['case_id']}. Judges can now test the triage agent."
    info = {
        "recommended_path": [],
        "expert_action": env.expert_policy(state),
        "explanation": {
            "verdict": "ready",
            "rationale": "Environment reset. Use guided actions to evaluate triage quality.",
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
        if done:
            message = f"Action accepted. Reward {reward}. Episode complete with total score {state['total_reward']}."
        else:
            message = f"Action accepted. Reward {reward}. Continue investigating or route care."
        return _render(state, message, info)
    except Exception as exc:
        return _render(state, f"Error: {exc}")


with gr.Blocks(title="MediAssist Triage Arena") as demo:
    gr.HTML(
        """
        <div class="hero-card">
          <p style="margin:0; letter-spacing:0.18em; text-transform:uppercase; color:#d97706; font-weight:700;">OpenEnv Hackathon Demo</p>
          <h1 class="hero-title">MediAssist Triage Arena</h1>
          <p class="hero-subtitle">
            A clinically aware, equity-sensitive health triage environment with explainable scoring,
            benchmark mode, and a polished judge-ready simulation interface.
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
                label="Demo Case",
                info="Choose a showcase scenario for judges.",
            )
            action_input = gr.Dropdown(
                choices=ACTION_CATALOG,
                value="ASK_FOLLOWUP",
                label="Triage Action",
                info="Use structured actions to drive the environment.",
            )
            with gr.Row():
                step_btn = gr.Button("Run Triage Step", variant="primary")
                reset_btn = gr.Button("Load Case")
            message_output = gr.Textbox(label="System Output", interactive=False, value="Ready for the demo.")
            reward_output = gr.Textbox(label="Reward System", interactive=False, value=f"Total Reward: {state['total_reward']}")
            history_output = gr.Markdown(value=_history_markdown(state.get("history", [])), label="Episode Timeline")
        with gr.Column(scale=8):
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
            metrics_html,
            case_brief_html,
            equity_html,
            explanation_html,
            history_output,
            benchmark_html,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS, inbrowser=True)
