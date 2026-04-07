import gradio as gr
from medienv.environment import HealthTriageEnvironment

SCENARIOS = {
    "Mild headache": {
        "symptoms": "mild headache and fatigue",
        "age_group": "adult",
        "severity": "low",
        "rural_access": False,
        "mental_state": "neutral",
        "fall_flag": False,
        "epidemic_flag": False,
    },
    "Fever and cough": {
        "symptoms": "fever and cough for two days",
        "age_group": "adult",
        "severity": "medium",
        "rural_access": True,
        "mental_state": "neutral",
        "fall_flag": False,
        "epidemic_flag": True,
    },
    "Fall emergency": {
        "symptoms": "fell down and cannot stand up",
        "age_group": "elderly",
        "severity": "high",
        "rural_access": False,
        "mental_state": "distressed",
        "fall_flag": True,
        "epidemic_flag": False,
    },
}

ACTION_LABELS = {
    "ASK_FOLLOWUP": "Ask follow-up question",
    "ESCALATE_EMERGENCY": "Escalate to emergency care",
    "RECOMMEND_CLINIC": "Recommend clinic visit",
    "RECOMMEND_DOCTOR_VISIT": "Recommend doctor visit",
    "RECOMMEND_SELF_CARE": "Recommend self-care",
    "PROVIDE_SUPPORT_MESSAGE": "Provide support message",
}

def pretty_state(state):
    return f"""
*Symptoms:* {state['symptoms']}
*Age group:* {state['age_group']}
*Severity:* {state['severity']}
*Rural access:* {state['rural_access']}
*Mental state:* {state['mental_state']}
*Fall flag:* {state['fall_flag']}
*Epidemic flag:* {state['epidemic_flag']}
*Step count:* {state['step_count']}
*Done:* {state['done']}
"""

def reset_env(scenario_name):
    env = HealthTriageEnvironment(SCENARIOS[scenario_name])
    state = env.reset()
    return env, scenario_name, state, pretty_state(state), 0.0, False, "Environment reset."

def do_step(env, scenario_name, state, action):
    if env is None:
        return env, scenario_name, state, "Please reset first.", 0.0, False, "No active environment."

    next_state, reward, done, info = env.step(action)
    return env, scenario_name, next_state, pretty_state(next_state), reward, done, str(info)

with gr.Blocks() as demo:
    gr.Markdown("# Health Triage OpenEnv Demo")

    env_state = gr.State(None)
    scenario_state = gr.State("Mild headache")
    data_state = gr.State(None)

    with gr.Row():
        scenario_dropdown = gr.Dropdown(
            choices=list(SCENARIOS.keys()),
            value="Mild headache",
            label="Scenario",
        )
        action_dropdown = gr.Dropdown(
            choices=list(ACTION_LABELS.keys()),
            value="RECOMMEND_SELF_CARE",
            label="Action",
        )

    with gr.Row():
        reset_btn = gr.Button("Reset")
        step_btn = gr.Button("Run Step")

    state_box = gr.Markdown()
    reward_box = gr.Number(label="Reward")
    done_box = gr.Checkbox(label="Done")
    info_box = gr.Textbox(label="Info", lines=4)

    reset_btn.click(
        fn=reset_env,
        inputs=[scenario_dropdown],
        outputs=[env_state, scenario_state, data_state, state_box, reward_box, done_box, info_box],
    )

    step_btn.click(
        fn=do_step,
        inputs=[env_state, scenario_state, data_state, action_dropdown],
        outputs=[env_state, scenario_state, data_state, state_box, reward_box, done_box, info_box],
    )

demo.launch()
