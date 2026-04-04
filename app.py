import json
import logging
import os
from pathlib import Path

import gradio as gr

from medienv.environment import HealthTriageEnv, load_scenarios


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
SESSION_LOG_PATH = Path("session_logs.json")

SCENARIOS = {item["name"]: item for item in load_scenarios()}

ACTION_LABELS = {
    "ASK_FOLLOWUP": "Ask follow-up question",
    "ESCALATE_EMERGENCY": "Escalate to emergency care",
    "RECOMMEND_CLINIC": "Recommend clinic visit",
    "RECOMMEND_DOCTOR_VISIT": "Recommend doctor visit",
    "RECOMMEND_SELF_CARE": "Recommend self-care",
    "PROVIDE_SUPPORT_MESSAGE": "Provide support message",
}


def suggestion_text(env):
    if env is None:
        return "Reset the environment to get an action suggestion."
    return f"Suggested next action: {env.expert_policy()}"


def reward_explanation(reward):
    if reward >= 3:
        return "Strong positive reward for a safe high-priority decision."
    if reward >= 2:
        return "Positive reward for an appropriate disposition."
    if reward > 0:
        return "Small positive reward for useful information gathering."
    if reward == 0:
        return "Neutral reward."
    return "Negative reward because the action was weak or unsafe for the case."


def _default_metrics():
    return {
        "episodes_started": 0,
        "episodes_completed": 0,
        "successful_episodes": 0,
        "total_reward": 0.0,
        "last_reward": 0.0,
    }


def metrics_text(metrics):
    average_reward = 0.0
    if metrics["episodes_completed"]:
        average_reward = metrics["total_reward"] / metrics["episodes_completed"]
    success_rate = 0.0
    if metrics["episodes_completed"]:
        success_rate = (metrics["successful_episodes"] / metrics["episodes_completed"]) * 100
    return (
        f"Episodes started: {metrics['episodes_started']}\n"
        f"Episodes completed: {metrics['episodes_completed']}\n"
        f"Successful episodes: {metrics['successful_episodes']}\n"
        f"Cumulative reward: {metrics['total_reward']:.1f}\n"
        f"Average reward: {average_reward:.2f}\n"
        f"Success rate: {success_rate:.1f}%\n"
        f"Last reward: {metrics['last_reward']:.1f}"
    )


def save_session_log(scenario_name, state, info, reward):
    entry = {
        "scenario": scenario_name,
        "reward": reward,
        "done": state["done"],
        "step_count": state["step_count"],
        "state": state,
        "info": info,
    }
    existing = []
    if SESSION_LOG_PATH.exists():
        try:
            existing = json.loads(SESSION_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing.append(entry)
    if len(existing) > 1000:
        existing = existing[-500:]
    SESSION_LOG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")

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

def reset_env(scenario_name, metrics):
    metrics = metrics or _default_metrics()
    if scenario_name not in SCENARIOS:
        return None, scenario_name, None, "Invalid scenario.", 0.0, False, "Invalid scenario selected.", "", metrics, metrics_text(metrics)

    try:
        env = HealthTriageEnv(SCENARIOS[scenario_name])
        state = env.reset()
        metrics["episodes_started"] += 1
        LOGGER.info("Scenario reset: %s", scenario_name)
        return (
            env,
            scenario_name,
            state,
            pretty_state(state),
            0.0,
            False,
            "Environment reset.",
            suggestion_text(env),
            metrics,
            metrics_text(metrics),
        )
    except Exception as exc:
        LOGGER.exception("Reset failed")
        return None, scenario_name, None, "Reset failed.", 0.0, True, str(exc), "", metrics, metrics_text(metrics)

def do_step(env, scenario_name, state, action, metrics):
    metrics = metrics or _default_metrics()
    if env is None:
        return env, scenario_name, state, "Please reset first.", 0.0, False, "No active environment.", "", metrics, metrics_text(metrics)
    if action not in ACTION_LABELS:
        safe_state = pretty_state(state) if state else "No state available."
        return env, scenario_name, state, safe_state, -1.0, False, "Invalid action selected.", suggestion_text(env), metrics, metrics_text(metrics)

    try:
        next_state, reward, done, info = env.step(action)
        metrics["last_reward"] = reward
        metrics["total_reward"] += reward
        if done:
            metrics["episodes_completed"] += 1
            metrics["successful_episodes"] += int(action == env.expert_policy(next_state))
        save_session_log(scenario_name, next_state, info, reward)
        info_text = f"{info}\n\nReward meaning: {reward_explanation(reward)}"
        return env, scenario_name, next_state, pretty_state(next_state), reward, done, info_text, suggestion_text(env), metrics, metrics_text(metrics)
    except Exception as exc:
        LOGGER.exception("Step failed")
        safe_state = pretty_state(state) if state else "No state available."
        return env, scenario_name, state, safe_state, 0.0, True, f"Error occurred: {exc}", suggestion_text(env), metrics, metrics_text(metrics)

with gr.Blocks() as demo:
    gr.Markdown("# AI-Assisted Triage\n\n## MediAssist Triage System")

    env_state = gr.State(None)
    scenario_state = gr.State("Mild headache")
    data_state = gr.State(None)
    metrics_state = gr.State(_default_metrics())

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
    suggestion_box = gr.Textbox(label="AI Suggestion", lines=2)
    metrics_box = gr.Textbox(label="Performance Metrics", lines=7, value=metrics_text(_default_metrics()))

    reset_btn.click(
        fn=reset_env,
        inputs=[scenario_dropdown, metrics_state],
        outputs=[env_state, scenario_state, data_state, state_box, reward_box, done_box, info_box, suggestion_box, metrics_state, metrics_box],
    )

    step_btn.click(
        fn=do_step,
        inputs=[env_state, scenario_state, data_state, action_dropdown, metrics_state],
        outputs=[env_state, scenario_state, data_state, state_box, reward_box, done_box, info_box, suggestion_box, metrics_state, metrics_box],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
