import json
import logging
import os
import socket
import sys
from pathlib import Path
from threading import Lock
from typing import Optional

sys.dont_write_bytecode = True

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from medienv.environment import HealthTriageEnv, load_scenarios


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
SESSION_LOG_PATH = Path("session_logs.json")

SCENARIOS = {item["name"]: item for item in load_scenarios()}

ACTION_LABELS = {
    "ASK_FOLLOWUP": "Ask follow-up question",
    "PROVIDE_SUPPORT_MESSAGE": "Provide support message",
    "RECOMMEND_SELF_CARE": "Recommend self-care",
    "RECOMMEND_CLINIC": "Recommend clinic visit",
    "RECOMMEND_DOCTOR_VISIT": "Recommend doctor visit",
    "ESCALATE_EMERGENCY": "Escalate to emergency care",
}

DEFAULT_SCENARIO_NAME = "Mild headache"


def available_actions():
    return list(ACTION_LABELS.keys())


def suggestion_text(env):
    if env is None:
        return "Reset the environment to get an action suggestion."
    if env.done:
        return "Episode complete. Reset the case to start again."
    return f"Suggested next action: {env.expert_policy()}"


def _default_metrics():
    return {
        "episodes_started": 0,
        "episodes_completed": 0,
        "successful_episodes": 0,
        "unsafe_episodes": 0,
        "total_reward": 0.0,
        "last_reward": 0.0,
    }


def metrics_text(metrics):
    average_reward = 0.0
    success_rate = 0.0
    unsafe_rate = 0.0
    if metrics["episodes_completed"]:
        average_reward = metrics["total_reward"] / metrics["episodes_completed"]
        success_rate = (metrics["successful_episodes"] / metrics["episodes_completed"]) * 100
        unsafe_rate = (metrics["unsafe_episodes"] / metrics["episodes_completed"]) * 100
    return (
        f"Episodes started: {metrics['episodes_started']}\n"
        f"Episodes completed: {metrics['episodes_completed']}\n"
        f"Safe final decisions: {metrics['successful_episodes']}\n"
        f"Unsafe final decisions: {metrics['unsafe_episodes']}\n"
        f"Cumulative reward: {metrics['total_reward']:.1f}\n"
        f"Average reward: {average_reward:.2f}\n"
        f"Safe final rate: {success_rate:.1f}%\n"
        f"Unsafe final rate: {unsafe_rate:.1f}%\n"
        f"Last reward: {metrics['last_reward']:.1f}"
    )


def save_session_log(scenario_name, state, info, reward):
    entry = {
        "scenario": scenario_name,
        "reward": reward,
        "done": state["done"],
        "step_count": state["step_count"],
        "total_reward": state["total_reward"],
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


def _history_lines(history):
    if not history:
        return "- No actions taken yet."
    return "\n".join(
        f"- Step {item['step']}: `{item['action']}` ({item['reward']:+.1f}, {item['alignment']})"
        for item in history
    )


def pretty_state(state):
    if not state:
        return "### No case loaded"

    chronic_conditions = ", ".join(state.get("chronic_conditions", [])) or "None"
    red_flags = ", ".join(state.get("red_flags", [])) or "None"
    return f"""### Current Case
**Scenario:** {state.get("name", "Unknown")}

**Summary:** {state.get("summary", "No summary available.")}

- Symptoms: `{state.get("symptoms", "No symptoms recorded")}`
- Age group: **{state.get("age_group", "unknown").title()}**
- Severity: **{state.get("severity", "unknown").title()}**
- Urgency: **{state.get("urgency", "unknown").title()}**
- Risk score: **{state.get("risk_score", 0)} / 100**
- Rural access barrier: **{"Yes" if state.get("rural_access") else "No"}**
- Mental state: **{state.get("mental_state", "unknown")}**
- Fall risk flag: **{"Yes" if state.get("fall_flag") else "No"}**
- Epidemic flag: **{"Yes" if state.get("epidemic_flag") else "No"}**
- Chronic conditions: **{chronic_conditions}**
- Red flags: **{red_flags}**

### Episode Progress
- Steps used: **{state.get("step_count", 0)} / 4**
- Total reward: **{state.get("total_reward", 0.0):+.1f}**
- Context collected: **{"Yes" if state.get("context_collected") else "No"}**
- Support provided: **{"Yes" if state.get("support_provided") else "No"}**
- Done: **{"Yes" if state.get("done") else "No"}**

### History
{_history_lines(state.get("history", []))}
"""


def reward_panel(info=None, reward=0.0, state=None):
    if not isinstance(info, dict):
        urgency = (state or {}).get("urgency", "unknown")
        risk_score = (state or {}).get("risk_score", 0)
        return f"""### Reward Analysis
- Current risk: **{risk_score} / 100**
- Urgency: **{str(urgency).title()}**
- The reward model scores **safety**, **sequence fit**, **access fit**, **empathy**, and **efficiency**.
- Unsafe under-triage gets the strongest penalty.
"""

    reward_breakdown = info.get("reward_breakdown", {})
    care_plan = " -> ".join(info.get("care_plan", [])) or "Not available"
    next_action = info.get("suggested_next_action") or "Episode complete"
    return f"""### Reward Analysis
- Step reward: **{reward:+.1f}**
- Verdict: **{info.get("verdict", "unknown").replace("_", " ").title()}**
- Resolution: **{info.get("resolution_quality", "unknown").replace("_", " ").title()}**
- Safety: **{reward_breakdown.get("safety", 0.0):+.1f}**
- Sequence fit: **{reward_breakdown.get("sequence", 0.0):+.1f}**
- Access fit: **{reward_breakdown.get("access", 0.0):+.1f}**
- Empathy: **{reward_breakdown.get("empathy", 0.0):+.1f}**
- Efficiency: **{reward_breakdown.get("efficiency", 0.0):+.1f}**
- Care plan: **{care_plan}**
- Next best action: **{next_action}**

**Rationale:** {info.get("rationale", "No rationale available.")}
"""


def format_info_text(info):
    if not isinstance(info, dict):
        return str(info)
    care_plan = " -> ".join(info.get("care_plan", [])) or "Not available"
    return (
        f"Step: {info.get('step_count', '-')}\n"
        f"Urgency: {info.get('urgency', '-')}\n"
        f"Alignment: {info.get('action_alignment', '-')}\n"
        f"Resolution: {info.get('resolution_quality', '-')}\n"
        f"Care plan: {care_plan}\n"
        f"Rationale: {info.get('rationale', 'No rationale available.')}"
    )


class ResetRequest(BaseModel):
    scenario_name: Optional[str] = None


class StepRequest(BaseModel):
    action: str


class OpenEnvSession:
    def __init__(self):
        self._lock = Lock()
        self._env = None
        self._scenario_name = DEFAULT_SCENARIO_NAME
        self._last_observation = None

    def _reset_unlocked(self, scenario_name=None):
        selected_name = scenario_name or self._scenario_name
        if selected_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {selected_name}")
        self._scenario_name = selected_name
        self._env = HealthTriageEnv(SCENARIOS[selected_name])
        state = self._env.reset()
        self._last_observation = state
        return {
            "scenario": selected_name,
            "observation": state,
            "state": state,
            "done": state["done"],
            "available_actions": available_actions(),
        }

    def reset(self, scenario_name=None):
        with self._lock:
            return self._reset_unlocked(scenario_name)

    def state(self):
        with self._lock:
            if self._env is None:
                return self._reset_unlocked(self._scenario_name)
            state = self._last_observation or self._env.reset()
            return {
                "scenario": self._scenario_name,
                "observation": state,
                "state": state,
                "done": state["done"],
                "available_actions": available_actions(),
            }

    def step(self, action):
        with self._lock:
            if action not in available_actions():
                raise ValueError(f"Invalid action: {action}")
            if self._env is None:
                self._reset_unlocked(self._scenario_name)

            # Ensure env is initialized before stepping
            if self._env is None:
                raise RuntimeError("Failed to initialize environment")

            observation, reward, done, info = self._env.step(action)
            self._last_observation = observation
            save_session_log(self._scenario_name, observation, info, reward)
            return {
                "scenario": self._scenario_name,
                "observation": observation,
                "state": observation,
                "reward": reward,
                "done": done,
                "info": info,
                "available_actions": available_actions(),
            }


def _choose_server_port():
    configured_port = os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT")
    if configured_port:
        return int(configured_port)

    for candidate in range(7860, 7891):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    return 7860


def reset_env(scenario_name, metrics):
    metrics = metrics or _default_metrics()
    if scenario_name not in SCENARIOS:
        return (
            None,
            scenario_name,
            None,
            "Invalid scenario.",
            0.0,
            False,
            "Invalid scenario selected.",
            reward_panel(),
            "",
            metrics,
            metrics_text(metrics),
        )

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
            reward_panel(state=state),
            suggestion_text(env),
            metrics,
            metrics_text(metrics),
        )
    except Exception as exc:
        LOGGER.exception("Reset failed")
        return (
            None,
            scenario_name,
            None,
            "Reset failed.",
            0.0,
            True,
            str(exc),
            reward_panel(),
            "",
            metrics,
            metrics_text(metrics),
        )


def do_step(env, scenario_name, state, action, metrics):
    metrics = metrics or _default_metrics()
    if env is None:
        return (
            env,
            scenario_name,
            state,
            "Please reset first.",
            0.0,
            False,
            "No active environment.",
            reward_panel(state=state),
            "",
            metrics,
            metrics_text(metrics),
        )
    if action not in available_actions():
        safe_state = pretty_state(state) if state else "No state available."
        return (
            env,
            scenario_name,
            state,
            safe_state,
            -2.0,
            False,
            "Invalid action selected.",
            reward_panel(state=state),
            suggestion_text(env),
            metrics,
            metrics_text(metrics),
        )

    try:
        next_state, reward, done, info = env.step(action)
        metrics["last_reward"] = reward
        metrics["total_reward"] += reward
        if done:
            metrics["episodes_completed"] += 1
            metrics["successful_episodes"] += int(info.get("resolution_quality") == "safe_final")
            metrics["unsafe_episodes"] += int(info.get("resolution_quality") == "unsafe_final")
        save_session_log(scenario_name, next_state, info, reward)
        return (
            env,
            scenario_name,
            next_state,
            pretty_state(next_state),
            reward,
            done,
            format_info_text(info),
            reward_panel(info, reward, next_state),
            suggestion_text(env),
            metrics,
            metrics_text(metrics),
        )
    except Exception as exc:
        LOGGER.exception("Step failed")
        safe_state = pretty_state(state) if state else "No state available."
        return (
            env,
            scenario_name,
            state,
            safe_state,
            0.0,
            True,
            f"Error occurred: {exc}",
            reward_panel(state=state),
            suggestion_text(env),
            metrics,
            metrics_text(metrics),
        )


with gr.Blocks() as demo:
    gr.Markdown("# AI-Assisted Triage\n\n## MediAssist Triage System")

    env_state = gr.State(None)
    scenario_state = gr.State(DEFAULT_SCENARIO_NAME)
    data_state = gr.State(None)
    metrics_state = gr.State(_default_metrics())

    with gr.Row():
        scenario_dropdown = gr.Dropdown(
            choices=list(SCENARIOS.keys()),
            value=DEFAULT_SCENARIO_NAME,
            label="Scenario",
        )
        action_dropdown = gr.Dropdown(
            choices=available_actions(),
            value="RECOMMEND_SELF_CARE",
            label="Action",
        )

    with gr.Row():
        reset_btn = gr.Button("Reset")
        step_btn = gr.Button("Run Step")

    state_box = gr.Markdown()
    reward_box = gr.Number(label="Reward")
    done_box = gr.Checkbox(label="Done")
    info_box = gr.Textbox(label="Decision Log", lines=6)
    reward_detail_box = gr.Markdown("### Reward Analysis\nTake an action to see the advanced score breakdown.")
    suggestion_box = gr.Textbox(label="AI Suggestion", lines=2)
    metrics_box = gr.Textbox(label="Performance Metrics", lines=9, value=metrics_text(_default_metrics()))

    reset_btn.click(
        fn=reset_env,
        inputs=[scenario_dropdown, metrics_state],
        outputs=[
            env_state,
            scenario_state,
            data_state,
            state_box,
            reward_box,
            done_box,
            info_box,
            reward_detail_box,
            suggestion_box,
            metrics_state,
            metrics_box,
        ],
    )

    step_btn.click(
        fn=do_step,
        inputs=[env_state, scenario_state, data_state, action_dropdown, metrics_state],
        outputs=[
            env_state,
            scenario_state,
            data_state,
            state_box,
            reward_box,
            done_box,
            info_box,
            reward_detail_box,
            suggestion_box,
            metrics_state,
            metrics_box,
        ],
    )


openenv_session = OpenEnvSession()
api = FastAPI(title="MediAssist OpenEnv API", version="1.0.0")


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/reset")
def api_reset(payload: Optional[ResetRequest] = None):
    try:
        scenario_name = payload.scenario_name if payload else None
        return openenv_session.reset(scenario_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/step")
def api_step(payload: StepRequest):
    try:
        return openenv_session.step(payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.get("/state")
@api.post("/state")
def api_state():
    return openenv_session.state()


@api.get("/actions")
def api_actions():
    return {
        "available_actions": available_actions(),
        "action_labels": ACTION_LABELS,
    }


app = gr.mount_gradio_app(api, demo, path="/")


def main():
    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        port=_choose_server_port(),
    )


if __name__ == "__main__":
    main()
