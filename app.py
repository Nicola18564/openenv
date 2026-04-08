from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from medienv.environment import HealthTriageEnv, load_scenarios


sys.dont_write_bytecode = True

SCENARIOS = {item["name"]: item for item in load_scenarios()}
SESSION_LOG_PATH = Path(__file__).with_name("session_logs.json")
MAX_SESSION_LOG_ENTRIES = 500


class ResetRequest(BaseModel):
    scenario_name: Optional[str] = None


class StepRequest(BaseModel):
    action: str


def pretty_state(state):
    return f"""
**Case:** {state.get('title', state.get('case_id', 'unknown'))}

**Symptoms:** {state.get('symptoms')}

**Age group:** {state.get('age_group')}

**Severity:** {state.get('severity')}

**Urgency:** {state.get('urgency')}

**Risk score:** {state.get('risk_score')}

**Rural access:** {state.get('rural_access')}

**Mental state:** {state.get('mental_state')}

**Step count:** {state.get('step_count')}

**Done:** {state.get('done')}
"""


def available_actions():
    return list(HealthTriageEnv.ACTIONS)


def save_session_log(scenario_name, state, info, reward):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario_name,
        "reward": reward,
        "state": state,
        "info": info,
    }

    if SESSION_LOG_PATH.exists():
        try:
            existing = json.loads(SESSION_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    existing.append(entry)
    if len(existing) > 1000:
        existing = existing[-MAX_SESSION_LOG_ENTRIES:]

    SESSION_LOG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


class OpenEnvSession:
    def __init__(self, scenario_name=None):
        self.scenario_name = scenario_name or next(iter(SCENARIOS))
        self.env = HealthTriageEnv(SCENARIOS[self.scenario_name])
        self._last_observation = self.env.reset()

    def reset(self, scenario_name=None):
        if scenario_name is not None:
            if scenario_name not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario_name}")
            self.scenario_name = scenario_name
            self.env = HealthTriageEnv(SCENARIOS[self.scenario_name])
        self._last_observation = self.env.reset()
        return self._last_observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._last_observation = observation
        return observation, reward, done, info

    def state(self):
        return self._last_observation

    def benchmark(self, episodes=20):
        return self.env.benchmark(episodes=episodes)


session = OpenEnvSession()


app = FastAPI(title="MediAssist Triage Arena")


@app.get("/health")
def health():
    return {"status": "ok", "scenario": session.scenario_name}


@app.get("/actions")
def get_actions():
    return {"available_actions": available_actions()}


@app.api_route("/state", methods=["GET", "POST"])
def get_state():
    return {
        "scenario": session.scenario_name,
        "observation": session.state(),
        "available_actions": available_actions(),
    }


@app.post("/reset")
def api_reset(payload: Optional[ResetRequest] = None):
    scenario_name = payload.scenario_name if payload else None
    observation = session.reset(scenario_name)
    return {
        "scenario": session.scenario_name,
        "observation": observation,
        "available_actions": available_actions(),
        "reward": 0.0,
        "done": False,
        "info": {"message": "Environment reset"},
    }


@app.post("/step")
def api_step(payload: StepRequest):
    if payload.action not in available_actions():
        raise HTTPException(status_code=400, detail=f"Unsupported action: {payload.action}")

    observation, reward, done, info = session.step(payload.action)
    save_session_log(session.scenario_name, observation, info, reward)
    return {
        "scenario": session.scenario_name,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
        "available_actions": available_actions(),
        "total_reward": observation.get("total_reward", 0.0),
    }


def ui_reset(scenario_name):
    observation = session.reset(scenario_name)
    info = {"message": "Environment reset"}
    return pretty_state(observation), 0.0, False, json.dumps(info, indent=2)


def ui_step(action):
    observation, reward, done, info = session.step(action)
    save_session_log(session.scenario_name, observation, info, reward)
    return pretty_state(observation), reward, done, json.dumps(info, indent=2)


with gr.Blocks() as demo:
    gr.Markdown("# MediAssist Triage System")
    gr.Markdown("Interactive OpenEnv-compatible triage workflow with structured feedback.")

    scenario_dropdown = gr.Dropdown(
        choices=list(SCENARIOS.keys()),
        value=session.scenario_name,
        label="Scenario",
    )
    action_dropdown = gr.Dropdown(
        choices=available_actions(),
        value=available_actions()[0],
        label="Action",
    )

    with gr.Row():
        reset_btn = gr.Button("Reset")
        step_btn = gr.Button("Step")

    state_box = gr.Markdown(value=pretty_state(session.state()))
    reward_box = gr.Number(label="Reward")
    done_box = gr.Checkbox(label="Done")
    info_box = gr.Code(label="Info", language="json")

    reset_btn.click(
        fn=ui_reset,
        inputs=[scenario_dropdown],
        outputs=[state_box, reward_box, done_box, info_box],
    )
    step_btn.click(
        fn=ui_step,
        inputs=[action_dropdown],
        outputs=[state_box, reward_box, done_box, info_box],
    )


app = gr.mount_gradio_app(app, demo, path="/")


def main():
    host = os.getenv("GRADIO_SERVER_NAME", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
