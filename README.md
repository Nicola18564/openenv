---
title: MediAssist Triage Arena
emoji: "🏥"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.10.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# MediAssist Triage Arena

MediAssist Triage Arena is an OpenEnv-style health triage simulator built for hackathon demos.

It provides a scenario-based environment where the user can choose actions such as:

- ask follow-up questions
- request vitals
- check medication history
- schedule telemedicine
- recommend self-care
- recommend clinic visit
- recommend doctor visit
- escalate to emergency care

The environment returns:

- observation/state
- reward
- done flag
- info with explanation and recommended path

## Highlights

- Equity-aware triage signals including rural access, mobility, language, and financial barriers
- Explainable scoring with urgency, risk score, verdict, and rationale
- Scenario-driven decisions for emergency, clinic, telemedicine, self-care, and doctor-visit cases
- Benchmark mode for quick quantitative evaluation
- Gradio interface for interactive demo viewing
- Scenario selector for switching between multiple patient cases
- Step history and accumulated reward tracking during each episode

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Baseline benchmark

```bash
python baseline/run_baseline.py
```

Benchmark mode reports:

- average reward
- successful triage rate
- urgency breakdown across sampled cases
- count of correct final decisions

## OpenEnv entry

The environment entry point is:

```text
medienv.environment:HealthTriageEnv
```

## Submission contents

This repository includes:

- `app.py`
- `requirements.txt`
- `README.md`
- environment implementation under `medienv/`
- benchmark script under `baseline/`
- scenario definitions under `medienv/tasks.py`
- reward logic under `medienv/grader.py`

## Demo behavior

The Gradio demo is designed to:

- display the current observation clearly, including symptoms, severity, age group, and access constraints
- allow the user to select a scenario
- let the user take a `step` action or `reset` the environment
- show reward, rationale, completion status, and decision history after each action

## How Scoring Works

The reward system is designed to encourage safe and appropriate triage decisions.

| Action | Good use case | Reward behavior |
| --- | --- | --- |
| `ASK_FOLLOWUP` | when more clarification is needed | small positive reward |
| `REQUEST_VITALS` | when risk needs more evidence | small positive reward |
| `CHECK_MEDICATION_HISTORY` | when treatment history matters | small positive reward |
| `RECOMMEND_SELF_CARE` | mild low-risk cases | positive reward if appropriate, negative if unsafe |
| `RECOMMEND_DOCTOR_VISIT` | moderate or high-risk cases | positive reward if appropriate |
| `RECOMMEND_CLINIC` | access-sensitive or higher-risk cases | positive reward if appropriate |
| `SCHEDULE_TELEMEDICINE` | moderate cases suitable for remote follow-up | positive reward if appropriate |
| `ESCALATE_EMERGENCY` | severe or critical cases | highest reward when required, strong penalty if missed |

In general:

- correct final actions receive the strongest reward
- information-gathering actions receive a small reward
- unsafe decisions, especially failing to escalate emergencies, receive strong penalties

## Notes

- The environment is designed to run offline.
- No external APIs or cloud databases are required.
- The demo runs from the repository and the deployed Hugging Face Space.
- The OpenEnv entry must exactly match the class name in code: `medienv.environment:HealthTriageEnv`
- `app.py` builds the Gradio UI, connects the `step` and `reset` controls, and launches the demo
