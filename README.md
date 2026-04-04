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

MediAssist Triage Arena is an OpenEnv-style health triage system for structured patient assessment and care routing.

It provides a scenario-based environment where the user can choose actions such as:

- ask follow-up questions
- provide support message
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
- Scenario-driven decisions for emergency, clinic, self-care, support, and doctor-visit cases
- Benchmark mode for quick quantitative evaluation
- Gradio interface for interactive triage demo
- Scenario selector for switching between multiple patient cases
- Step history and accumulated reward tracking during each episode
- AI action suggestion based on expert policy
- Local JSON session logging for offline review
- Performance metrics dashboard for reward and success tracking

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Requirements

- Python 3.12
- Gradio
- OpenEnv-compatible environment structure
- No external API access required

## Baseline benchmark

```bash
python baseline/run_baseline.py
```

Benchmark mode reports:

- average reward
- successful triage rate
- urgency breakdown across sampled cases

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
- scenario definitions under `medienv/scenarios.json`
- unit tests in `test_environment.py`

## System behavior

The Gradio interface is designed to:

- display the current observation clearly, including symptoms, severity, age group, access constraints, and mental state
- allow the user to select a scenario
- let the user choose an action and step through the episode
- let the user reset for a new case
- show reward, rationale, completion status, and decision history after each action
- show an AI suggestion for the next action
- show cumulative performance metrics

## How to use the interface

- select a scenario
- read the case summary and current state
- choose an action from the action list
- click `step` to advance the episode
- review the reward, rationale, AI suggestion, and updated history
- click `reset` to load another case

## How Scoring Works

The reward system is designed to encourage safe and appropriate triage decisions.

Reward interpretation:

- small positive reward for information gathering
- small positive reward for supportive low-risk actions
- medium positive reward for correct non-emergency disposition
- large positive reward for correct emergency escalation
- negative reward for unsafe under-triage or incorrect final recommendations

| Action | Good use case | Reward behavior |
| --- | --- | --- |
| `ASK_FOLLOWUP` | when more clarification is needed | small positive reward |
| `PROVIDE_SUPPORT_MESSAGE` | low-risk or reassurance-focused cases | small positive reward if appropriate |
| `RECOMMEND_SELF_CARE` | mild low-risk cases | positive reward if appropriate, negative if unsafe |
| `RECOMMEND_DOCTOR_VISIT` | moderate or high-risk cases | positive reward if appropriate |
| `RECOMMEND_CLINIC` | access-sensitive or higher-risk cases | positive reward if appropriate |
| `ESCALATE_EMERGENCY` | severe or critical cases | highest positive reward when required, strong penalty if missed |

In general:

- correct final actions receive the strongest reward
- information-gathering actions receive a small reward
- unsafe decisions, especially failing to escalate emergencies, receive strong penalties

## Notes

- The environment is designed to run offline.
- No external APIs or cloud databases are required.
- The interface runs from the repository and the deployed Hugging Face Space.
- The OpenEnv entry must exactly match the class name in code: `medienv.environment:HealthTriageEnv`
- `app.py` builds the Gradio UI, connects the `step` and `reset` controls, and launches the interface
- session history is stored locally in `session_logs.json`
- if you see an ICU or runtime descriptor warning, it is usually a local Python/Gradio environment issue rather than a logic error in the triage code
