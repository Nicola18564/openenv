---
title: Placement Intelligence Arena
emoji: "🚀"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Placement Intelligence Arena

An explainable OpenEnv-compatible placement-readiness environment for structured self-growth, proof-based eligibility, and final job application decisions.

## What It Does

The environment simulates a realistic path from analysis to application:

Analyze -> Learn -> Build -> Test -> Track -> Show -> Prove -> Apply -> Improve -> Grow

It supports these modules:

- Company and skill analysis
- Skill development
- Project building
- AI and advanced system design
- Testing and validation
- Performance tracking
- Personal branding
- Resume optimization
- Smart applications
- Interview preparation
- Feedback and improvement
- Opportunity exploration
- Proof-based eligibility gating

## Key Actions

- `ANALYZE_COMPANY`
- `EXTRACT_SKILLS`
- `UPDATE_SKILL_MAP`
- `LEARN_PYTHON`
- `LEARN_DSA`
- `LEARN_AI`
- `LEARN_BACKEND`
- `BUILD_AI_PROJECT`
- `BUILD_BACKEND_PROJECT`
- `BUILD_FULLSTACK_PROJECT`
- `WRITE_TESTS`
- `TRACK_PROGRESS`
- `PUBLISH_GITHUB`
- `OPTIMIZE_RESUME`
- `APPLY_JOB`
- `PRACTICE_INTERVIEW`
- `REVIEW_FAILURE`
- `EXPLORE_HACKATHON`
- `GATHER_CERTIFICATION`
- `VALIDATE_READINESS`

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## OpenEnv Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Benchmark

```bash
python baseline/run_baseline.py
```

## Inference

```bash
python inference.py
```

The inference script prints validator-friendly structured blocks:

```text
[START]
[STEP]
[END]
```

## Testing

```bash
python -m unittest test_environment.py
```

## Repo Contents

- `app.py` - Gradio + FastAPI interface
- `server/app.py` - OpenEnv-native server entry
- `server/models.py` - OpenEnv schema models
- `server/` - OpenEnv wrapper around the placement engine
- `medienv/environment.py` - Core placement environment
- `medienv/grader.py` - Reward and grading logic
- `medienv/tasks.py` - Scenario definitions and action catalog
- `baseline/run_baseline.py` - Benchmark runner
- `inference.py` - Structured validator script
- `client.py` - Local client helper
- `test_environment.py` - Unit tests
- `Dockerfile` - Docker runtime
- `openenv.yaml` - OpenEnv runtime config
- `pyproject.toml` - Packaging metadata

## Notes

- The environment is offline-first and does not depend on a cloud database.
- LLM-assisted inference uses injected `API_BASE_URL` and `API_KEY` values when available.
- The OpenEnv entry point is `medienv.environment:PlacementIntelligenceEnv`.

## Links

- GitHub: https://github.com/Nicola18564/openenv
- Hugging Face Space: https://huggingface.co/spaces/Shivakumar184510/mediassist-triage-arena
- Live App: https://Shivakumar184510-mediassist-triage-arena.hf.space
