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

## Tasks and Difficulty

This environment includes three primary agent evaluation tasks:

- `placement_readiness_startup` — easy: align skills and build an AI portfolio.
- `placement_readiness_service` — medium: strengthen backend and testing proof.
- `placement_readiness_internship` — hard: build a full-stack portfolio and polish interview readiness.

Each task has a deterministic scorer and progress-based reward shaping.

## Benchmark

```bash
python baseline/run_baseline.py
```

## Inference

```bash
OPENAI_API_KEY="your_key"
MODEL_NAME="gpt-4.1-mini"
API_BASE_URL="https://api.openai.com/v1"
python inference.py
```

The inference script uses the OpenAI client and prints validator-friendly structured blocks:

```text
[START] task=...
[STEP] step=... reward=...
[END] task=... score=... steps=...
```

## Deployment

- Build the container locally:

```bash
docker build -t openenv-test .
```

- Run the app in Docker:

```bash
docker run -p 7860:7860 openenv-test
```

- Validate the environment with OpenEnv CLI:

```bash
python -m openenv.cli validate .
```

## Testing

```bash
python -m unittest test_environment.py
```

## Repo Contents

- `app.py` - Gradio + FastAPI interface
- `server/app.py` - OpenEnv-native server entry
- `server/models.py` - OpenEnv schema models
- `server/environment.py` - OpenEnv wrapper around the placement engine
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
