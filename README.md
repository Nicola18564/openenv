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

## Problem Statement

Simulate the human task of preparing for placement or job applications at technology companies. The agent must choose actions that improve skills, build proof artifacts, and meet readiness thresholds before applying.

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

## Action Space

The agent selects one token from the action catalog above on each step.

## Observation / State Space

Each observation includes:

- scenario metadata: `scenario_name`, `scenario_id`, `title`, `summary`, `company_type`, `role`, `stage`
- current progress: `skill_levels`, `projects`, `project_count`, `company_analysis_score`, `testing_score`, `progress_score`, `brand_score`, `resume_score`, `interview_score`, `application_score`, `proof_score`
- readiness and proof status: `skill_average`, `company_match_score`, `readiness_score`, `readiness_state`, `proof_ready`
- step tracking: `step_count`, `applications_submitted`, `feedback_pending`, `history`, `total_reward`
- recommended actions: `recommended_action`, `correct_action`, `current_action_suggestion`
- reward diagnostics: `reward_breakdown`, `resolution_quality`, `growth_plan`

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

This environment includes three primary agent evaluation tasks used by the baseline runner:

- `placement_readiness_startup` — easy: align skills, analyze a product company, and build an AI-focused portfolio.
- `placement_readiness_service` — medium: improve backend skills, build service-grade proof, and meet testing/readiness thresholds.
- `placement_readiness_internship` — hard: develop a full-stack portfolio, strengthen branding and interview readiness, and satisfy proof gates.

Each task is scored by the environment grader and provides partial progress rewards along the trajectory.

## Reward and Grading

- The grader computes rewards per action and normalizes them into a bounded score range.
- Partial progress is rewarded for learning, project building, testing, tracking, branding, resume improvement, and interview practice.
- Unsafe or premature actions such as applying too early are penalized.
- The baseline inference runner reports a normalized score for each task.

## Benchmark

```bash
python baseline/run_baseline.py
```

## Inference

```bash
API_KEY="your_key"
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

- Push to Hugging Face using your `HF_TOKEN`:

```bash
export HF_TOKEN="your_token"
python -m openenv.cli push .
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

## Evaluation Criteria

This environment is designed to meet the OpenEnv Round 1 evaluation parameters:

### Real-world utility (30%)
- Models the genuine task of preparing for job applications in technology companies.
- Agents learn to analyze companies, build skills, create portfolios, and apply strategically.
- Useful for training agents on career development and decision-making in professional contexts.

### Task & grader quality (25%)
- Three tasks with increasing difficulty: startup (easy), service (medium), internship (hard).
- Graders use deterministic scoring based on proof targets (projects, skills, readiness).
- Success measured by meeting eligibility thresholds and applying at the right time.

### Environment design (20%)
- Clean state management with typed Pydantic models for actions, observations, and state.
- Sensible action space (20 discrete actions) and rich observation space (skill levels, projects, scores).
- Reward shaping provides partial progress signals; episodes end on application or max steps.
- Proper episode boundaries with done flags and info dictionaries.

### Code quality & spec compliance (15%)
- Full OpenEnv spec compliance with `step()`, `reset()`, and `state` properties.
- Clean project structure with separate modules for environment, grader, tasks, and server.
- Typed models, comprehensive tests, working Dockerfile, and OpenEnv validation passing.
- Well-documented with this README and inline comments.

### Creativity & novelty (10%)
- Novel domain: career readiness simulation with proof-based gating.
- Interesting mechanics: skill progression, project building, and strategic application timing.
- Original approach combining RL with professional development scenarios.

## Notes

- The environment is offline-first and does not depend on a cloud database.
- LLM-assisted inference uses injected `API_BASE_URL` and `API_KEY` values when available.
- The OpenEnv entry point is `medienv.environment:PlacementIntelligenceEnv`.

## Links

- GitHub: https://github.com/Nicola18564/openenv
- Hugging Face Space: https://huggingface.co/spaces/Shivakumar184510/mediassist-triage-arena
- Live App: https://Shivakumar184510-mediassist-triage-arena.hf.space
