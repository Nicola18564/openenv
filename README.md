---
title: MediAssist Triage Arena
emoji: "🏥"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# MediAssist Triage Arena - Health Triage OpenEnv

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered health triage system for structured patient assessment and care routing using the OpenEnv framework. This system combines explainability, equity awareness, and reward-based learning to support medical decision-making.

## 🎯 Features

### Core Functionality
- **Scenario-based triage environment** with 5+ patient case scenarios
- **Action selection** supporting 6 different care pathways:
  - Ask follow-up questions
  - Provide support message
  - Recommend self-care
  - Recommend clinic visit
  - Recommend doctor visit
  - Escalate to emergency care

### Explainability & Transparency
- **Detailed reward breakdowns** across 5 dimensions:
  - Safety (emergency risk management)
  - Sequence fit (action order appropriateness)
  - Access fit (equity-aware routing)
  - Empathy (patient support)
  - Efficiency (resource optimization)
- Reasoning rationales for each decision
- Care plan suggestions based on expert policy
- Structured output format (`[START]`, `[STEP]`, `[END]` blocks) for validation

### Equity & Fairness
- Rural access barriers detection
- Elderly patient vulnerability assessment
- Epidemic/outbreak awareness
- Mental state tracking and support
- Chronic condition management

### User Interfaces
- **Gradio web interface** for interactive exploration
- **RESTful API** endpoints for programmatic access
- **Local client library** for Python integration
- **Batch inference script** for evaluation

### Monitoring & Analytics
- Session logging to JSON for offline analysis
- Performance metrics dashboard:
  - Episodes completed
  - Safe vs unsafe decisions
  - Average reward tracking
  - Success rates
- Step-by-step action history with rewards

## 🚀 Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/Nicola18564/openenv.git
cd openenv

# Install dependencies
pip install -r requirements.txt

# Run web interface
python app.py
```

Then visit: `http://localhost:7860`

### OpenEnv Native Server

If you want the reference-style OpenEnv interface instead of the Gradio UI, run:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

That entry point powers `openenv.yaml` and uses the OpenEnv wrapper in `server/triage_environment.py` plus the schema models in `server/models.py`.

The local `app.py` UI is still available for interactive debugging and visual exploration.
OpenEnv responses follow the reference server shape: top-level `reward` and `done`, with the current observation nested under `observation`.

### Docker Deployment

```bash
docker build -t mediassist-triage .
docker run -p 7860:7860 mediassist-triage
```

## 📊 Project Structure

```
openenv/
├── app.py                    # Gradio + FastAPI web interface
├── inference.py              # Batch inference with structured output
├── client.py                 # Python client library
├── test_environment.py       # Unit tests
├── requirements.txt          # Dependencies
├── Dockerfile               # Container configuration
├── README.md                # This file
├── baseline/
│   └── run_baseline.py      # Performance benchmark script
├── medienv/
│   ├── __init__.py
│   ├── environment.py       # Core HealthTriageEnvironment class
│   ├── grader.py           # Reward computation logic
│   ├── tasks.py            # Task definitions
│   └── scenarios.json       # 5+ patient scenarios
└── server/
    └── app.py              # Server entry point
```

## 💻 API Endpoints

### REST API (FastAPI)

```bash
# Health check
GET /health

# Reset environment
POST /reset
Body: { "scenario_name": "Mild headache" }  # optional

# Take action
POST /step
Body: { "action": "ASK_FOLLOWUP" }

# Get current state
GET /state
POST /state

# List actions
GET /actions
```

### Python Client

```python
from client import MediAssistClient

# Initialize
env = MediAssistClient(scenario_name="Mild headache")

# Reset and get initial state
state = env.reset()
print(state.observation)

# Take actions
result = env.step("ASK_FOLLOWUP")
print(f"Reward: {result.reward}, Done: {result.done}")

# Available actions
actions = env.available_actions()
```

## 🧪 Inference & Evaluation

### Running Inference with Structured Output

```bash
python inference.py
```

This will:
1. Initialize a local triage environment
2. Emit a single validator-compliant episode
3. Execute up to 3 steps with action selection
4. Output structured blocks for validation:
   ```
   [START] task=medical_triage env=medical_triage model=gpt-4.1-mini
   [STEP] step=1 action=ASK_FOLLOWUP reward=0.00 done=false error=null
   [STEP] step=2 action=REQUEST_VITALS reward=0.00 done=false error=null
   [END] success=true steps=2 rewards=0.00,0.00
   ```

### LLM Integration

The inference script uses the OpenAI client with an HF token:

```bash
HF_TOKEN="hf_..." API_BASE_URL="https://api.openai.com/v1" MODEL_NAME="gpt-4.1-mini" python inference.py
```

`HF_TOKEN` is required.

### Benchmark Mode

```bash
python baseline/run_baseline.py
```

Reports aggregate metrics across 20 episodes.

## 📋 Scenarios

The system includes diverse patient scenarios:

1. **Mild headache** - Low risk, reassurance appropriate
2. **Fever and cough** - Medium risk with epidemic context
3. **Fall emergency** - High risk elderly patient
4. **Pediatric concern** - Child-specific vulnerability
5. **Chronic disorder** - Complex multi-condition case

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | FastAPI + Gradio |
| Server | Uvicorn |
| Environment | Custom Python implementation |
| Testing | unittest |
| Container | Docker |
| LLM Integration | OpenAI SDK |

## ✅ Validation & Testing

### Phase 1: Basic Functionality ✓
- Environment initialization
- State/action interface compliance
- No unhandled exceptions

### Phase 2: Structured Output ✓
- Properly formatted `[START]`/`[STEP]`/`[END]` blocks
- All output to stdout (not stderr)
- Proper output flushing

### Phase 3+: Advanced Features
- Improved reward functions
- LLM policy optimization
- Equity metrics validation

## 🔧 Configuration

### Environment Variables

```bash
# LLM Configuration
HF_TOKEN              # Required Hugging Face token
API_BASE_URL          # Default: https://api.openai.com/v1
MODEL_NAME            # Default: gpt-4.1-mini
API_TIMEOUT           # API call timeout in seconds (default: 10)

# Server Configuration
GRADIO_SERVER_NAME    # Bind address (default: 127.0.0.1)
GRADIO_SERVER_PORT    # Port number (default: auto-detect 7860-7890)
PORT                  # Alternative port specification
```

### Runtime Options

Modify in `app.py` or `inference.py` as needed:
- `DEFAULT_SCENARIO_NAME`
- `MODEL_NAME` (for LLM calls)
- `MAX_STEPS` per episode
- Reward weighting

## 📝 Error Handling

The system includes robust error handling:
- Missing API keys: Graceful fallback to heuristic policies
- API timeouts: 10-second timeout with automatic retry
- Invalid actions: Validation with informative error messages
- Network failures: Comprehensive exception catching

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional patient scenarios
- Improved reward functions
- Better LLM prompts for action selection
- Healthcare domain expertise integration
- Performance optimizations

## 📄 License

This project is part of the PyTorch Hackathon submission.

## 🔗 Links

- **GitHub**: https://github.com/Nicola18564/openenv
- **Hugging Face Space**: https://huggingface.co/spaces/Shivakumar184510/mediassist-triage-arena
- **API Documentation**: Available at `/docs` endpoint when running server

## 📞 Support

For issues or questions:
1. Check the GitHub Issues
2. Review test cases in `test_environment.py`
3. Consult scenario definitions in `medienv/scenarios.json`

---

**Built with ❤️ for healthcare equity and explainable AI**
- prints structured `START`, `STEP`, and `END` logs

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
