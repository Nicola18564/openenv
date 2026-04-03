# MediAssist Triage Arena

MediAssist Triage Arena is a polished OpenEnv-style health triage simulator built for hackathon demos. It combines explainable reward design, care-equity signals, benchmark mode, and a judge-friendly Gradio interface.

## Highlights

- Equity-aware triage environment with rural, mobility, language, and financial barriers.
- Explainable scoring with urgency, risk score, verdict, and rationale.
- Scenario-driven evaluation for emergency, clinic, telemedicine, self-care, and doctor-visit decisions.
- Benchmark mode for quick quantitative proof during judging.
- Attractive Gradio dashboard designed for demo day storytelling.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Baseline benchmark

```bash
python baseline/run_baseline.py
```

## OpenEnv entry

The environment entry point is:

```text
medienv.environment:HealthTriageEnv
```
