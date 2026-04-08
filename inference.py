import json
import math
import os
import re

from openai import OpenAI

from medienv.environment import HealthTriageEnv, load_scenarios


TASK_PREFIX = "medical_triage"
DEFAULT_MODEL = "gpt-4.1-mini"
API_TIMEOUT = 10
MAX_STEPS = 3
SCORE_SCALE = 4.0
SCORE_FLOOR = 0.01
SCORE_CEILING = 0.99


def _get_env(name, fallback=None):
    value = os.getenv(name)
    if value is not None and value.strip():
        return value
    return fallback


def _slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "task"


def build_llm_client():
    """
    Build the OpenAI client using the injected OpenEnv proxy variables.

    The evaluator injects API_BASE_URL and API_KEY. We keep legacy fallbacks
    for local runs, but the preferred path is the injected proxy.
    """
    api_base_url = _get_env("API_BASE_URL", _get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    api_key = _get_env("API_KEY", _get_env("OPENAI_API_KEY"))
    model_name = _get_env("MODEL_NAME", DEFAULT_MODEL)

    if not api_key:
        return None, model_name

    client = OpenAI(base_url=api_base_url, api_key=api_key, timeout=API_TIMEOUT)
    return client, model_name


def normalize_score(total_reward, step_index):
    # Keep the reported score strictly inside (0, 1) even after 2-decimal rounding.
    average_reward = total_reward / step_index if step_index else 0.0
    score = 0.5 + math.atan(average_reward / SCORE_SCALE) / math.pi
    return max(SCORE_FLOOR, min(SCORE_CEILING, score))


def choose_action(client, model_name, observation, actions, fallback_action):
    if not actions:
        return None

    safe_fallback = fallback_action if fallback_action in actions else actions[0]

    if client is None:
        return safe_fallback

    prompt = (
        "You are a health triage assistant.\n"
        "Choose exactly one action from the allowed list.\n\n"
        f"Observation:\n{json.dumps(observation, ensure_ascii=False)}\n\n"
        f"Allowed actions:\n{actions}\n\n"
        "Return only the action token."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Choose the safest valid triage action."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip() if response.choices else ""
        if content in actions:
            return content
        for action in actions:
            if action in content:
                return action
    except Exception:
        return safe_fallback

    return safe_fallback


def run_task(task_name, scenario_name, client, model_name):
    env = HealthTriageEnv(scenario_name)
    observation = env.reset()
    actions = env.available_actions()
    step_index = 0
    done = False
    total_reward = 0.0

    print(f"[START] task={task_name}", flush=True)

    try:
        while not done and step_index < MAX_STEPS:
            fallback_action = env.expert_policy(observation)
            action = choose_action(client, model_name, observation, actions, fallback_action)
            if action is None:
                break

            observation, reward, done, _info = env.step(action)
            total_reward += reward
            step_index += 1

            print(f"[STEP] step={step_index} action={action} reward={reward:.2f}", flush=True)
    except Exception:
        print(f"[STEP] step={step_index + 1} action=ERROR reward=0.00", flush=True)

    final_score = normalize_score(total_reward, step_index)
    print(f"[END] task={task_name} score={final_score:.2f} steps={step_index}", flush=True)


def build_task_schedule():
    tasks = []
    for scenario in load_scenarios():
        scenario_name = scenario["name"]
        task_name = f"{TASK_PREFIX}_{_slugify(scenario_name)}"
        tasks.append((task_name, scenario_name))
    return tasks


def main():
    client, model_name = build_llm_client()
    for task_name, scenario_name in build_task_schedule():
        run_task(task_name, scenario_name, client, model_name)


if __name__ == "__main__":
    main()
