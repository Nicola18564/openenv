import json
import os

from openai import OpenAI

from client import MediAssistClient


TASK_NAME = "medical_triage"
DEFAULT_MODEL = "gpt-4.1-mini"
API_TIMEOUT = 10


def _get_env(name, fallback=None):
    value = os.getenv(name)
    if value is not None and value.strip():
        return value
    return fallback


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


def choose_action(client, model_name, observation, actions):
    if not actions:
        return None

    if client is None:
        return actions[0]

    prompt = (
        "You are a health triage assistant.\n"
        "Choose exactly one action from the allowed list.\n\n"
        f"Observation:\n{json.dumps(observation, ensure_ascii=False)}\n\n"
        f"Allowed actions:\n{actions}\n\n"
        "Return only the action token."
    )

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
    return actions[0]


def main():
    env = MediAssistClient()
    observation = env.reset()
    actions = env.available_actions()
    client, model_name = build_llm_client()

    print(f"[START] task={TASK_NAME}", flush=True)

    step_index = 0
    done = False
    total_reward = 0.0

    while not done and step_index < 3:
        action = choose_action(client, model_name, observation, actions)
        if action is None:
            break

        result = env.step(action)
        observation = result.observation
        done = result.done
        total_reward += result.reward
        step_index += 1

        print(f"[STEP] step={step_index} action={action} reward={result.reward:.2f}", flush=True)

    final_score = total_reward / step_index if step_index else 0.0
    print(f"[END] task={TASK_NAME} score={final_score:.2f} steps={step_index}", flush=True)


if __name__ == "__main__":
    main()
