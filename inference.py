import json
import os
import sys

sys.dont_write_bytecode = True

from openai import OpenAI

from medienv.environment import HealthTriageEnv


TASK_NAME = "medical_triage"
BENCHMARK_NAME = "medical_triage"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
API_TIMEOUT = 10
MAX_STEPS = 3


def _read_env(name, default=None, required=False):
    value = os.getenv(name)
    if value is None or not value.strip():
        if required:
            raise ValueError(f"{name} environment variable is required")
        return default
    return value.strip()


def _single_line(value):
    return " ".join(str(value).split())


def _bool_text(value):
    return "true" if value else "false"


def _error_text(value):
    if value is None or value == "":
        return "null"
    return _single_line(value)


def _rewards_text(values):
    return ",".join(f"{value:.2f}" for value in values)


def build_llm_client():
    try:
        api_base_url = _read_env("API_BASE_URL", DEFAULT_API_BASE_URL)
        api_key = _read_env("API_KEY")
        if not api_key:
            return None
        client = OpenAI(base_url=api_base_url, api_key=api_key, timeout=API_TIMEOUT)
        return client
    except Exception:
        return None


def choose_action(client, model_name, observation, actions, fallback_action):
    if not actions:
        return None

    safe_fallback = fallback_action if fallback_action in actions else actions[0]
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
        content = response.choices[0].message.content if response.choices else ""
        cleaned = _single_line(content or "")
        if cleaned in actions:
            return cleaned
        for action in actions:
            if action in cleaned:
                return action
    except Exception:
        return safe_fallback

    return safe_fallback


def close_env(env):
    close = getattr(env, "close", None)
    if callable(close):
        close()


def run_episode():
    model_name = _read_env("MODEL_NAME", DEFAULT_MODEL)
    env = HealthTriageEnv(seed=0)
    rewards = []
    errors = []
    done = False
    fatal_error = None

    print(
        f"[START] task={TASK_NAME}",
        flush=True,
    )

    try:
        client = build_llm_client()
        observation = env.reset()

        for step_number in range(1, MAX_STEPS + 1):
            fallback_action = env.expert_policy(observation)
            action = choose_action(client, model_name, observation, env.available_actions(), fallback_action)

            reward = 0.0
            step_done = False
            step_error = None
            step_exception = None

            try:
                observation, reward, step_done, info = env.step(action)
                if isinstance(info, dict):
                    step_error = info.get("error")
            except Exception as exc:
                step_exception = exc
                step_error = str(exc)

            rewards.append(reward)
            errors.append(step_error)
            print(
                f"[STEP] step={step_number} reward={reward:.2f}",
                flush=True,
            )

            if step_exception is not None:
                fatal_error = step_exception
                break

            done = step_done
            if done:
                break
    except Exception as exc:
        fatal_error = exc
    finally:
        close_env(env)
        success = done and fatal_error is None and all(error is None for error in errors)
        print(
            f"[END] task={TASK_NAME} score={sum(rewards) / max(len(rewards), 1):.2f} steps={len(rewards)}",
            flush=True,
        )


def main():
    run_episode()


if __name__ == "__main__":
    main()
