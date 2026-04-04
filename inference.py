import json
import os

from openai import OpenAI

from client import MediAssistClient


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def build_llm_client():
    kwargs = {"base_url": API_BASE_URL}
    if HF_TOKEN:
        kwargs["api_key"] = HF_TOKEN
    return OpenAI(**kwargs)


def choose_action(client, observation, actions):
    prompt = (
        "You are a health triage assistant. Choose exactly one action from the allowed list.\n"
        f"Observation: {json.dumps(observation)}\n"
        f"Allowed actions: {actions}\n"
        "Return only the action token."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Choose the safest valid triage action."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    action = response.choices[0].message.content.strip()
    if action not in actions:
        return actions[0]
    return action


def main():
    env = MediAssistClient()
    llm = build_llm_client()
    observation = env.reset()
    actions = env.available_actions()

    print("START", json.dumps({"scenario": env.scenario_name, "observation": observation}))

    step_index = 0
    done = False
    while not done and step_index < 3:
        action = choose_action(llm, observation, actions)
        result = env.step(action)
        observation = result.observation
        done = result.done
        print(
            "STEP",
            json.dumps(
                {
                    "index": step_index,
                    "action": action,
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                }
            ),
        )
        step_index += 1

    print("END", json.dumps({"final_observation": observation, "done": done}))


if __name__ == "__main__":
    main()
