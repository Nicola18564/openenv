import json
import os
import sys
from openai import OpenAI
from client import MediAssistClient

# ===== CONFIG =====
MODEL_NAME = "gpt-3.5-turbo"
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_TIMEOUT = 10  # seconds

# ===== BUILD LLM CLIENT SAFELY =====
def build_llm_client():
    """Build an OpenAI client with proper validation and timeout."""
    try:
        # Check if API key is available
        if not API_KEY or API_KEY.strip() == "":
            return None

        kwargs = {
            "base_url": API_BASE_URL,
            "api_key": API_KEY,
            "timeout": API_TIMEOUT,
        }
        client = OpenAI(**kwargs)
        return client
    except Exception as e:
        return None


# ===== SAFE ACTION SELECTION =====
def choose_action(client, observation, actions):
    """Choose an action using LLM or fallback to first action."""
    try:
        # fallback if client not available
        if client is None:
            return actions[0] if actions else None

        prompt = (
            "You are a health triage assistant.\n"
            "Choose exactly one action from the allowed list.\n\n"
            f"Observation:\n{json.dumps(observation)}\n\n"
            f"Allowed actions:\n{actions}\n\n"
            "Return ONLY the action name."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Choose the safest valid triage action."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=API_TIMEOUT,
        )

        # Validate response structure
        if not response or not response.choices:
            return actions[0] if actions else None

        action = response.choices[0].message.content.strip()

        # ensure valid action
        if not actions:
            return None

        if action not in actions:
            return actions[0]

        return action

    except Exception as e:
        return actions[0] if actions else None


# ===== MAIN EXECUTION =====
def main():
    """Main execution loop with comprehensive error handling."""
    try:
        # Initialize environment client (local, not remote)
        env = MediAssistClient()
    except Exception as e:
        print(f"[START] task=medical_triage", flush=True)
        print(f"[END] task=medical_triage score=0.0 steps=0", flush=True)
        return

    try:
        reset_result = env.reset()
    except Exception as e:
        print(f"[START] task=medical_triage", flush=True)
        print(f"[END] task=medical_triage score=0.0 steps=0", flush=True)
        return

    # Reset result is a dictionary from the environment
    observation = reset_result
    actions = env.available_actions()

    print(f"[START] task=medical_triage", flush=True)

    llm = build_llm_client()

    done = False
    step_index = 0
    total_reward = 0.0

    while not done and step_index < 3:
        # ===== SAFE ACTION SELECTION =====
        try:
            action = choose_action(llm, observation, actions)
            if action is None:
                break
        except Exception as e:
            if actions:
                action = actions[0]
            else:
                break

        # ===== SAFE STEP =====
        try:
            result = env.step(action)
        except Exception as e:
            break

        observation = result.observation
        done = result.done
        reward = result.reward
        total_reward += reward

        # Print structured step output
        print(f"[STEP] step={step_index+1} reward={reward:.2f}", flush=True)

        step_index += 1

    # Calculate final score (normalized by steps, capped at 1.0)
    final_score = min(max(total_reward / max(step_index, 1), 0.0), 1.0)

    print(f"[END] task=medical_triage score={final_score:.2f} steps={step_index}", flush=True)

    env.close()


# ===== ENTRY POINT =====
if __name__ == "__main__":
    main()
