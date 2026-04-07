import json
import os
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
            print("WARNING: OPENAI_API_KEY not set, will use fallback policy")
            return None

        kwargs = {
            "base_url": API_BASE_URL,
            "api_key": API_KEY,
            "timeout": API_TIMEOUT,
        }
        client = OpenAI(**kwargs)
        print(f"LLM client initialized with base_url: {API_BASE_URL}")
        return client
    except Exception as e:
        print(f"LLM init failed: {str(e)}")
        return None


# ===== SAFE ACTION SELECTION =====
def choose_action(client, observation, actions):
    """Choose an action using LLM or fallback to first action."""
    try:
        # fallback if client not available
        if client is None:
            print("No LLM client available, using fallback policy")
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
            print("ERROR: Invalid response structure from API")
            return actions[0] if actions else None

        action = response.choices[0].message.content.strip()

        # ensure valid action
        if not actions:
            print("ERROR: No actions available")
            return None

        if action not in actions:
            print(f"WARNING: Invalid action '{action}' returned, using fallback")
            return actions[0]

        return action

    except Exception as e:
        print(f"ERROR in choose_action: {str(e)}")
        return actions[0] if actions else None


# ===== MAIN EXECUTION =====
def main():
    """Main execution loop with comprehensive error handling."""
    try:
        # Initialize environment client (local, not remote)
        env = MediAssistClient()
        print("Environment client initialized successfully")
    except Exception as e:
        print(f"FATAL ERROR connecting to env: {str(e)}")
        return

    try:
        reset_result = env.reset()
        print("Environment reset successfully")
    except Exception as e:
        print(f"FATAL ERROR during reset: {str(e)}")
        return

    # Reset result is a dictionary from the environment
    observation = reset_result
    actions = env.available_actions()

    print(f"Initial observation: {json.dumps(observation)}")
    print(f"Available actions: {actions}")

    llm = build_llm_client()

    done = False
    step_index = 0

    while not done and step_index < 3:
        # ===== SAFE ACTION SELECTION =====
        try:
            action = choose_action(llm, observation, actions)
            if action is None:
                print(f"ERROR: No valid action could be selected")
                break
        except Exception as e:
            print(f"FATAL ERROR choosing action: {str(e)}")
            if actions:
                action = actions[0]
            else:
                break

        # ===== SAFE STEP =====
        try:
            result = env.step(action)
        except Exception as e:
            print(f"ERROR in env.step: {str(e)}")
            break

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

    print(f"Episode completed. Steps taken: {step_index}")
    env.close()


# ===== ENTRY POINT =====
if __name__ == "__main__":
    main()
