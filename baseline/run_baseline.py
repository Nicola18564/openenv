from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medienv.environment import HealthTriageEnv


def run_baseline(episodes=20):
    env = HealthTriageEnv(seed=11)
    rewards = []
    solved = 0

    for _ in range(episodes):
        state = env.reset()
        action = env.expert_policy(state)
        _, reward, done, info = env.step(action)
        rewards.append(reward)
        solved += int(done and action == info["correct_action"])

    average_reward = sum(rewards) / len(rewards)
    success_rate = (solved / episodes) * 100
    print(f"Episodes: {episodes}")
    print(f"Average reward: {average_reward:.2f}")
    print(f"Successful triage rate: {success_rate:.1f}%")


if __name__ == "__main__":
    run_baseline()
