"""Local client for the placement-intelligence environment."""

from dataclasses import dataclass

from medienv.environment import PlacementIntelligenceEnv, load_scenarios


SCENARIOS = {item["name"]: item for item in load_scenarios()}


@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict


class PlacementIntelligenceClient:
    """Lightweight local client for interacting with the placement environment."""

    def __init__(self, scenario_name=None, seed=None):
        self.scenario_name = scenario_name or next(iter(SCENARIOS))
        self.env = PlacementIntelligenceEnv(SCENARIOS[self.scenario_name], seed=seed)
        self._last_state = None

    @staticmethod
    def list_scenarios():
        return list(SCENARIOS.keys())

    def available_actions(self):
        return self.env.available_actions()

    def reset(self, scenario_name=None):
        if scenario_name is not None:
            if scenario_name not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario_name}")
            self.scenario_name = scenario_name
            self.env = PlacementIntelligenceEnv(SCENARIOS[self.scenario_name])
        self._last_state = self.env.reset()
        return self._last_state

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._last_state = observation
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self):
        return self._last_state or self.env.reset()

    def benchmark(self, episodes=20):
        return self.env.benchmark(episodes=episodes)

    def close(self):
        return None


MediAssistClient = PlacementIntelligenceClient


if __name__ == "__main__":
    client = PlacementIntelligenceClient()
    state = client.reset()
    print("Scenario:", client.scenario_name)
    print("Initial state:", state)
    result = client.step(client.env.expert_policy(state))
    print("Step result:", result)
