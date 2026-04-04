import json
import unittest
from pathlib import Path

import app
from fastapi.testclient import TestClient
from medienv.environment import HealthTriageEnv, load_scenarios


class HealthTriageEnvironmentTests(unittest.TestCase):
    def test_load_scenarios(self):
        scenarios = load_scenarios()
        self.assertGreaterEqual(len(scenarios), 3)
        self.assertIn("severity", scenarios[0])

    def test_reset_returns_state(self):
        env = HealthTriageEnv(seed=1)
        state = env.reset()
        self.assertIn("symptoms", state)
        self.assertIn("severity", state)
        self.assertIn("risk_score", state)
        self.assertIn("urgency", state)

    def test_invalid_action_penalized(self):
        env = HealthTriageEnv(seed=1)
        env.reset()
        _, reward, done, info = env.step("BAD_ACTION")
        self.assertEqual(reward, -2.0)
        self.assertFalse(done)
        self.assertIn("error", info)

    def test_import_name_works(self):
        env = HealthTriageEnv(seed=1)
        state = env.reset()
        self.assertIn("done", state)

    def test_expert_policy_returns_known_action(self):
        env = HealthTriageEnv(seed=1)
        state = env.reset()
        self.assertIn(env.expert_policy(state), env.available_actions())

    def test_emergency_case_rewards_safe_escalation(self):
        scenarios = load_scenarios()
        emergency_case = next(item for item in scenarios if item["name"] == "Fall emergency")
        env = HealthTriageEnv(emergency_case)
        env.reset()
        state, reward, done, info = env.step("ESCALATE_EMERGENCY")
        self.assertTrue(done)
        self.assertGreater(reward, 0)
        self.assertEqual(info["resolution_quality"], "safe_final")
        self.assertEqual(state["total_reward"], reward)

    def test_support_message_is_useful_for_low_risk_distress(self):
        scenarios = load_scenarios()
        support_case = next(item for item in scenarios if item["name"] == "Panic with palpitations")
        env = HealthTriageEnv(support_case)
        env.reset()
        _, reward, done, info = env.step("PROVIDE_SUPPORT_MESSAGE")
        self.assertFalse(done)
        self.assertGreater(reward, 0)
        self.assertIn("empathy", info["reward_breakdown"])

    def test_benchmark_returns_expected_fields(self):
        env = HealthTriageEnv(seed=1)
        summary = env.benchmark(episodes=5)
        self.assertIn("average_reward", summary)
        self.assertIn("successful_triage_rate", summary)
        self.assertIn("urgency_breakdown", summary)

    def test_openenv_reset_endpoint(self):
        client = TestClient(app.app)
        response = client.post("/reset", json={"scenario_name": "Mild headache"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("observation", payload)
        self.assertIn("available_actions", payload)
        self.assertEqual(payload["scenario"], "Mild headache")

    def test_openenv_step_endpoint(self):
        client = TestClient(app.app)
        client.post("/reset", json={"scenario_name": "Fall emergency"})
        response = client.post("/step", json={"action": "ESCALATE_EMERGENCY"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("reward", payload)
        self.assertIn("info", payload)
        self.assertTrue(payload["done"])

    def test_session_log_is_capped(self):
        original_path = app.SESSION_LOG_PATH
        test_path = Path("session_logs_test.json")
        try:
            app.SESSION_LOG_PATH = test_path
            existing = [{"idx": i} for i in range(1005)]
            app.SESSION_LOG_PATH.write_text(json.dumps(existing), encoding="utf-8")
            state = {
                "done": False,
                "step_count": 1,
                "symptoms": "test",
                "age_group": "adult",
                "severity": "low",
                "rural_access": False,
                "mental_state": "neutral",
                "fall_flag": False,
                "epidemic_flag": False,
                "risk_score": 18,
                "urgency": "low",
                "total_reward": 0.0,
                "context_collected": False,
                "support_provided": False,
                "history": [],
            }
            app.save_session_log("Mild headache", state, {"ok": True}, 1.0)
            saved = json.loads(app.SESSION_LOG_PATH.read_text(encoding="utf-8"))
            self.assertLessEqual(len(saved), 500)
        finally:
            app.SESSION_LOG_PATH = original_path
            if test_path.exists():
                try:
                    test_path.unlink()
                except PermissionError:
                    pass


if __name__ == "__main__":
    unittest.main()
