import json
import unittest
from pathlib import Path

import app
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

    def test_invalid_action_penalized(self):
        env = HealthTriageEnv(seed=1)
        env.reset()
        _, reward, done, info = env.step("BAD_ACTION")
        self.assertEqual(reward, -1.0)
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
