import json
import unittest
from pathlib import Path

import app
from fastapi.testclient import TestClient
from medienv.environment import PlacementIntelligenceEnv, load_scenarios
from server.app import app as openenv_app


class PlacementIntelligenceEnvironmentTests(unittest.TestCase):
    def test_load_scenarios(self):
        scenarios = load_scenarios()
        self.assertGreaterEqual(len(scenarios), 5)
        self.assertIn("target_company", scenarios[0])
        self.assertIn("proof_targets", scenarios[0])
        self.assertEqual(scenarios[0]["correct_action"], "APPLY_JOB")

    def test_reset_returns_state(self):
        env = PlacementIntelligenceEnv(seed=1)
        state = env.reset()
        self.assertIn("target_company", state)
        self.assertIn("role", state)
        self.assertIn("readiness_score", state)
        self.assertIn("proof_ready", state)

    def test_invalid_action_penalized(self):
        env = PlacementIntelligenceEnv(seed=1)
        env.reset()
        _, reward, done, info = env.step("BAD_ACTION")
        self.assertEqual(reward, -3.0)
        self.assertFalse(done)
        self.assertIn("error", info)

    def test_import_name_works(self):
        env = PlacementIntelligenceEnv(seed=1)
        state = env.reset()
        self.assertIn("done", state)

    def test_expert_policy_returns_known_action(self):
        env = PlacementIntelligenceEnv(seed=1)
        state = env.reset()
        self.assertIn(env.expert_policy(state), env.available_actions())

    def test_apply_job_rewards_readiness(self):
        scenarios = load_scenarios()
        scenario = next(item for item in scenarios if item["name"] == "AI Startup Hiring Sprint")
        env = PlacementIntelligenceEnv(scenario)
        env.reset()
        env.skill_levels = {"python": 88, "dsa": 82, "ai": 90, "backend": 78, "web": 75}
        env.projects = ["AI project", "Backend project"]
        env.company_analysis_score = 80
        env.testing_score = 88
        env.progress_score = 84
        env.brand_score = 82
        env.resume_score = 86
        env.interview_score = 84
        state, reward, done, info = env.step("APPLY_JOB")
        self.assertTrue(done)
        self.assertGreater(reward, 0)
        self.assertEqual(info["resolution_quality"], "submission_ready")
        self.assertTrue(state["proof_ready"])

    def test_benchmark_returns_expected_fields(self):
        env = PlacementIntelligenceEnv(seed=1)
        summary = env.benchmark(episodes=5)
        self.assertIn("average_reward", summary)
        self.assertIn("successful_readiness_rate", summary)
        self.assertIn("proof_ready_rate", summary)
        self.assertIn("application_rate", summary)

    def test_openenv_reset_endpoint(self):
        client = TestClient(app.app)
        response = client.post("/reset", json={"scenario_name": "AI Startup Hiring Sprint"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("observation", payload)
        self.assertIn("available_actions", payload)
        self.assertEqual(payload["scenario"], "AI Startup Hiring Sprint")

    def test_openenv_step_endpoint(self):
        client = TestClient(app.app)
        client.post("/reset", json={"scenario_name": "AI Startup Hiring Sprint"})
        response = client.post("/step", json={"action": "ANALYZE_COMPANY"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("reward", payload)
        self.assertIn("info", payload)
        self.assertFalse(payload["done"])

    def test_openenv_native_reset_endpoint(self):
        client = TestClient(openenv_app)
        response = client.post("/reset", json={"scenario_name": "AI Startup Hiring Sprint"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("observation", payload)
        self.assertIn("reward", payload)
        self.assertIn("done", payload)
        self.assertEqual(payload["observation"]["scenario_name"], "AI Startup Hiring Sprint")

    def test_openenv_native_step_endpoint(self):
        client = TestClient(openenv_app)
        client.post("/reset", json={"scenario_name": "AI Startup Hiring Sprint"})
        response = client.post("/step", json={"action": {"action": "ANALYZE_COMPANY"}})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("observation", payload)
        self.assertIn("reward", payload)
        self.assertGreaterEqual(payload["reward"], 0)
        self.assertFalse(payload["done"])

    def test_openenv_root_route(self):
        client = TestClient(openenv_app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

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
                "target_company": "test",
                "role": "engineer",
                "stage": "proof",
                "focus_modules": [],
                "skill_average": 60,
                "readiness_score": 72,
                "readiness_state": "almost_ready",
                "proof_ready": False,
                "total_reward": 0.0,
                "applications_submitted": 0,
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
