"""
Unit tests for Evaluation API endpoints: dataset listing, running evaluation, and exporting reports.
"""

import json
import unittest
from fastapi.testclient import TestClient
from objectmemory.api.app import create_app
from objectmemory.api.workspace_manager import WorkspaceManager


class TestEvaluationAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_list_datasets(self):
        resp = self.client.get("/api/evaluation/datasets")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        pilot = data[0]
        self.assertEqual(pilot["dataset_id"], "desk_pilot_01")
        self.assertFalse(pilot["is_template"])
        self.assertIn("Pilot sequence", pilot["provenance"])

    def test_run_evaluation(self):
        resp = self.client.post(
            "/api/evaluation/run",
            json={"dataset_id": "desk_pilot_01", "confidence_threshold": 0.35},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("run_id", data)
        self.assertEqual(data["status"], "EVALUATED_PHYSICAL_PILOT")
        self.assertIn("PROVENANCE:", data["provenance_disclaimer"])
        self.assertIn("metrics", data)
        self.assertIn("f1_score", data["metrics"])
        self.assertIn("events", data)
        self.assertGreater(len(data["events"]), 0)

        run_id = data["run_id"]

        # Test export JSON
        json_resp = self.client.get(f"/api/evaluation/runs/{run_id}/export?format=json")
        self.assertEqual(json_resp.status_code, 200)
        self.assertIn("run_id", json_resp.json())

        # Test export CSV
        csv_resp = self.client.get(f"/api/evaluation/runs/{run_id}/export?format=csv")
        self.assertEqual(csv_resp.status_code, 200)
        self.assertIn("visit_id,entity_id", csv_resp.text)


if __name__ == "__main__":
    unittest.main()
