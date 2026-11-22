"""
Tests for Phase 15 Integration:
- Workspace experiment export endpoint.
- Batch image sequence processing.
- Comprehensive 13-visit physical benchmark manifest loader.
- Physical dataset integrity constraints.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src is in python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from objectmemory.api.app import app
from objectmemory.api.workspace_manager import WorkspaceManager
from objectmemory.datasets.physical_workbench import (
    PhysicalWorkbenchDataset,
    DatasetCaptureStatus,
    PhysicalVisibility,
)
from objectmemory.domain.states import EpistemicState


class TestPhase15WorkspaceExportAndBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspaces_root = Path(self.temp_dir.name)
        self.manager = WorkspaceManager(str(self.workspaces_root))
        # Override manager in app state
        app.state.manager = self.manager
        self.client = TestClient(app)

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def _create_sample_image_bytes(self, color=(120, 180, 240), label="visit") -> bytes:
        img = np.full((300, 400, 3), color, dtype=np.uint8)
        cv2.putText(img, label, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def test_workspace_export_empty_and_populated(self) -> None:
        """Verify GET /api/workspaces/{id}/export returns complete experiment structure."""
        # 1. Create workspace
        create_resp = self.client.post(
            "/api/workspaces",
            json={"workspace_id": "ws-export-01", "name": "Export Test Workspace"},
        )
        self.assertEqual(create_resp.status_code, 201)
        ws_id = create_resp.json()["workspace"]["workspace_id"]

        # 2. Export empty workspace
        exp_resp = self.client.get(f"/api/workspaces/{ws_id}/export")
        self.assertEqual(exp_resp.status_code, 200)
        data = exp_resp.json()
        self.assertEqual(data["workspace"]["workspace_id"], ws_id)
        self.assertEqual(data["workspace"]["name"], "Export Test Workspace")
        self.assertEqual(len(data["sessions"]), 0)
        self.assertIn("exported_at", data)

        # 3. Add single observation
        img1 = self._create_sample_image_bytes(label="Visit 1")
        obs_resp = self.client.post(
            f"/api/workspaces/{ws_id}/observations",
            data={"session_name": "Baseline Visit"},
            files={"image": ("v01.png", io.BytesIO(img1), "image/png")},
        )
        self.assertEqual(obs_resp.status_code, 201)

        # 4. Export populated workspace
        exp_resp2 = self.client.get(f"/api/workspaces/{ws_id}/export")
        self.assertEqual(exp_resp2.status_code, 200)
        data2 = exp_resp2.json()
        self.assertEqual(data2["workspace"]["total_sessions"], 1)
        self.assertEqual(len(data2["sessions"]), 1)
        self.assertIn("exported_at", data2)

    def test_batch_observation_ingest(self) -> None:
        """Verify POST /api/workspaces/{id}/observations/batch chronologically ingests files."""
        create_resp = self.client.post(
            "/api/workspaces",
            json={"workspace_id": "ws-batch-01", "name": "Batch Ingest Workspace"},
        )
        self.assertEqual(create_resp.status_code, 201)
        ws_id = create_resp.json()["workspace"]["workspace_id"]

        img1 = self._create_sample_image_bytes(color=(100, 100, 100), label="frame_01")
        img2 = self._create_sample_image_bytes(color=(120, 120, 120), label="frame_02")
        img3 = self._create_sample_image_bytes(color=(140, 140, 140), label="frame_03")

        files = [
            ("files", ("visit_03.png", io.BytesIO(img3), "image/png")),
            ("files", ("visit_01.png", io.BytesIO(img1), "image/png")),
            ("files", ("visit_02.png", io.BytesIO(img2), "image/png")),
        ]

        batch_resp = self.client.post(
            f"/api/workspaces/{ws_id}/observations/batch",
            data={"confidence_threshold": 0.25, "detector_name": "mock"},
            files=files,
        )
        self.assertEqual(batch_resp.status_code, 201)
        results = batch_resp.json()
        self.assertEqual(len(results), 3)

        # Confirm alphabetical sorting: visit_01 should be processed first
        self.assertIn("visit_01", results[0]["session_name"].lower())
        self.assertIn("visit_02", results[1]["session_name"].lower())
        self.assertIn("visit_03", results[2]["session_name"].lower())

        # Verify export now reflects 3 visits
        exp_resp = self.client.get(f"/api/workspaces/{ws_id}/export")
        self.assertEqual(exp_resp.json()["workspace"]["total_sessions"], 3)


class TestPhase15PhysicalBenchmarkManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.dataset_dir = self.repo_root / "data" / "physical_workbench_v1"

    def test_manifest_contains_exp01_comprehensive_sequence(self) -> None:
        """Verify EXP01_comprehensive_workbench exists with 13 sequential visits."""
        lifecycle_manifest = self.dataset_dir / "manifest_lifecycle.json"
        self.assertTrue(lifecycle_manifest.is_file(), f"Missing {lifecycle_manifest}")

        dataset = PhysicalWorkbenchDataset.load(str(lifecycle_manifest))
        self.assertEqual(dataset.capture_status, DatasetCaptureStatus.AWAITING_PHYSICAL_DATA)
        self.assertGreaterEqual(len(dataset.sequences), 1)

        # Retrieve EXP01
        seq = dataset.sequences.get("EXP01_comprehensive_workbench")
        self.assertIsNotNone(seq)
        self.assertEqual(len(seq.visits), 13)

        # Check coverage of all 8 physical scenarios across the 13 visits
        expected_events = [v.expected_event for v in seq.visits]
        self.assertIn("baseline_established", expected_events)
        self.assertIn("baseline_stable", expected_events)
        self.assertIn("lighting_change", expected_events)
        self.assertIn("viewpoint_jitter", expected_events)
        self.assertIn("object_moved", expected_events)
        self.assertIn("movement_persists", expected_events)
        self.assertIn("return_to_baseline", expected_events)
        self.assertIn("partial_occlusion", expected_events)
        self.assertIn("revealed", expected_events)
        self.assertIn("object_pending_removal", expected_events)
        self.assertIn("object_removed", expected_events)
        self.assertIn("reappearance", expected_events)

        # Verify target objects from manifest_data
        target_objs = dataset.manifest_data.get("target_objects", [])
        target_ids = {obj.get("object_id") for obj in target_objs}
        self.assertIn("cup_01", target_ids)
        self.assertIn("bottle_01", target_ids)
        self.assertIn("scissors_01", target_ids)
        self.assertIn("phone_01", target_ids)
        self.assertIn("book_01", target_ids)


if __name__ == "__main__":
    unittest.main()
