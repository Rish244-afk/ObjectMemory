"""
Comprehensive test suite for ObjectMemory FastAPI backend and WorkspaceManager.

Tests cover:
1. Workspace lifecycle (creation, duplicate rejection, listing, summary, deletion).
2. Workspace isolation (ensuring independent visual memory and zero cross-talk).
3. Image upload and temporal memory execution (visit 1 baseline, visit 2 registration and updates).
4. Durability across server restarts (persisting SQLite and restoring reference frame).
5. History and entity retrieval (filtering by state, entity details, session details, event log).
6. Multi-visit comparison and diagnostics endpoints.
7. Robust error handling (404 on missing resources, 400 on corrupted images, 400 on out-of-order timestamps, 413 on oversized files).
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.api.app import create_app
from objectmemory.api.workspace_manager import WorkspaceManager
from objectmemory.detection.detector import OpenCVDNNDetector


def _create_synthetic_test_image(text: str = "WORKBENCH") -> bytes:
    """Generates a valid 640x480 test image as encoded PNG bytes."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    # Add high-contrast features for planar registration
    cv2.rectangle(img, (50, 50), (250, 200), (40, 40, 180), -1)
    cv2.rectangle(img, (350, 80), (550, 300), (30, 160, 30), -1)
    cv2.circle(img, (200, 350), 60, (200, 50, 50), -1)
    cv2.putText(img, text, (50, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    success, buf = cv2.imencode(".png", img)
    assert success
    return buf.tobytes()


class TestWorkspaceManager(unittest.TestCase):
    """Direct unit tests for WorkspaceManager."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_om_mgr_")
        self.weights_path = Path("models/yolov8n.onnx")
        self.manager = WorkspaceManager(
            base_dir=Path(self.temp_dir),
            weights_path=self.weights_path,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_workspace_crud(self) -> None:
        meta = self.manager.create_workspace("bench_a", "Workbench A", "Primary test bench")
        self.assertEqual(meta["workspace_id"], "bench_a")
        self.assertEqual(meta["name"], "Workbench A")

        # Duplicate rejection
        with self.assertRaises(ValueError):
            self.manager.create_workspace("bench_a", "Duplicate A")

        # Listing
        workspaces = self.manager.list_workspaces()
        self.assertEqual(len(workspaces), 1)
        self.assertEqual(workspaces[0]["workspace_id"], "bench_a")

        # Summary
        summary = self.manager.get_workspace_summary("bench_a")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["total_sessions"], 0)
        self.assertEqual(summary["total_entities"], 0)

        # Deletion
        deleted = self.manager.delete_workspace("bench_a")
        self.assertTrue(deleted)
        self.assertIsNone(self.manager.get_workspace_meta("bench_a"))
        self.assertEqual(len(self.manager.list_workspaces()), 0)

    def test_workspace_isolation(self) -> None:
        self.manager.create_workspace("ws_one", "Workspace One")
        self.manager.create_workspace("ws_two", "Workspace Two")

        img_bytes = _create_synthetic_test_image("ISOLATION")
        # Submit observation to ws_one
        res1 = self.manager.process_image_observation(
            workspace_id="ws_one",
            image_bytes=img_bytes,
            session_name="Visit 01",
            timestamp="2026-10-20T10:00:00Z",
        )
        self.assertEqual(res1["session_id"], "visit_01")
        self.assertTrue(res1["is_reference_frame"])

        # Check ws_one summary has 1 session
        sum_one = self.manager.get_workspace_summary("ws_one")
        self.assertEqual(sum_one["total_sessions"], 1)

        # Check ws_two remains completely clean (zero sessions, zero entities)
        sum_two = self.manager.get_workspace_summary("ws_two")
        self.assertEqual(sum_two["total_sessions"], 0)
        self.assertEqual(sum_two["total_entities"], 0)
        self.assertEqual(sum_two["total_events"], 0)

    def test_durability_across_manager_restart(self) -> None:
        self.manager.create_workspace("durable_bench", "Durable Bench")
        img_bytes = _create_synthetic_test_image("DURABILITY_1")

        # Visit 1
        res1 = self.manager.process_image_observation(
            workspace_id="durable_bench",
            image_bytes=img_bytes,
            session_name="V1",
            timestamp="2026-10-20T10:00:00Z",
        )
        self.assertTrue(res1["is_reference_frame"])

        # Simulate process shutdown and reload: create a brand new WorkspaceManager on the same path
        new_manager = WorkspaceManager(
            base_dir=Path(self.temp_dir),
            weights_path=self.weights_path,
        )

        # Check metadata and summary survive restart
        summary = new_manager.get_workspace_summary("durable_bench")
        self.assertEqual(summary["total_sessions"], 1)
        self.assertEqual(summary["reference_session_id"], "visit_01")

        # Submit Visit 2 using new manager
        img_bytes_2 = _create_synthetic_test_image("DURABILITY_2")
        res2 = new_manager.process_image_observation(
            workspace_id="durable_bench",
            image_bytes=img_bytes_2,
            session_name="V2",
            timestamp="2026-10-20T11:00:00Z",
        )
        self.assertEqual(res2["session_id"], "visit_02")
        self.assertFalse(res2["is_reference_frame"])
        # Reference frame was successfully recovered from disk
        runner = new_manager._get_or_create_runner("durable_bench")
        self.assertIsNotNone(runner.ref_image)
        self.assertEqual(runner.ref_visit_id, "visit_01")
        self.assertEqual(runner.processed_visits_count, 2)


class TestFastAPIBackend(unittest.TestCase):
    """End-to-end integration tests using FastAPI TestClient."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_om_api_")
        self.manager = WorkspaceManager(
            base_dir=Path(self.temp_dir),
            weights_path=Path("models/yolov8n.onnx"),
        )
        self.app = create_app(manager=self.manager)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_health_check(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertTrue(data["detector_available"])
        self.assertEqual(data["workspaces_count"], 0)

    def test_workspace_api_lifecycle(self) -> None:
        # Create
        create_resp = self.client.post(
            "/api/workspaces",
            json={
                "workspace_id": "lab_01",
                "name": "Electronics Bench",
                "description": "Hardware diagnostics area",
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        data = create_resp.json()
        self.assertEqual(data["workspace"]["workspace_id"], "lab_01")
        self.assertEqual(data["workspace"]["name"], "Electronics Bench")

        # Duplicate fails
        dup_resp = self.client.post(
            "/api/workspaces",
            json={"workspace_id": "lab_01", "name": "Duplicate"},
        )
        self.assertEqual(dup_resp.status_code, 400)

        # Get
        get_resp = self.client.get("/api/workspaces/lab_01")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["workspace"]["workspace_id"], "lab_01")

        # List
        list_resp = self.client.get("/api/workspaces")
        self.assertEqual(list_resp.status_code, 200)
        self.assertEqual(len(list_resp.json()), 1)

        # Delete
        del_resp = self.client.delete("/api/workspaces/lab_01")
        self.assertEqual(del_resp.status_code, 200)

        # Get after delete -> 404
        get_resp404 = self.client.get("/api/workspaces/lab_01")
        self.assertEqual(get_resp404.status_code, 404)

    def test_image_upload_and_observation_pipeline(self) -> None:
        # 1. Create workspace
        self.client.post(
            "/api/workspaces",
            json={"workspace_id": "obs_bench", "name": "Observation Bench"},
        )

        # 2. Upload visit 1 (using synthetic pilot visit_01.png)
        pilot_v1 = Path("data/real_workbench/visits/visit_01.png")
        self.assertTrue(pilot_v1.exists(), "Pilot visit_01.png must exist for end-to-end testing")
        with open(pilot_v1, "rb") as f:
            v1_bytes = f.read()

        resp_v1 = self.client.post(
            "/api/workspaces/obs_bench/observations",
            files={"image": ("visit_01.png", v1_bytes, "image/png")},
            data={
                "session_name": "Pilot Baseline",
                "timestamp": "2026-10-20T10:00:00Z",
                "notes": "Initial calibration visit",
                "confidence_threshold": 0.35,
            },
        )
        self.assertEqual(resp_v1.status_code, 201)
        d1 = resp_v1.json()
        self.assertEqual(d1["session_id"], "visit_01")
        self.assertTrue(d1["is_reference_frame"])
        self.assertEqual(d1["registration"]["status"], "SUCCESS")
        self.assertGreater(d1["total_active_entities"], 0)
        self.assertIn("visit_01.png", d1["image_url"])

        # 3. Upload visit 2 (using synthetic pilot visit_02.png)
        pilot_v2 = Path("data/real_workbench/visits/visit_02.png")
        with open(pilot_v2, "rb") as f:
            v2_bytes = f.read()

        resp_v2 = self.client.post(
            "/api/workspaces/obs_bench/observations",
            files={"image": ("visit_02.png", v2_bytes, "image/png")},
            data={
                "session_name": "Pilot Visit 2",
                "timestamp": "2026-10-20T11:00:00Z",
                "notes": "Follow-up observation",
                "confidence_threshold": 0.35,
            },
        )
        self.assertEqual(resp_v2.status_code, 201)
        d2 = resp_v2.json()
        self.assertEqual(d2["session_id"], "visit_02")
        self.assertFalse(d2["is_reference_frame"])
        self.assertEqual(d2["registration"]["status"], "SUCCESS")
        self.assertGreater(d2["registration"]["inlier_ratio"], 0.20)
        self.assertGreater(len(d2["belief_updates"]), 0)

        # 4. Verify Image retrieval endpoint
        img_resp = self.client.get(d1["image_url"])
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.headers["content-type"], "image/png")
        self.assertGreater(len(img_resp.content), 1000)

        # 5. Verify Sessions list endpoint
        sess_resp = self.client.get("/api/workspaces/obs_bench/observations")
        self.assertEqual(sess_resp.status_code, 200)
        sessions = sess_resp.json()
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["session_id"], "visit_01")
        self.assertEqual(sessions[1]["session_id"], "visit_02")

        # 6. Verify Session detail endpoint
        v1_detail = self.client.get("/api/workspaces/obs_bench/observations/visit_01")
        self.assertEqual(v1_detail.status_code, 200)
        det_data = v1_detail.json()
        self.assertEqual(det_data["session"]["session_id"], "visit_01")
        self.assertGreater(len(det_data["observations"]), 0)

        # 7. Verify Entities list endpoint
        ent_resp = self.client.get("/api/workspaces/obs_bench/entities")
        self.assertEqual(ent_resp.status_code, 200)
        entities = ent_resp.json()
        self.assertGreater(len(entities), 0)
        first_entity_id = entities[0]["entity_id"]

        # 8. Verify Entity history endpoint
        ent_detail = self.client.get(f"/api/workspaces/obs_bench/entities/{first_entity_id}")
        self.assertEqual(ent_detail.status_code, 200)
        e_data = ent_detail.json()
        self.assertEqual(e_data["entity"]["entity_id"], first_entity_id)
        self.assertGreater(len(e_data["observations"]), 0)

        # 9. Verify Change events endpoint
        ev_resp = self.client.get("/api/workspaces/obs_bench/events")
        self.assertEqual(ev_resp.status_code, 200)
        self.assertIsInstance(ev_resp.json(), list)

        # 10. Verify Comparison endpoint
        comp_resp = self.client.get("/api/workspaces/obs_bench/compare")
        self.assertEqual(comp_resp.status_code, 200)
        comp_data = comp_resp.json()
        self.assertEqual(comp_data["workspace_id"], "obs_bench")
        self.assertEqual(comp_data["reference_session_id"], "visit_01")
        self.assertEqual(comp_data["query_session_id"], "visit_02")
        self.assertGreater(len(comp_data["items"]), 0)

        # 11. Verify Diagnostics endpoint
        diag_resp = self.client.get("/api/workspaces/obs_bench/diagnostics")
        self.assertEqual(diag_resp.status_code, 200)
        diag = diag_resp.json()
        self.assertEqual(diag["total_sessions"], 2)
        self.assertIn("registration_health", diag)
        self.assertEqual(diag["registration_health"]["status"], "HEALTHY")

    def test_validation_and_error_handling(self) -> None:
        self.client.post(
            "/api/workspaces",
            json={"workspace_id": "err_bench", "name": "Error Bench"},
        )

        # 1. Non-existent workspace
        r_404 = self.client.get("/api/workspaces/non_existent_workspace")
        self.assertEqual(r_404.status_code, 404)

        # 2. Empty upload file
        r_empty = self.client.post(
            "/api/workspaces/err_bench/observations",
            files={"image": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(r_empty.status_code, 400)
        self.assertIn("empty", r_empty.json()["detail"].lower())

        # 3. Invalid image data (corrupted / text)
        r_bad = self.client.post(
            "/api/workspaces/err_bench/observations",
            files={"image": ("bad.png", b"not_an_image_file", "image/png")},
        )
        self.assertEqual(r_bad.status_code, 400)
        self.assertIn("decode", r_bad.json()["detail"].lower())

        # 4. Out-of-order timestamp rejection
        valid_img = _create_synthetic_test_image("T1")
        # Submit first visit at 2026-10-20T12:00:00Z
        r1 = self.client.post(
            "/api/workspaces/err_bench/observations",
            files={"image": ("v1.png", valid_img, "image/png")},
            data={"timestamp": "2026-10-20T12:00:00Z"},
        )
        self.assertEqual(r1.status_code, 201)

        # Submit second visit with earlier timestamp 2026-10-20T10:00:00Z
        r2 = self.client.post(
            "/api/workspaces/err_bench/observations",
            files={"image": ("v2.png", valid_img, "image/png")},
            data={"timestamp": "2026-10-20T10:00:00Z"},
        )
        self.assertEqual(r2.status_code, 400)
        self.assertIn("earlier", r2.json()["detail"].lower())

        # 5. Oversized payload rejection (> 25MB)
        huge_bytes = b"0" * (25 * 1024 * 1024 + 1024)
        r_huge = self.client.post(
            "/api/workspaces/err_bench/observations",
            files={"image": ("huge.png", huge_bytes, "image/png")},
        )
        self.assertEqual(r_huge.status_code, 413)

        # 6. Non-existent entity and session detail 404s
        r_no_ent = self.client.get("/api/workspaces/err_bench/entities/nonexistent_ent")
        self.assertEqual(r_no_ent.status_code, 404)

        r_no_sess = self.client.get("/api/workspaces/err_bench/observations/nonexistent_sess")
        self.assertEqual(r_no_sess.status_code, 404)

        # 7. Comparison with only 1 session -> 400
        r_comp_single = self.client.get("/api/workspaces/err_bench/compare")
        self.assertEqual(r_comp_single.status_code, 400)

        # 8. Entity filter by invalid state -> 400
        r_bad_filter = self.client.get("/api/workspaces/err_bench/entities?state=INVALID_STATE")
        self.assertEqual(r_bad_filter.status_code, 400)

    def test_get_detectors_endpoint(self) -> None:
        """Verifies GET /api/detectors returns registered detector list, defaults, and metadata."""
        resp = self.client.get("/api/detectors")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("default_detector", data)
        self.assertEqual(data["default_detector"], "yolov8m")
        self.assertIn("detectors", data)
        self.assertIsInstance(data["detectors"], list)

        detector_map = {d["name"]: d for d in data["detectors"]}
        self.assertIn("yolov8m", detector_map)
        self.assertIn("yolov8n", detector_map)
        self.assertIn("mock", detector_map)
        self.assertIn("yolo-world", detector_map)
        self.assertTrue(detector_map["yolov8m"]["is_default"])
        self.assertFalse(detector_map["yolov8n"]["is_default"])
        self.assertTrue(detector_map["yolo-world"]["is_open_vocabulary"])
        self.assertFalse(detector_map["yolov8m"]["is_open_vocabulary"])
        self.assertFalse(detector_map["yolov8n"]["is_open_vocabulary"])

    def test_observation_submission_with_yolo_world_and_prompts(self) -> None:
        """Verifies submitting an observation with YOLO-World and dynamic open-vocabulary prompts."""
        ws_resp = self.client.post(
            "/api/workspaces",
            json={
                "workspace_id": "yolo_world_ws",
                "name": "YOLO-World Test WS",
            },
        )
        self.assertEqual(ws_resp.status_code, 201)
        ws_id = ws_resp.json()["workspace"]["workspace_id"]

        img_path = Path("data/real_workbench/visits/visit_01.png")
        if not img_path.exists():
            self.skipTest("Visit 1 image not found.")

        with open(img_path, "rb") as f:
            resp = self.client.post(
                f"/api/workspaces/{ws_id}/observations",
                files={"image": ("visit_01.png", f, "image/png")},
                data={
                    "session_name": "Visit 01 OpenVocab",
                    "confidence_threshold": 0.10,
                    "detector_name": "yolo-world",
                    "prompts": "book, notebook, bottle, drawing compass",
                },
            )

        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["session_id"], "visit_01")
        self.assertGreater(len(data["detections"]), 0)
        labels = [d["label"] for d in data["detections"]]
        for label in labels:
            self.assertIn(label, ["book", "notebook", "bottle", "drawing compass"])


if __name__ == "__main__":
    unittest.main()
