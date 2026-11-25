"""
Comprehensive regression tests for explicit MOVED / NOT_MOVED / uncertainty
comparison reporting across backend API and temporal pipeline.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi.testclient import TestClient

from objectmemory.api.app import create_app
from objectmemory.api.workspace_manager import WorkspaceManager
from objectmemory.detection.detector import ObjectDetector
from objectmemory.detection.models import BoundingBox, Detection, DetectionBatch


class _ScriptedDetector(ObjectDetector):
    def __init__(self, frames_by_call: List[List[Detection]]) -> None:
        self.frames = frames_by_call
        self.idx = -1

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.25, **kwargs: Any) -> DetectionBatch:
        if confidence_threshold >= 0.20:
            self.idx += 1
        cur = min(max(self.idx, 0), len(self.frames) - 1)
        filtered = [d for d in self.frames[cur] if d.confidence >= confidence_threshold]
        h, w = image.shape[:2]
        return DetectionBatch(detections=filtered, image_size=(w, h), latency_ms=1.0, detector_name="scripted")

    def is_available(self) -> bool:
        return True

    def model_info(self) -> Dict[str, Any]:
        return {"name": "scripted", "architecture": "mock"}


def _textured_img(objs: List[tuple[int, int, tuple[int, int, int]]]) -> bytes:
    rng = np.random.default_rng(11)
    img = np.full((480, 640, 3), 215, dtype=np.uint8)
    for _ in range(130):
        x, y = int(rng.integers(15, 620)), int(rng.integers(15, 460))
        cv2.circle(img, (x, y), int(rng.integers(3, 8)), (int(rng.integers(60, 160)), int(rng.integers(60, 160)), int(rng.integers(60, 160))), -1)
    for px, py, col in objs:
        cv2.rectangle(img, (px, py), (px + 70, py + 90), col, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestMovementReporting(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.base_dir = Path(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stationary_and_moved_and_stayed_reporting(self) -> None:
        cup_box = BoundingBox(60, 60, 130, 150)
        bot_v1 = BoundingBox(200, 60, 270, 150)
        bot_v2 = BoundingBox(420, 260, 490, 350)
        frames = [
            [Detection("cup", 0.90, cup_box), Detection("bottle", 0.85, bot_v1)],
            [Detection("cup", 0.91, cup_box), Detection("bottle", 0.84, bot_v2)],
            [Detection("cup", 0.90, cup_box), Detection("bottle", 0.86, bot_v2)],
        ]
        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_ScriptedDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("mv_ws", "Movement WS")
        mgr.process_image_observation("mv_ws", _textured_img([(60, 60, (20, 120, 200)), (200, 60, (40, 40, 180))]), "V01", "2026-11-25T10:01:00Z")
        mgr.process_image_observation("mv_ws", _textured_img([(60, 60, (20, 120, 200)), (420, 260, (40, 40, 180))]), "V02", "2026-11-25T10:02:00Z")
        mgr.process_image_observation("mv_ws", _textured_img([(60, 60, (20, 120, 200)), (420, 260, (40, 40, 180))]), "V03", "2026-11-25T10:03:00Z")

        comp13 = client.get("/api/workspaces/mv_ws/compare?reference_session_id=visit_01&query_session_id=visit_03").json()
        cup_it = next(i for i in comp13["items"] if i["label"] == "cup")
        bot_it = next(i for i in comp13["items"] if i["label"] == "bottle")
        self.assertEqual(cup_it["movement_status"], "NOT_MOVED")
        self.assertIsNotNone(cup_it["displacement_px"])
        self.assertLess(cup_it["displacement_px"], 15.0)
        self.assertIsNotNone(cup_it["ref_bbox"])
        self.assertIsNotNone(cup_it["query_bbox"])

        self.assertEqual(bot_it["movement_status"], "MOVED")
        self.assertEqual(bot_it["status_change"], "CONFIRMED_MOVED")
        self.assertGreater(bot_it["displacement_px"], 200.0)

    def test_temporary_miss_not_removed_and_persistent_absence_removed(self) -> None:
        cup_box = BoundingBox(60, 60, 130, 150)
        bot_box = BoundingBox(200, 60, 270, 150)
        frames = [
            [Detection("cup", 0.90, cup_box), Detection("bottle", 0.85, bot_box)],
            [Detection("cup", 0.90, cup_box)],
            [Detection("cup", 0.90, cup_box)],
            [Detection("cup", 0.90, cup_box)],
        ]
        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_ScriptedDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("abs_ws", "Absence WS")
        mgr.process_image_observation("abs_ws", _textured_img([(60, 60, (20, 120, 200)), (200, 60, (40, 40, 180))]), "V01", "2026-11-25T10:01:00Z")
        for i in (2, 3, 4):
            mgr.process_image_observation("abs_ws", _textured_img([(60, 60, (20, 120, 200))]), f"V0{i}", f"2026-11-25T10:0{i}:00Z")

        # After 1 miss (V01 -> V02): NOT_DETECTED, NOT REMOVED
        comp12 = client.get("/api/workspaces/abs_ws/compare?reference_session_id=visit_01&query_session_id=visit_02").json()
        bot_12 = next(i for i in comp12["items"] if i["label"] == "bottle")
        self.assertEqual(bot_12["movement_status"], "NOT_DETECTED")
        self.assertNotEqual(bot_12["movement_status"], "REMOVED")

        # After 3 misses (V01 -> V04): REMOVED
        comp14 = client.get("/api/workspaces/abs_ws/compare?reference_session_id=visit_01&query_session_id=visit_04").json()
        bot_14 = next(i for i in comp14["items"] if i["label"] == "bottle")
        self.assertEqual(bot_14["movement_status"], "REMOVED")
        self.assertEqual(bot_14["status_change"], "CONFIRMED_REMOVED")

    def test_registration_failure_reports_registration_unreliable(self) -> None:
        cup_box = BoundingBox(60, 60, 130, 150)
        frames = [
            [Detection("cup", 0.90, cup_box)],
            [Detection("cup", 0.90, cup_box)],
        ]
        blank = np.full((480, 640, 3), 240, dtype=np.uint8)
        ok, blank_buf = cv2.imencode(".png", blank)
        assert ok

        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_ScriptedDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("reg_ws", "Reg Fail WS")
        mgr.process_image_observation("reg_ws", _textured_img([(60, 60, (20, 120, 200))]), "V01", "2026-11-25T10:01:00Z")
        mgr.process_image_observation("reg_ws", blank_buf.tobytes(), "V02", "2026-11-25T10:02:00Z")

        comp = client.get("/api/workspaces/reg_ws/compare?reference_session_id=visit_01&query_session_id=visit_02").json()
        self.assertEqual(comp["registration_status"], "DEGRADED")
        cup_it = next(i for i in comp["items"] if i["label"] == "cup")
        self.assertEqual(cup_it["movement_status"], "REGISTRATION_UNRELIABLE")
        self.assertIsNone(cup_it["displacement_px"])

    def test_duplicate_detections_and_multiple_similar_objects(self) -> None:
        # Two distinct cups + a duplicate lower-confidence box near cup 1
        cup1 = BoundingBox(60, 60, 130, 150)
        cup2 = BoundingBox(400, 60, 470, 150)
        frames = [
            [Detection("cup", 0.92, cup1), Detection("cup", 0.89, cup2)],
            [Detection("cup", 0.91, cup1), Detection("cup", 0.55, BoundingBox(62, 62, 132, 152)), Detection("cup", 0.88, cup2)],
        ]
        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_ScriptedDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("dup_ws", "Dup WS")
        img = _textured_img([(60, 60, (20, 120, 200)), (400, 60, (20, 120, 200))])
        mgr.process_image_observation("dup_ws", img, "V01", "2026-11-25T10:01:00Z")
        mgr.process_image_observation("dup_ws", img, "V02", "2026-11-25T10:02:00Z")

        comp = client.get("/api/workspaces/dup_ws/compare?reference_session_id=visit_01&query_session_id=visit_02").json()
        ambig = [i for i in comp["items"] if i["label"] == "cup" and i["movement_status"] == "IDENTITY_AMBIGUOUS"]
        self.assertEqual(len(ambig), 2)
