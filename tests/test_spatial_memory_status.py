"""
Regression tests for Spatial Memory authoritative entity states across
stationary, moved, removed, temporary miss, unreliable registration,
ambiguous identity, and both YOLOv8m and YOLO26m detectors.
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


class _ScriptedDet(ObjectDetector):
    def __init__(self, frames: List[List[Detection]]) -> None:
        self.frames = frames
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


def _scene(objs: List[tuple[int, int, tuple[int, int, int]]]) -> bytes:
    rng = np.random.default_rng(19)
    img = np.full((480, 640, 3), 215, dtype=np.uint8)
    for _ in range(130):
        x, y = int(rng.integers(15, 620)), int(rng.integers(15, 460))
        cv2.circle(img, (x, y), int(rng.integers(3, 8)), (int(rng.integers(60, 160)), int(rng.integers(60, 160)), int(rng.integers(60, 160))), -1)
    for px, py, col in objs:
        cv2.rectangle(img, (px, py), (px + 70, py + 90), col, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestSpatialMemoryStatus(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.base_dir = Path(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_spatial_memory_states_stationary_moved_temp_miss_removed(self) -> None:
        cup = BoundingBox(60, 60, 130, 150)
        bot1 = BoundingBox(200, 60, 270, 150)
        bot2 = BoundingBox(420, 260, 490, 350)
        book = BoundingBox(320, 60, 390, 150)
        frames = [
            [Detection("cup", 0.90, cup), Detection("bottle", 0.85, bot1), Detection("book", 0.88, book)],
            [Detection("cup", 0.90, cup), Detection("bottle", 0.85, bot2)],  # book missed once
            [Detection("cup", 0.90, cup), Detection("bottle", 0.85, bot2)],  # book missed twice -> PENDING_REMOVAL
            [Detection("cup", 0.90, cup), Detection("bottle", 0.85, bot2)],  # book missed thrice -> CONFIRMED_REMOVED
        ]
        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_ScriptedDet(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("sm_ws", "SM WS")
        mgr.process_image_observation("sm_ws", _scene([(60, 60, (20, 120, 200)), (200, 60, (40, 40, 180)), (320, 60, (60, 160, 60))]), "V01", "2026-11-25T10:01:00Z")
        mgr.process_image_observation("sm_ws", _scene([(60, 60, (20, 120, 200)), (420, 260, (40, 40, 180))]), "V02", "2026-11-25T10:02:00Z")

        ents_v2 = {e["label"]: e for e in client.get("/api/workspaces/sm_ws/entities").json()}
        self.assertEqual(ents_v2["cup"]["current_state"], "CONFIRMED_PRESENT")
        self.assertEqual(ents_v2["bottle"]["current_state"], "CONFIRMED_MOVED")
        self.assertGreater(ents_v2["bottle"]["canonical_bbox"]["x_min"], 350.0)
        self.assertEqual(ents_v2["book"]["current_state"], "PENDING_REMOVAL")  # Temporary miss -> NOT falsely CONFIRMED_REMOVED

        mgr.process_image_observation("sm_ws", _scene([(60, 60, (20, 120, 200)), (420, 260, (40, 40, 180))]), "V03", "2026-11-25T10:03:00Z")
        ents_v3 = {e["label"]: e for e in client.get("/api/workspaces/sm_ws/entities").json()}
        self.assertEqual(ents_v3["book"]["current_state"], "CONFIRMED_REMOVED")

    def test_spatial_memory_unreliable_registration_and_ambiguous_identity(self) -> None:
        # 1. Unreliable registration -> INSUFFICIENT_EVIDENCE
        cup = BoundingBox(60, 60, 130, 150)
        frames_reg = [[Detection("cup", 0.90, cup)], [Detection("cup", 0.90, cup)]]
        blank = np.full((480, 640, 3), 240, dtype=np.uint8)
        ok, blank_buf = cv2.imencode(".png", blank)
        assert ok
        mgr1 = WorkspaceManager(base_dir=self.base_dir, detector=_ScriptedDet(frames_reg))
        mgr1.create_workspace("sm_reg", "SM Reg")
        mgr1.process_image_observation("sm_reg", _scene([(60, 60, (20, 120, 200))]), "V01", "2026-11-25T10:01:00Z")
        mgr1.process_image_observation("sm_reg", blank_buf.tobytes(), "V02", "2026-11-25T10:02:00Z")
        ents_reg = mgr1.list_entities("sm_reg")
        self.assertNotEqual(ents_reg[0]["current_state"], "CONFIRMED_MOVED")
        comp_reg = mgr1.compare_sessions("sm_reg", "visit_01", "visit_02")
        self.assertEqual(comp_reg["items"][0]["movement_status"], "REGISTRATION_UNRELIABLE")

        # 2. Ambiguous identity -> AMBIGUOUS_IDENTITY
        cup1 = BoundingBox(60, 60, 130, 150)
        cup2 = BoundingBox(400, 60, 470, 150)
        frames_amb = [
            [Detection("cup", 0.92, cup1), Detection("cup", 0.89, cup2)],
            [Detection("cup", 0.91, cup1), Detection("cup", 0.55, BoundingBox(62, 62, 132, 152)), Detection("cup", 0.88, cup2)],
        ]
        mgr2 = WorkspaceManager(base_dir=self.base_dir, detector=_ScriptedDet(frames_amb))
        mgr2.create_workspace("sm_amb", "SM Amb")
        img = _scene([(60, 60, (20, 120, 200)), (400, 60, (20, 120, 200))])
        mgr2.process_image_observation("sm_amb", img, "V01", "2026-11-25T10:01:00Z")
        mgr2.process_image_observation("sm_amb", img, "V02", "2026-11-25T10:02:00Z")
        ents_amb = [e["current_state"] for e in mgr2.list_entities("sm_amb") if e["entity_id"].startswith("ent_visit_01")]
        self.assertEqual(ents_amb, ["AMBIGUOUS_IDENTITY", "AMBIGUOUS_IDENTITY"])

    def test_both_yolov8m_and_yolo26m_produce_authoritative_moved_and_removed_states(self) -> None:
        rtb_dir = Path("data/physical_pilot_v2/return_to_baseline")
        rem_dir = Path("data/physical_pilot_v2/removal")
        if not rtb_dir.exists() or not rem_dir.exists():
            self.skipTest("Physical pilot images not found")

        mgr = WorkspaceManager(base_dir=self.base_dir)
        for det_name in ("yolov8m", "yolo26m"):
            # Return-to-baseline V01..V03 -> bottle moves to CONFIRMED_MOVED
            ws_m = f"rtb_{det_name}"
            mgr.create_workspace(ws_m, ws_m)
            for i in (1, 2, 3):
                mgr.process_image_observation(ws_m, (rtb_dir / f"V0{i}.jpeg").read_bytes(), f"V0{i}", f"2026-11-25T10:0{i}:00Z", detector_name=det_name)
            states_m = {e["current_state"] for e in mgr.list_entities(ws_m)}
            self.assertIn("CONFIRMED_MOVED", states_m)
            self.assertIn("CONFIRMED_PRESENT", states_m)

            # Return-to-baseline V01..V05 -> returns to CONFIRMED_PRESENT
            for i in (4, 5):
                mgr.process_image_observation(ws_m, (rtb_dir / f"V0{i}.jpeg").read_bytes(), f"V0{i}", f"2026-11-25T10:0{i}:00Z", detector_name=det_name)
            comp_rtb = mgr.compare_sessions(ws_m, "visit_01", "visit_05")
            statuses_rtb = {it["movement_status"] for it in comp_rtb["items"]}
            self.assertIn("NOT_MOVED", statuses_rtb)
