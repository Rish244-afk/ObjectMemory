"""
Comprehensive test suite for Persistent User-Confirmed Labels, Identity Continuity,
Evidence-Gated Sub-Threshold Recovery, and Human-Assisted Location Confirmation.
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
from objectmemory.domain.states import EpistemicState


class _SequenceMockDetector(ObjectDetector):
    """Deterministic detector returning scripted detections by threshold & call index."""

    def __init__(self, frames_by_call: List[Dict[str, List[Detection]]]) -> None:
        self.frames = frames_by_call
        self.call_idx = -1

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.25, **kwargs: Any) -> DetectionBatch:
        if confidence_threshold >= 0.20:
            self.call_idx += 1
        idx = min(max(self.call_idx, 0), len(self.frames) - 1)
        all_dets = self.frames[idx]["dets"]
        filtered = [d for d in all_dets if d.confidence >= confidence_threshold]
        h, w = image.shape[:2]
        return DetectionBatch(detections=filtered, image_size=(w, h), latency_ms=1.0, detector_name="seq_mock")

    def is_available(self) -> bool:
        return True

    def get_availability_error(self):
        return None

    def model_info(self):
        return {"name": "seq_mock", "architecture": "mock"}


def _textured_scene_with_patch(patch_x: int, patch_y: int, include_object: bool = True) -> bytes:
    rng = np.random.default_rng(7)
    img = np.full((480, 640, 3), 210, dtype=np.uint8)
    # Rich background features for reliable SIFT registration
    for _ in range(120):
        x, y = int(rng.integers(15, 620)), int(rng.integers(15, 460))
        cv2.circle(img, (x, y), int(rng.integers(3, 9)), (int(rng.integers(60, 160)), int(rng.integers(60, 160)), int(rng.integers(60, 160))), -1)
    # Distinct second object (stationary cup)
    cv2.rectangle(img, (60, 60), (140, 140), (20, 120, 200), -1)
    if include_object:
        # Distinct dark bottle-like object with high-contrast texture
        cv2.rectangle(img, (patch_x, patch_y), (patch_x + 70, patch_y + 120), (30, 30, 50), -1)
        cv2.rectangle(img, (patch_x + 15, patch_y + 10), (patch_x + 55, patch_y + 50), (180, 180, 220), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestIdentityAndRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir.name, ignore_errors=True)

    def test_semantic_label_lifecycle_and_restart_persistence(self) -> None:
        frames = [
            {"dets": [
                Detection(label="hair drier", confidence=0.50, raw_bbox=BoundingBox(400, 40, 470, 160)),
                Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140)),
            ]},
            {"dets": [
                Detection(label="bottle", confidence=0.45, raw_bbox=BoundingBox(402, 42, 472, 162)),
                Detection(label="cup", confidence=0.91, raw_bbox=BoundingBox(60, 60, 140, 140)),
            ]},
        ]
        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_SequenceMockDetector(frames))
        app = create_app(manager=mgr)
        client = TestClient(app)
        mgr.create_workspace("lbl_ws", "Label WS")

        img1 = _textured_scene_with_patch(400, 40, True)
        mgr.process_image_observation("lbl_ws", img1, session_name="V01", timestamp="2026-11-25T10:01:00Z")

        ents = client.get("/api/workspaces/lbl_ws/entities").json()
        hd = next(e for e in ents if e["raw_label"] == "hair drier")
        cup = next(e for e in ents if e["raw_label"] == "cup")

        # 1. Confirm hair drier -> bottle
        r_conf = client.patch(f"/api/workspaces/lbl_ws/entities/{hd['entity_id']}/label", json={"label": "bottle"})
        self.assertEqual(r_conf.status_code, 200)
        self.assertEqual(r_conf.json()["label"], "bottle")
        self.assertEqual(r_conf.json()["confirmed_label"], "bottle")
        self.assertEqual(r_conf.json()["raw_label"], "hair drier")
        self.assertEqual(r_conf.json()["label_provenance"], "USER_CONFIRMED")
        self.assertIsNotNone(r_conf.json()["label_confirmed_at"])

        # 2. Independent entity (cup) remains untouched
        cup_after = next(e for e in client.get("/api/workspaces/lbl_ws/entities").json() if e["entity_id"] == cup["entity_id"])
        self.assertIsNone(cup_after["confirmed_label"])
        self.assertEqual(cup_after["label"], "cup")

        # 3. Survives backend restart (new WorkspaceManager + TestClient instance)
        mgr2 = WorkspaceManager(base_dir=self.base_dir, detector=_SequenceMockDetector(frames[1:]))
        client2 = TestClient(create_app(manager=mgr2))
        e_reloaded = client2.get(f"/api/workspaces/lbl_ws/entities/{hd['entity_id']}").json()["entity"]
        self.assertEqual(e_reloaded["confirmed_label"], "bottle")
        self.assertEqual(e_reloaded["raw_label"], "hair drier")
        self.assertEqual(e_reloaded["label_provenance"], "USER_CONFIRMED")

        # 4. Later visit with raw label 'bottle' (or disagreeing label) preserves confirmed identity & raw predictions
        mgr2.process_image_observation("lbl_ws", img1, session_name="V02", timestamp="2026-11-25T10:02:00Z")
        comp = client2.get("/api/workspaces/lbl_ws/compare?reference_session_id=visit_01&query_session_id=visit_02").json()
        hd_comp = next(it for it in comp["items"] if it["entity_id"] == hd["entity_id"])
        self.assertEqual(hd_comp["label"], "bottle")
        self.assertEqual(hd_comp["raw_label"], "hair drier")
        self.assertTrue(hd_comp["class_uncertain"])

        # 5. Revise confirmed label -> water bottle
        r_rev = client2.patch(f"/api/workspaces/lbl_ws/entities/{hd['entity_id']}/label", json={"label": "water bottle"})
        self.assertEqual(r_rev.json()["confirmed_label"], "water bottle")
        self.assertEqual(r_rev.json()["raw_label"], "hair drier")

        # 6. Clear confirmation -> restores original raw label 'hair drier'
        r_clr = client2.delete(f"/api/workspaces/lbl_ws/entities/{hd['entity_id']}/label")
        self.assertEqual(r_clr.status_code, 200)
        self.assertIsNone(r_clr.json()["confirmed_label"])
        self.assertEqual(r_clr.json()["label"], "hair drier")
        self.assertEqual(r_clr.json()["label_provenance"], "MODEL_PREDICTED")

    def test_subthreshold_recovery_and_genuine_removal_preservation(self) -> None:
        # V01: high conf hair drier at (400, 40). V02: sub-threshold (0.17) at moved (400, 260).
        frames = [
            {"dets": [
                Detection(label="hair drier", confidence=0.50, raw_bbox=BoundingBox(400, 40, 470, 160)),
                Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140)),
            ]},
            {"dets": [
                Detection(label="hair drier", confidence=0.17, raw_bbox=BoundingBox(400, 260, 470, 380)),
                Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140)),
            ]},
        ]
        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_SequenceMockDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("rec_ws", "Recovery WS")

        mgr.process_image_observation("rec_ws", _textured_scene_with_patch(400, 40, True), session_name="V01", timestamp="2026-11-25T10:01:00Z")
        mgr.process_image_observation("rec_ws", _textured_scene_with_patch(400, 260, True), session_name="V02", timestamp="2026-11-25T10:02:00Z")

        comp = client.get("/api/workspaces/rec_ws/compare?reference_session_id=visit_01&query_session_id=visit_02").json()
        # Normal raw_detections_count in V02 remains 1 (cup only; sub-threshold was NOT added to raw_detections)
        self.assertEqual(comp["query_detections_count"], 1)
        hd_item = next(it for it in comp["items"] if it["raw_label"] == "hair drier")
        self.assertEqual(hd_item["status_change"], "CONFIRMED_MOVED")
        self.assertEqual(hd_item["evidence_source"], "SUB_THRESHOLD_RECOVERY")
        self.assertGreater(hd_item["displacement_px"], 180.0)

    def test_genuine_removal_with_no_recovery_candidate(self) -> None:
        frames = [
            {"dets": [Detection(label="hair drier", confidence=0.50, raw_bbox=BoundingBox(400, 40, 470, 160)), Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140))]},
            {"dets": [Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140))]},
            {"dets": [Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140))]},
            {"dets": [Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140))]},
        ]
        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_SequenceMockDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("rem_ws", "Removal WS")
        mgr.process_image_observation("rem_ws", _textured_scene_with_patch(400, 40, True), session_name="V01", timestamp="2026-11-25T10:01:00Z")
        for i in (2, 3, 4):
            mgr.process_image_observation("rem_ws", _textured_scene_with_patch(400, 40, False), session_name=f"V0{i}", timestamp=f"2026-11-25T10:0{i}:00Z")

        comp = client.get("/api/workspaces/rem_ws/compare?reference_session_id=visit_01&query_session_id=visit_04").json()
        hd_item = next(it for it in comp["items"] if it["raw_label"] == "hair drier")
        self.assertEqual(hd_item["status_change"], "CONFIRMED_REMOVED")

    def test_human_confirmed_location_persistence_and_provenance(self) -> None:
        frames = [
            {"dets": [Detection(label="hair drier", confidence=0.50, raw_bbox=BoundingBox(400, 40, 470, 160)), Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140))]},
            {"dets": [Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140))]},
        ]
        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_SequenceMockDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("hum_ws", "Human Confirm WS")
        mgr.process_image_observation("hum_ws", _textured_scene_with_patch(400, 40, True), session_name="V01", timestamp="2026-11-25T10:01:00Z")
        mgr.process_image_observation("hum_ws", _textured_scene_with_patch(400, 260, True), session_name="V02", timestamp="2026-11-25T10:02:00Z")

        ents = client.get("/api/workspaces/hum_ws/entities").json()
        hd = next(e for e in ents if e["raw_label"] == "hair drier")

        r_hum = client.post(
            "/api/workspaces/hum_ws/observations/visit_02/confirm-location",
            json={"entity_id": hd["entity_id"], "bbox": {"x_min": 400, "y_min": 260, "x_max": 470, "y_max": 380}, "notes": "Confirmed dark bottle"},
        )
        self.assertEqual(r_hum.status_code, 200)
        hdata = r_hum.json()
        self.assertEqual(hdata["evidence_source"], "HUMAN_CONFIRMED")
        self.assertTrue(hdata["registration_reliable"])
        self.assertEqual(hdata["new_state"], "CONFIRMED_MOVED")
        self.assertGreater(hdata["displacement_px"], 180.0)

        # Survives restart & keeps raw_detections_count = 1
        mgr2 = WorkspaceManager(base_dir=self.base_dir, detector=_SequenceMockDetector(frames))
        client2 = TestClient(create_app(manager=mgr2))
        comp = client2.get("/api/workspaces/hum_ws/compare?reference_session_id=visit_01&query_session_id=visit_02").json()
        self.assertEqual(comp["query_detections_count"], 1)
        hd_comp = next(it for it in comp["items"] if it["entity_id"] == hd["entity_id"])
        self.assertEqual(hd_comp["status_change"], "CONFIRMED_MOVED")
        self.assertEqual(hd_comp["evidence_source"], "HUMAN_CONFIRMED")

    def test_ambiguous_subthreshold_candidates_are_rejected(self) -> None:
        # Two nearly identical sub-threshold candidates in V02 -> margin < 0.05 -> must NOT recover
        frames = [
            {"dets": [
                Detection(label="hair drier", confidence=0.50, raw_bbox=BoundingBox(400, 40, 470, 160)),
                Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140)),
            ]},
            {"dets": [
                Detection(label="hair drier", confidence=0.17, raw_bbox=BoundingBox(250, 260, 320, 380)),
                Detection(label="hair drier", confidence=0.16, raw_bbox=BoundingBox(400, 260, 470, 380)),
                Detection(label="cup", confidence=0.90, raw_bbox=BoundingBox(60, 60, 140, 140)),
            ]},
        ]
        # Build V02 image with two identical patches at (250, 260) and (400, 260)
        rng = np.random.default_rng(7)
        img2 = np.full((480, 640, 3), 210, dtype=np.uint8)
        for _ in range(120):
            x, y = int(rng.integers(15, 620)), int(rng.integers(15, 460))
            cv2.circle(img2, (x, y), int(rng.integers(3, 9)), (int(rng.integers(60, 160)), int(rng.integers(60, 160)), int(rng.integers(60, 160))), -1)
        cv2.rectangle(img2, (60, 60), (140, 140), (20, 120, 200), -1)
        for px, py in ((250, 260), (400, 260)):
            cv2.rectangle(img2, (px, py), (px + 70, py + 120), (30, 30, 50), -1)
            cv2.rectangle(img2, (px + 15, py + 10), (px + 55, py + 50), (180, 180, 220), -1)
        ok, buf = cv2.imencode(".png", img2)
        assert ok

        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_SequenceMockDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("amb_ws", "Ambiguous WS")
        mgr.process_image_observation("amb_ws", _textured_scene_with_patch(400, 40, True), session_name="V01", timestamp="2026-11-25T10:01:00Z")
        mgr.process_image_observation("amb_ws", buf.tobytes(), session_name="V02", timestamp="2026-11-25T10:02:00Z")

        comp = client.get("/api/workspaces/amb_ws/compare?reference_session_id=visit_01&query_session_id=visit_02").json()
        hd_item = next(it for it in comp["items"] if it["raw_label"] == "hair drier")
        self.assertNotEqual(hd_item["status_change"], "CONFIRMED_MOVED")
        self.assertNotEqual(hd_item["evidence_source"], "SUB_THRESHOLD_RECOVERY")

    def test_unreliable_registration_blocks_confirm_location(self) -> None:
        frames = [
            {"dets": [Detection(label="hair drier", confidence=0.50, raw_bbox=BoundingBox(400, 40, 470, 160))]},
            {"dets": []},
        ]
        blank = np.full((480, 640, 3), 240, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", blank)
        assert ok

        mgr = WorkspaceManager(base_dir=self.base_dir, detector=_SequenceMockDetector(frames))
        client = TestClient(create_app(manager=mgr))
        mgr.create_workspace("unrel_ws", "Unreliable WS")
        mgr.process_image_observation("unrel_ws", _textured_scene_with_patch(400, 40, True), session_name="V01", timestamp="2026-11-25T10:01:00Z")
        mgr.process_image_observation("unrel_ws", buf.tobytes(), session_name="V02", timestamp="2026-11-25T10:02:00Z")

        ents = client.get("/api/workspaces/unrel_ws/entities").json()
        hd = ents[0]
        r_hum = client.post(
            "/api/workspaces/unrel_ws/observations/visit_02/confirm-location",
            json={"entity_id": hd["entity_id"], "bbox": {"x_min": 400, "y_min": 260, "x_max": 470, "y_max": 380}},
        )
        self.assertEqual(r_hum.status_code, 200)
        hdata = r_hum.json()
        self.assertFalse(hdata["registration_reliable"])
        self.assertEqual(hdata["new_state"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(hdata["displacement_px"])
        self.assertIsNone(hdata["canonical_bbox"])
        comp = client.get("/api/workspaces/unrel_ws/compare?reference_session_id=visit_01&query_session_id=visit_02").json()
        hd_comp = next(it for it in comp["items"] if it["entity_id"] == hd["entity_id"])
        self.assertNotEqual(hd_comp["status_change"], "CONFIRMED_MOVED")
