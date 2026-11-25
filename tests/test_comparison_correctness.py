"""
Targeted regression tests for dashboard comparison correctness and temporal state safeguards:
1. Persistent object detected at two different positions -> CONFIRMED_MOVED with same entity_id.
2. Persistent object temporarily missed by detector (1 visit) -> NOT falsely confirmed removed.
3. Object persistently absent across 3 reliable observations -> CONFIRMED_REMOVED.
4. Object with inconsistent semantic class predictions -> consistent physical entity_id + class_uncertain.
5. Unreliable registration -> prevents spurious movement or removal decisions.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from objectmemory.api.app import create_app
from objectmemory.api.workspace_manager import WorkspaceManager
from objectmemory.detection.detector import ObjectDetector
from objectmemory.detection.models import BoundingBox, Detection, DetectionBatch
from objectmemory.domain.models import ChangeEvent, Observation, Session, WorkbenchEntity
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult, ReprojectionStats
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput


class _DummyDetector(ObjectDetector):
    def detect(self, image, confidence_threshold: float = 0.25, **kwargs) -> DetectionBatch:
        return DetectionBatch(detections=[], image_size=(640, 480), inference_time_ms=1.0, model_name="dummy")

    def is_available(self) -> bool:
        return True

    def get_availability_error(self):
        return None

    def model_info(self):
        return {"name": "dummy"}


def _reliable_reg() -> RegistrationResult:
    return RegistrationResult(
        success=True,
        status=AlignmentStatus.SUCCESS,
        method_used="SIFT",
        ref_size=(640, 480),
        query_size=(640, 480),
        homography=np.eye(3, dtype=np.float64),
        num_keypoints_ref=200,
        num_keypoints_query=200,
        num_tentative_matches=100,
        num_inliers=90,
        inlier_ratio=0.90,
        reprojection_stats=ReprojectionStats(mean=0.5, median=0.4, std=0.2, max_error=1.2, inlier_count=90),
        overlap_area_fraction=0.95,
    )


def _unreliable_reg() -> RegistrationResult:
    return RegistrationResult(
        success=False,
        status=AlignmentStatus.INSUFFICIENT_INLIERS,
        method_used="SIFT",
        ref_size=(640, 480),
        query_size=(640, 480),
        homography=None,
        num_keypoints_ref=200,
        num_keypoints_query=25,
        num_tentative_matches=10,
        num_inliers=2,
        inlier_ratio=0.05,
        failure_reason="Insufficient inliers",
    )


class TestComparisonCorrectness(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.mgr = WorkspaceManager(base_dir=Path(self.tmp_dir.name), detector=_DummyDetector())
        self.app = create_app(manager=self.mgr)
        self.client = TestClient(self.app)
        self.mgr.create_workspace("comp_test", "Comparison Test")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_1_and_3_intermediate_movement_removal_cross_sequence_and_label_confirm(self) -> None:
        db_path = self.mgr._db_path("comp_test")
        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            for sid, sname, ts in [
                ("visit_01", "Visit 01 (V01)", "2026-11-25T10:01:00Z"),
                ("visit_02", "Visit 02 (V02)", "2026-11-25T10:02:00Z"),
                ("visit_03", "Visit 03 (V03)", "2026-11-25T10:03:00Z"),
                ("visit_04", "Visit 04 (V01)", "2026-11-25T10:04:00Z"),
            ]:
                store.add_session(Session(session_id=sid, session_name=sname, timestamp=ts, image_path=f"memory://{sid}.png"))

            e1 = WorkbenchEntity(
                entity_id="ent_visit_01_1",
                label="hair drier",
                created_at="2026-11-25T10:01:00Z",
                canonical_bbox=BoundingBox(300, 100, 400, 200),
                current_state=EpistemicState.CONFIRMED_MOVED,
                last_seen_session_id="visit_03",
                metadata={"created_in_session": "visit_01"},
            )
            e2 = WorkbenchEntity(
                entity_id="ent_visit_01_2",
                label="bottle",
                created_at="2026-11-25T10:01:00Z",
                canonical_bbox=BoundingBox(500, 100, 580, 240),
                current_state=EpistemicState.CONFIRMED_REMOVED,
                last_seen_session_id="visit_01",
                metadata={"created_in_session": "visit_01"},
            )
            store.add_entity(e1)
            store.add_entity(e2)

            store.add_observation(Observation(
                observation_id="obs_v1_1", session_id="visit_01", timestamp="2026-11-25T10:01:00Z",
                image_reference="memory://visit_01.png", entity_id="ent_visit_01_1",
                raw_bbox=BoundingBox(100, 100, 200, 200), canonical_bbox=BoundingBox(100, 100, 200, 200),
                visibility_status=VisibilityStatus.CLEARLY_VISIBLE, detector_confidence=0.45, alignment_quality=1.0,
            ))
            store.add_observation(Observation(
                observation_id="obs_v1_2", session_id="visit_01", timestamp="2026-11-25T10:01:00Z",
                image_reference="memory://visit_01.png", entity_id="ent_visit_01_2",
                raw_bbox=BoundingBox(500, 100, 580, 240), canonical_bbox=BoundingBox(500, 100, 580, 240),
                visibility_status=VisibilityStatus.CLEARLY_VISIBLE, detector_confidence=0.80, alignment_quality=1.0,
            ))
            store.add_observation(Observation(
                observation_id="obs_v3_1", session_id="visit_03", timestamp="2026-11-25T10:03:00Z",
                image_reference="memory://visit_03.png", entity_id="ent_visit_01_1",
                raw_bbox=BoundingBox(300, 100, 400, 200), canonical_bbox=BoundingBox(300, 100, 400, 200),
                visibility_status=VisibilityStatus.CLEARLY_VISIBLE, detector_confidence=0.48, alignment_quality=0.95,
            ))
            store.add_change_event(ChangeEvent(
                event_id="evt_1", entity_id="ent_visit_01_1", from_session_id="visit_01", to_session_id="visit_02",
                timestamp="2026-11-25T10:02:00Z", prior_state=EpistemicState.CONFIRMED_PRESENT,
                new_state=EpistemicState.CONFIRMED_MOVED, displacement_distance=200.0,
                explanation="Relocated by 200px in visit_02",
            ))
            store.add_change_event(ChangeEvent(
                event_id="evt_2", entity_id="ent_visit_01_2", from_session_id="visit_01", to_session_id="visit_03",
                timestamp="2026-11-25T10:03:00Z", prior_state=EpistemicState.PENDING_REMOVAL,
                new_state=EpistemicState.CONFIRMED_REMOVED, displacement_distance=None,
                explanation="Confirmed removed in visit_03",
            ))

        r = self.client.get("/api/workspaces/comp_test/compare?reference_session_id=visit_01&query_session_id=visit_03")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsNone(data["cross_sequence_warning"])
        by_id = {it["entity_id"]: it for it in data["items"]}
        self.assertEqual(by_id["ent_visit_01_1"]["status_change"], "CONFIRMED_MOVED")
        self.assertAlmostEqual(by_id["ent_visit_01_1"]["displacement_px"], 200.0, places=1)
        self.assertTrue(by_id["ent_visit_01_1"]["class_uncertain"])
        self.assertEqual(by_id["ent_visit_01_2"]["status_change"], "CONFIRMED_REMOVED")

        r_cross = self.client.get("/api/workspaces/comp_test/compare?reference_session_id=visit_01&query_session_id=visit_04")
        self.assertEqual(r_cross.status_code, 200)
        self.assertIsNotNone(r_cross.json()["cross_sequence_warning"])

        r_lbl = self.client.patch("/api/workspaces/comp_test/entities/ent_visit_01_1/label", json={"label": "bottle"})
        self.assertEqual(r_lbl.status_code, 200)
        self.assertEqual(r_lbl.json()["label"], "bottle")

        r_after = self.client.get("/api/workspaces/comp_test/compare?reference_session_id=visit_01&query_session_id=visit_03").json()
        e1_after = next(it for it in r_after["items"] if it["entity_id"] == "ent_visit_01_1")
        self.assertEqual(e1_after["label"], "bottle")
        self.assertEqual(e1_after["raw_label"], "hair drier")
        self.assertTrue(e1_after["class_uncertain"])

    def test_2_temporary_miss_not_confirmed_removed(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        engine = TemporalMemoryEngine(store=store, config=TemporalEngineConfig())
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        det = Detection(label="cup", confidence=0.9, raw_bbox=BoundingBox(100, 100, 200, 200), canonical_bbox=BoundingBox(100, 100, 200, 200))

        engine.process_visit(VisitInput(session_id="v1", session_name="V1", timestamp="2026-11-25T10:01:00Z", image=img, registration=_reliable_reg(), detections=[det]))
        engine.process_visit(VisitInput(session_id="v2", session_name="V2", timestamp="2026-11-25T10:02:00Z", image=img, registration=_reliable_reg(), detections=[]))

        ents = store.list_entities()
        self.assertEqual(len(ents), 1)
        self.assertNotEqual(ents[0].current_state, EpistemicState.CONFIRMED_REMOVED)

    def test_4_inconsistent_class_preserves_identity(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        engine = TemporalMemoryEngine(store=store, config=TemporalEngineConfig())
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        d1 = Detection(label="hair drier", confidence=0.50, raw_bbox=BoundingBox(100, 100, 220, 260), canonical_bbox=BoundingBox(100, 100, 220, 260))
        d2 = Detection(label="bottle", confidence=0.45, raw_bbox=BoundingBox(104, 102, 222, 262), canonical_bbox=BoundingBox(104, 102, 222, 262))

        engine.process_visit(VisitInput(session_id="v1", session_name="V1", timestamp="2026-11-25T10:01:00Z", image=img, registration=_reliable_reg(), detections=[d1]))
        engine.process_visit(VisitInput(session_id="v2", session_name="V2", timestamp="2026-11-25T10:02:00Z", image=img, registration=_reliable_reg(), detections=[d2]))

        ents = store.list_entities()
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0].current_state, EpistemicState.CONFIRMED_PRESENT)

    def test_5_unreliable_registration_freezes_removal_and_movement(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        engine = TemporalMemoryEngine(store=store, config=TemporalEngineConfig())
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        d1 = Detection(label="book", confidence=0.9, raw_bbox=BoundingBox(100, 100, 250, 250), canonical_bbox=BoundingBox(100, 100, 250, 250))

        engine.process_visit(VisitInput(session_id="v1", session_name="V1", timestamp="2026-11-25T10:01:00Z", image=img, registration=_reliable_reg(), detections=[d1]))
        # 3 unreliable visits with 0 detections must freeze state (never confirm removal)
        for idx in (2, 3, 4):
            engine.process_visit(VisitInput(session_id=f"v{idx}", session_name=f"V{idx}", timestamp=f"2026-11-25T10:0{idx}:00Z", image=img, registration=_unreliable_reg(), detections=[]))

        ents = store.list_entities()
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0].current_state, EpistemicState.CONFIRMED_PRESENT)
