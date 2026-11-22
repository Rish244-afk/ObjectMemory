"""
Unit and integration tests for Live Camera Detection Persistence in ObjectMemory.

Verifies:
1. Valid detector outputs are persisted in raw_detections independently of registration success.
2. Unregistered detections are marked UNREGISTERED and do NOT create false persistent entities.
3. Failed registration does NOT advance absence confirmation or emit false removal events.
4. Successful registration continues normal canonical projection, association, and persistence.
5. API and WorkspaceManager expose raw_detections, raw_detections_count, and registration summary.
6. Database migration safely adds raw_detections and updates user_version to 3.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import numpy as np

from objectmemory.api.workspace_manager import WorkspaceManager
from objectmemory.baseline.comparator import calculate_bbox_iou
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.domain.models import BoundingBox, RawDetection, Session, WorkbenchEntity
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.persistence.schema import CURRENT_SCHEMA_VERSION, apply_migrations
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput


class TestLiveCameraPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_memory.db"
        self.store = SQLiteMemoryStore(self.db_path, check_same_thread=False)
        self.engine = TemporalMemoryEngine(store=self.store, config=TemporalEngineConfig())

    def tearDown(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_schema_migration_v2_to_v3(self) -> None:
        """Verifies that user_version is bumped to 3 and raw_detections table exists."""
        cursor = self.store.conn.cursor()
        cursor.execute("PRAGMA user_version;")
        ver = cursor.fetchone()[0]
        self.assertEqual(ver, CURRENT_SCHEMA_VERSION)
        self.assertEqual(ver, 3)

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_detections';")
        self.assertIsNotNone(cursor.fetchone())

    def test_detection_with_failed_registration_persists_raw_detections(self) -> None:
        """
        Tests that when registration fails on a visit:
        - All detections are saved to raw_detections as UNREGISTERED.
        - Canonical coordinates are None.
        - Zero observations are written to the registered observations table.
        """
        # Visit 1: Baseline reference establishment
        v1_det = Detection(
            label="cup",
            confidence=0.88,
            raw_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
            canonical_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
        )
        v1_reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            homography=np.eye(3, dtype=float),
            inlier_ratio=1.0,
            num_inliers=100,
            ref_size=(640, 480),
            query_size=(640, 480),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            method_used="IDENTITY_REFERENCE",
        )
        v1 = VisitInput(
            session_id="visit_01",
            session_name="Visit 01 Baseline",
            timestamp="2026-10-30T10:00:00Z",
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            registration=v1_reg,
            detections=[v1_det],
        )
        report1 = self.engine.process_visit(v1)
        self.assertEqual(report1.new_entities_created, 1)

        # Verify Visit 1 raw detection is persisted as REGISTERED
        raw_v1 = self.store.list_raw_detections_for_session("visit_01")
        self.assertEqual(len(raw_v1), 1)
        self.assertEqual(raw_v1[0].label, "cup")
        self.assertEqual(raw_v1[0].registration_status, "REGISTERED")
        self.assertIsNotNone(raw_v1[0].canonical_bbox)

        # Visit 2: Live camera frame where registration FAILS
        v2_det1 = Detection(
            label="bottle",
            confidence=0.75,
            raw_bbox=BoundingBox(300.0, 200.0, 400.0, 450.0),
        )
        v2_det2 = Detection(
            label="cup",
            confidence=0.82,
            raw_bbox=BoundingBox(110.0, 95.0, 210.0, 205.0),
        )
        v2_reg_failed = RegistrationResult(
            success=False,
            status=AlignmentStatus.INSUFFICIENT_MATCHES,
            homography=None,
            inlier_ratio=0.0,
            num_inliers=2,
            ref_size=(640, 480),
            query_size=(1280, 720),
            failure_reason="Insufficient feature matches: 2 < 10",
            method_used="SIFT_HOMOGRAPHY",
        )
        v2 = VisitInput(
            session_id="visit_02",
            session_name="Visit 02 Live Camera",
            timestamp="2026-10-30T10:01:00Z",
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
            registration=v2_reg_failed,
            detections=[v2_det1, v2_det2],
        )
        report2 = self.engine.process_visit(v2)

        # Verify raw detections are persisted independently of registration failure
        raw_v2 = self.store.list_raw_detections_for_session("visit_02")
        self.assertEqual(len(raw_v2), 2)
        labels = {d.label for d in raw_v2}
        self.assertEqual(labels, {"bottle", "cup"})
        for d in raw_v2:
            self.assertEqual(d.registration_status, "UNREGISTERED")
            self.assertIsNone(d.canonical_bbox)

        # Verify that ZERO registered observations were created for visit_02
        obs_v2 = self.store.list_observations_for_session("visit_02")
        self.assertEqual(len(obs_v2), 0)

    def test_failed_registration_does_not_create_false_entities_or_removals(self) -> None:
        """
        Tests that when registration fails:
        1. No new persistent entities are created from unregistered detections.
        2. Existing entities are NOT falsely marked PENDING_REMOVAL or CONFIRMED_REMOVED.
        3. Zero change events are emitted.
        """
        # Baseline visit with one entity
        v1_det = Detection(
            label="keyboard",
            confidence=0.90,
            raw_bbox=BoundingBox(50.0, 50.0, 250.0, 150.0),
            canonical_bbox=BoundingBox(50.0, 50.0, 250.0, 150.0),
        )
        v1_reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            homography=np.eye(3, dtype=float),
            inlier_ratio=1.0,
            num_inliers=100,
            ref_size=(640, 480),
            query_size=(640, 480),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            method_used="IDENTITY_REFERENCE",
        )
        self.engine.process_visit(
            VisitInput(
                session_id="visit_01",
                session_name="V1",
                timestamp="2026-10-30T10:00:00Z",
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                registration=v1_reg,
                detections=[v1_det],
            )
        )
        initial_entities = self.store.list_entities()
        self.assertEqual(len(initial_entities), 1)
        initial_ent_id = initial_entities[0].entity_id

        # Visit 2: Live camera frame with completely different detections and FAILED registration
        v2_dets = [
            Detection(label="mouse", confidence=0.85, raw_bbox=BoundingBox(300.0, 300.0, 350.0, 380.0)),
            Detection(label="notebook", confidence=0.78, raw_bbox=BoundingBox(400.0, 100.0, 500.0, 250.0)),
        ]
        v2_reg_failed = RegistrationResult(
            success=False,
            status=AlignmentStatus.INSUFFICIENT_INLIERS,
            homography=None,
            inlier_ratio=0.10,
            num_inliers=5,
            ref_size=(640, 480),
            query_size=(1280, 720),
            failure_reason="Inlier ratio below threshold",
            method_used="SIFT_HOMOGRAPHY",
        )
        report2 = self.engine.process_visit(
            VisitInput(
                session_id="visit_02",
                session_name="V2",
                timestamp="2026-10-30T10:02:00Z",
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
                registration=v2_reg_failed,
                detections=v2_dets,
            )
        )

        # Check: Zero new entities created
        self.assertEqual(report2.new_entities_created, 0)
        current_entities = self.store.list_entities()
        self.assertEqual(len(current_entities), 1)

        # Check: Zero state transitions
        self.assertEqual(report2.state_transitions_count, 0)

        # Check: Baseline entity remains CONFIRMED_PRESENT without false removal progression
        ent = self.store.get_entity(initial_ent_id)
        self.assertIsNotNone(ent)
        assert ent is not None
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)

        # Check: Zero change events emitted in visit_02
        events_v2 = self.store.list_change_events_for_session("visit_02")
        self.assertEqual(len(events_v2), 0)

    def test_successful_registration_continues_normal_pipeline(self) -> None:
        """
        Tests that when registration subsequently succeeds:
        - Raw detections are saved as REGISTERED.
        - Canonical bounding boxes are populated.
        - Entity associations, observations, and state transitions proceed normally.
        """
        # Visit 1: Baseline
        v1_det = Detection(
            label="book",
            confidence=0.92,
            raw_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
            canonical_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
        )
        v1_reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            homography=np.eye(3, dtype=float),
            inlier_ratio=1.0,
            num_inliers=100,
            ref_size=(640, 480),
            query_size=(640, 480),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            method_used="IDENTITY_REFERENCE",
        )
        self.engine.process_visit(
            VisitInput(
                session_id="visit_01",
                session_name="V1",
                timestamp="2026-10-30T10:00:00Z",
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                registration=v1_reg,
                detections=[v1_det],
            )
        )

        # Visit 2: Fails registration
        self.engine.process_visit(
            VisitInput(
                session_id="visit_02",
                session_name="V2",
                timestamp="2026-10-30T10:01:00Z",
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                registration=RegistrationResult(
                    success=False,
                    status=AlignmentStatus.INSUFFICIENT_MATCHES,
                    homography=None,
                    inlier_ratio=0.0,
                    num_inliers=0,
                    ref_size=(640, 480),
                    query_size=(640, 480),
                    method_used="SIFT_HOMOGRAPHY",
                ),
                detections=[Detection(label="book", confidence=0.85, raw_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0))],
            )
        )

        # Visit 3: Succeeds registration (identity homography)
        v3_det = Detection(
            label="book",
            confidence=0.89,
            raw_bbox=BoundingBox(102.0, 98.0, 201.0, 202.0),
        )
        v3_reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            homography=np.eye(3, dtype=float),
            inlier_ratio=0.85,
            num_inliers=85,
            ref_size=(640, 480),
            query_size=(640, 480),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            method_used="SIFT_HOMOGRAPHY",
        )
        report3 = self.engine.process_visit(
            VisitInput(
                session_id="visit_03",
                session_name="V3",
                timestamp="2026-10-30T10:02:00Z",
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                registration=v3_reg,
                detections=[v3_det],
            )
        )

        # Visit 3 matched the existing entity
        self.assertEqual(report3.matched_detections, 1)
        self.assertEqual(report3.new_entities_created, 0)

        # Visit 3 raw detection is REGISTERED
        raw_v3 = self.store.list_raw_detections_for_session("visit_03")
        self.assertEqual(len(raw_v3), 1)
        self.assertEqual(raw_v3[0].registration_status, "REGISTERED")
        self.assertIsNotNone(raw_v3[0].canonical_bbox)

        # Visit 3 wrote a valid observation
        obs_v3 = self.store.list_observations_for_session("visit_03")
        self.assertEqual(len(obs_v3), 1)
        self.assertEqual(obs_v3[0].visibility_status, VisibilityStatus.CLEARLY_VISIBLE)

    def test_workspace_manager_and_api_responses(self) -> None:
        """Tests that WorkspaceManager session details and summaries serialize raw detections."""
        workspace_id = "test-ws-persistence"
        wm = WorkspaceManager(base_dir=Path(self.temp_dir.name))
        wm.create_workspace(workspace_id, name="Test Persistence WS")

        db_path = wm._db_path(workspace_id)
        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            engine = TemporalMemoryEngine(store=store, config=TemporalEngineConfig())
            # Insert Visit 1 (failed registration with 2 raw detections)
            v1_dets = [
                Detection(label="bottle", confidence=0.88, raw_bbox=BoundingBox(10.0, 10.0, 50.0, 90.0)),
                Detection(label="cup", confidence=0.76, raw_bbox=BoundingBox(100.0, 100.0, 150.0, 160.0)),
            ]
            v1_reg = RegistrationResult(
                success=False,
                status=AlignmentStatus.INSUFFICIENT_MATCHES,
                homography=None,
                inlier_ratio=0.0,
                num_inliers=3,
                ref_size=(640, 480),
                query_size=(1280, 720),
                failure_reason="SIFT found insufficient keypoints",
                method_used="SIFT_HOMOGRAPHY",
            )
            engine.process_visit(
                VisitInput(
                    session_id="visit_01",
                    session_name="Visit 01 Unregistered",
                    timestamp="2026-10-30T10:00:00Z",
                    image=np.zeros((720, 1280, 3), dtype=np.uint8),
                    registration=v1_reg,
                    detections=v1_dets,
                )
            )

        # 1. Check list_sessions
        sessions = wm.list_sessions(workspace_id)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["observations_count"], 0)
        self.assertEqual(sessions[0]["raw_detections_count"], 2)

        # 2. Check get_session_detail
        detail = wm.get_session_detail(workspace_id, "visit_01")
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(len(detail["observations"]), 0)
        self.assertEqual(len(detail["raw_detections"]), 2)
        self.assertEqual(detail["session"]["raw_detections_count"], 2)

        # Check registration summary
        reg = detail["registration"]
        self.assertIsNotNone(reg)
        assert reg is not None
        self.assertFalse(reg["success"])
        self.assertEqual(reg["status"], "INSUFFICIENT_MATCHES")
        self.assertEqual(reg["failure_reason"], "SIFT found insufficient keypoints")


if __name__ == "__main__":
    unittest.main()
