"""
Automated tests for SQLite persistence, relationships, foreign-key enforcement,
reopening databases, schema migrations, and epistemic safeguards.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.domain.models import (
    BoundingBox,
    ChangeEvent,
    Observation,
    PairwiseBaselineResult,
    Session,
    WorkbenchEntity,
)
from objectmemory.domain.states import (
    BaselineChangeType,
    EpistemicState,
    VisibilityStatus,
)
from objectmemory.persistence.repository import SQLiteMemoryStore, _json_to_bbox
from objectmemory.persistence.schema import CURRENT_SCHEMA_VERSION


class TestSQLitePersistence(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_memory.db"
        self.store = SQLiteMemoryStore(self.db_path)

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_clean_database_initialization(self):
        """Verifies tables and indexes are initialized correctly from scratch."""
        cursor = self.store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        for expected in ["sessions", "entities", "observations", "change_events", "pairwise_baseline_results", "raw_detections"]:
            self.assertIn(expected, tables)

        # Check user_version
        cursor = self.store.conn.execute("PRAGMA user_version;")
        self.assertEqual(cursor.fetchone()[0], CURRENT_SCHEMA_VERSION)

    def test_session_lifecycle(self):
        session = Session(
            session_id="sess_001",
            session_name="Monday Morning Baseline",
            timestamp="2026-10-10T09:00:00Z",
            image_path="/data/workbench_001.jpg",
            notes="Controlled lighting, overhead phone capture",
            metadata={"illuminance_lux": 450, "operator": "Alice"},
        )
        self.store.add_session(session)

        retrieved = self.store.get_session("sess_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, "sess_001")
        self.assertEqual(retrieved.session_name, "Monday Morning Baseline")
        self.assertEqual(retrieved.metadata["illuminance_lux"], 450)

        # Test listing
        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 1)

    def test_entity_lifecycle_and_deprecated_update_warning(self):
        bbox = BoundingBox(50.0, 100.0, 150.0, 200.0)
        entity = WorkbenchEntity(
            entity_id="obj_multimeter",
            label="digital_multimeter",
            created_at="2026-10-10T09:00:00Z",
            canonical_bbox=bbox,
            current_state=EpistemicState.CONFIRMED_PRESENT,
            metadata={"color": "yellow", "model": "Fluke 87V"},
        )
        self.store.add_entity(entity)

        retrieved = self.store.get_entity("obj_multimeter")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.label, "digital_multimeter")
        self.assertEqual(retrieved.canonical_bbox, bbox)
        self.assertEqual(retrieved.current_state, EpistemicState.CONFIRMED_PRESENT)

        # Using deprecated update_entity_state issues warning
        with self.assertWarns(DeprecationWarning):
            self.store.update_entity_state(
                entity_id="obj_multimeter",
                new_state=EpistemicState.CONFIRMED_MOVED,
            )

    def test_record_state_transition_success(self):
        """Priority 1: Successfully records atomic state transition and event."""
        s1 = Session(session_id="s1", session_name="S1", timestamp="2026-10-10T09:00:00Z", image_path="1.jpg")
        s2 = Session(session_id="s2", session_name="S2", timestamp="2026-10-10T10:00:00Z", image_path="2.jpg")
        self.store.add_session(s1)
        self.store.add_session(s2)

        entity = WorkbenchEntity(
            entity_id="iron_1",
            label="soldering_iron",
            created_at="2026-10-10T09:00:00Z",
            canonical_bbox=BoundingBox(10.0, 10.0, 50.0, 50.0),
            current_state=EpistemicState.CONFIRMED_PRESENT,
            last_seen_session_id="s1",
        )
        self.store.add_entity(entity)

        new_bbox = BoundingBox(60.0, 10.0, 100.0, 50.0)
        event = ChangeEvent(
            event_id="evt_move_1",
            entity_id="iron_1",
            from_session_id="s1",
            to_session_id="s2",
            timestamp="2026-10-10T10:00:00Z",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_MOVED,
            displacement_distance=50.0,
            supporting_evidence={"delta_x": 50.0},
            explanation="Iron moved 50 units right.",
        )

        self.store.record_state_transition(
            entity_id="iron_1",
            new_state=EpistemicState.CONFIRMED_MOVED,
            event=event,
            canonical_bbox=new_bbox,
            last_seen_session_id="s2",
        )

        updated_entity = self.store.get_entity("iron_1")
        self.assertEqual(updated_entity.current_state, EpistemicState.CONFIRMED_MOVED)
        self.assertEqual(updated_entity.canonical_bbox, new_bbox)
        self.assertEqual(updated_entity.last_seen_session_id, "s2")

        events = self.store.list_change_events_for_entity("iron_1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, "evt_move_1")
        self.assertEqual(events[0].displacement_distance, 50.0)

    def test_record_state_transition_policy_rejection(self):
        """Priority 1: Policy rejects unverified direct jump from PRESENT to REMOVED."""
        s1 = Session(session_id="s1", session_name="S1", timestamp="2026-10-10T09:00:00Z", image_path="1.jpg")
        s2 = Session(session_id="s2", session_name="S2", timestamp="2026-10-10T10:00:00Z", image_path="2.jpg")
        self.store.add_session(s1)
        self.store.add_session(s2)

        entity = WorkbenchEntity(
            entity_id="meter_1",
            label="multimeter",
            created_at="2026-10-10T09:00:00Z",
            current_state=EpistemicState.CONFIRMED_PRESENT,
        )
        self.store.add_entity(entity)

        invalid_event = ChangeEvent(
            event_id="evt_premature_removal",
            entity_id="meter_1",
            from_session_id="s1",
            to_session_id="s2",
            timestamp="2026-10-10T10:00:00Z",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_REMOVED,
        )

        with self.assertRaises(ValueError):
            self.store.record_state_transition(
                entity_id="meter_1",
                new_state=EpistemicState.CONFIRMED_REMOVED,
                event=invalid_event,
            )

        # Entity remains PRESENT
        self.assertEqual(self.store.get_entity("meter_1").current_state, EpistemicState.CONFIRMED_PRESENT)

    def test_record_state_transition_mismatch_rejection(self):
        """Priority 1: Rejects transition when event prior_state doesn't match actual current state."""
        s1 = Session(session_id="s1", session_name="S1", timestamp="2026-10-10T09:00:00Z", image_path="1.jpg")
        s2 = Session(session_id="s2", session_name="S2", timestamp="2026-10-10T10:00:00Z", image_path="2.jpg")
        self.store.add_session(s1)
        self.store.add_session(s2)

        entity = WorkbenchEntity(
            entity_id="board_1",
            label="breadboard",
            created_at="2026-10-10T09:00:00Z",
            current_state=EpistemicState.CONFIRMED_PRESENT,
        )
        self.store.add_entity(entity)

        mismatched_event = ChangeEvent(
            event_id="evt_bad_prior",
            entity_id="board_1",
            from_session_id="s1",
            to_session_id="s2",
            timestamp="2026-10-10T10:00:00Z",
            prior_state=EpistemicState.INSUFFICIENT_EVIDENCE,  # Does not match entity's CONFIRMED_PRESENT
            new_state=EpistemicState.CONFIRMED_MOVED,
        )

        with self.assertRaises(ValueError):
            self.store.record_state_transition(
                entity_id="board_1",
                new_state=EpistemicState.CONFIRMED_MOVED,
                event=mismatched_event,
            )

    def test_record_state_transition_atomic_rollback(self):
        """Priority 1: When event insertion fails, entity update must be rolled back."""
        s1 = Session(session_id="s1", session_name="S1", timestamp="2026-10-10T09:00:00Z", image_path="1.jpg")
        self.store.add_session(s1)

        entity = WorkbenchEntity(
            entity_id="scope_1",
            label="oscilloscope",
            created_at="2026-10-10T09:00:00Z",
            current_state=EpistemicState.CONFIRMED_PRESENT,
        )
        self.store.add_entity(entity)

        # Event pointing to a non-existent session
        failing_event = ChangeEvent(
            event_id="evt_fail",
            entity_id="scope_1",
            from_session_id="s1",
            to_session_id="non_existent_session_id",  # Triggers foreign key IntegrityError
            timestamp="2026-10-10T10:00:00Z",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_MOVED,
        )

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.record_state_transition(
                entity_id="scope_1",
                new_state=EpistemicState.CONFIRMED_MOVED,
                event=failing_event,
            )

        # Verify rollback: entity state must NOT be modified
        recovered = self.store.get_entity("scope_1")
        self.assertEqual(recovered.current_state, EpistemicState.CONFIRMED_PRESENT)
        # Verify no event was persisted
        events = self.store.list_change_events_for_entity("scope_1")
        self.assertEqual(len(events), 0)

    def test_historical_records_protected_from_deletion(self):
        """Priority 2: ON DELETE RESTRICT prevents accidental deletion of historical evidence."""
        s1 = Session(session_id="s1", session_name="S1", timestamp="2026-10-10T09:00:00Z", image_path="1.jpg")
        s2 = Session(session_id="s2", session_name="S2", timestamp="2026-10-10T10:00:00Z", image_path="2.jpg")
        self.store.add_session(s1)
        self.store.add_session(s2)

        entity = WorkbenchEntity(
            entity_id="protected_item",
            label="calibrator",
            created_at="2026-10-10T09:00:00Z",
            current_state=EpistemicState.CONFIRMED_PRESENT,
        )
        self.store.add_entity(entity)

        event = ChangeEvent(
            event_id="evt_hist",
            entity_id="protected_item",
            from_session_id="s1",
            to_session_id="s2",
            timestamp="2026-10-10T10:00:00Z",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_MOVED,
        )
        self.store.record_state_transition(
            entity_id="protected_item",
            new_state=EpistemicState.CONFIRMED_MOVED,
            event=event,
        )

        # Deleting the referenced entity must fail with IntegrityError (RESTRICT)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.conn:
                self.store.conn.execute("DELETE FROM entities WHERE entity_id = 'protected_item';")

        # Deleting the referenced session must fail with IntegrityError (RESTRICT)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.conn:
                self.store.conn.execute("DELETE FROM sessions WHERE session_id = 's2';")

        # Verify entity and history still exist completely intact
        self.assertIsNotNone(self.store.get_entity("protected_item"))
        self.assertEqual(len(self.store.list_change_events_for_entity("protected_item")), 1)

    def test_deterministic_ordering_for_equal_timestamps(self):
        """Priority 4: Verifies deterministic ordering when timestamps are identical."""
        # Sessions with exact same timestamp
        sB = Session(session_id="sess_b", session_name="B", timestamp="2026-10-10T12:00:00Z", image_path="b.jpg")
        sA = Session(session_id="sess_a", session_name="A", timestamp="2026-10-10T12:00:00Z", image_path="a.jpg")
        self.store.add_session(sB)
        self.store.add_session(sA)

        sessions = self.store.list_sessions(order_asc=True)
        self.assertEqual([s.session_id for s in sessions], ["sess_a", "sess_b"])

        # Change events with exact same timestamp
        ent = WorkbenchEntity(entity_id="ent_tie", label="resistor", created_at="2026-10-10T12:00:00Z")
        self.store.add_entity(ent)

        ev2 = ChangeEvent(
            event_id="evt_02",
            entity_id="ent_tie",
            to_session_id="sess_a",
            timestamp="2026-10-10T12:00:00Z",
            prior_state=EpistemicState.INSUFFICIENT_EVIDENCE,
            new_state=EpistemicState.CONFIRMED_PRESENT,
        )
        ev1 = ChangeEvent(
            event_id="evt_01",
            entity_id="ent_tie",
            to_session_id="sess_a",
            timestamp="2026-10-10T12:00:00Z",
            prior_state=EpistemicState.INSUFFICIENT_EVIDENCE,
            new_state=EpistemicState.CONFIRMED_PRESENT,
        )
        self.store.add_change_event(ev2)
        self.store.add_change_event(ev1)

        events = self.store.list_change_events_for_entity("ent_tie")
        self.assertEqual([e.event_id for e in events], ["evt_01", "evt_02"])

    def test_malformed_bounding_box_json_raises_clear_error(self):
        """Priority 3: Malformed bounding box deserialization raises documented ValueError."""
        with self.assertRaises(ValueError):
            _json_to_bbox("invalid json {")
        with self.assertRaises(ValueError):
            _json_to_bbox("[10.0, 20.0]")  # Only 2 elements
        with self.assertRaises(ValueError):
            _json_to_bbox("[10.0, 20.0, 30.0, 40.0, 50.0]")  # 5 elements

    def test_schema_migration_from_v1_to_v2(self):
        """Priority 2: Safely migrates pre-existing v1 databases without data loss."""
        mig_db_path = Path(self.temp_dir.name) / "v1_legacy.db"
        # Simulate legacy v1 database
        conn = sqlite3.connect(str(mig_db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                notes TEXT,
                metadata_json TEXT
            );
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                canonical_bbox_json TEXT,
                current_state TEXT NOT NULL,
                last_seen_session_id TEXT,
                metadata_json TEXT
            );
            CREATE TABLE change_events (
                event_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                from_session_id TEXT,
                to_session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prior_state TEXT NOT NULL,
                new_state TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'TEMPORAL_MEMORY',
                displacement_distance REAL,
                supporting_evidence_json TEXT,
                explanation TEXT,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
                FOREIGN KEY (to_session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            CREATE TABLE observations (
                observation_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                entity_id TEXT,
                timestamp TEXT NOT NULL,
                image_reference TEXT NOT NULL,
                raw_bbox_json TEXT,
                canonical_bbox_json TEXT,
                visibility_status TEXT NOT NULL,
                detector_confidence REAL,
                identity_match_score REAL,
                alignment_quality REAL,
                appearance_descriptor_ref TEXT,
                raw_evidence_json TEXT
            );
            CREATE TABLE pairwise_baseline_results (
                result_id TEXT PRIMARY KEY,
                reference_session_id TEXT NOT NULL,
                query_session_id TEXT NOT NULL,
                entity_id TEXT,
                predicted_change TEXT NOT NULL,
                confidence REAL,
                details_json TEXT
            );
            PRAGMA user_version = 1;
        """)
        # Insert legacy records
        conn.execute("INSERT INTO sessions VALUES ('s_leg', 'Legacy', '2026-10-10T09:00:00Z', 'l.jpg', NULL, NULL);")
        conn.execute("INSERT INTO entities VALUES ('e_leg', 'Legacy Tool', '2026-10-10T09:00:00Z', NULL, 'CONFIRMED_PRESENT', 's_leg', NULL);")
        conn.execute("INSERT INTO change_events VALUES ('ev_leg', 'e_leg', NULL, 's_leg', '2026-10-10T09:00:00Z', 'INSUFFICIENT_EVIDENCE', 'CONFIRMED_PRESENT', 'TEMPORAL_MEMORY', NULL, NULL, 'Initial');")
        conn.commit()
        conn.close()

        # Open through SQLiteMemoryStore - triggers apply_migrations
        mig_store = SQLiteMemoryStore(mig_db_path)
        try:
            # Verify data retained
            self.assertIsNotNone(mig_store.get_session("s_leg"))
            self.assertIsNotNone(mig_store.get_entity("e_leg"))
            self.assertEqual(len(mig_store.list_change_events_for_entity("e_leg")), 1)

            # Verify user_version upgraded to CURRENT_SCHEMA_VERSION
            cursor = mig_store.conn.execute("PRAGMA user_version;")
            self.assertEqual(cursor.fetchone()[0], CURRENT_SCHEMA_VERSION)

            # Verify RESTRICT now protects against deletion
            with self.assertRaises(sqlite3.IntegrityError):
                with mig_store.conn:
                    mig_store.conn.execute("DELETE FROM entities WHERE entity_id = 'e_leg';")
        finally:
            mig_store.close()

    def test_database_persistence_across_reopen(self):
        """Ensures closing the store and reopening it restores all state cleanly."""
        session = Session(
            session_id="sess_persist",
            session_name="Persistence Test Session",
            timestamp="2026-10-10T09:00:00Z",
            image_path="/data/img.jpg",
        )
        self.store.add_session(session)

        entity = WorkbenchEntity(
            entity_id="ent_persist",
            label="soldering_iron",
            created_at="2026-10-10T09:00:00Z",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            last_seen_session_id="sess_persist",
        )
        self.store.add_entity(entity)

        obs = Observation(
            observation_id="obs_persist",
            session_id="sess_persist",
            entity_id="ent_persist",
            timestamp="2026-10-10T09:00:00Z",
            image_reference="crop_01.jpg",
            visibility_status=VisibilityStatus.CLEARLY_VISIBLE,
            detector_confidence=0.96,
            identity_match_score=0.91,
            alignment_quality=0.99,
            raw_evidence={"peak_snr": 34.5},
        )
        self.store.add_observation(obs)

        # Close the connection
        self.store.close()

        # Reopen with brand new store instance
        reopened_store = SQLiteMemoryStore(self.db_path)
        try:
            r_sess = reopened_store.get_session("sess_persist")
            self.assertIsNotNone(r_sess)
            self.assertEqual(r_sess.session_name, "Persistence Test Session")

            r_ent = reopened_store.get_entity("ent_persist")
            self.assertIsNotNone(r_ent)
            self.assertEqual(r_ent.label, "soldering_iron")
            self.assertEqual(r_ent.current_state, EpistemicState.CONFIRMED_PRESENT)

            r_obs = reopened_store.get_observation("obs_persist")
            self.assertIsNotNone(r_obs)
            self.assertEqual(r_obs.detector_confidence, 0.96)
            self.assertEqual(r_obs.raw_evidence["peak_snr"], 34.5)
        finally:
            reopened_store.close()

    def test_safeguard_uncertainty_is_not_removal(self):
        """
        Safeguard: An unobserved or uncertain observation must NOT silently
        convert an entity to CONFIRMED_REMOVED.
        """
        sess1 = Session(
            session_id="sess_t0",
            session_name="Visit 0",
            timestamp="2026-10-10T09:00:00Z",
            image_path="img0.jpg",
        )
        sess2 = Session(
            session_id="sess_t1",
            session_name="Visit 1",
            timestamp="2026-10-10T11:00:00Z",
            image_path="img1.jpg",
        )
        self.store.add_session(sess1)
        self.store.add_session(sess2)

        entity = WorkbenchEntity(
            entity_id="probe_1",
            label="oscilloscope_probe",
            created_at="2026-10-10T09:00:00Z",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            last_seen_session_id="sess_t0",
        )
        self.store.add_entity(entity)

        # In Visit 1, the detector fails or view is occluded (detector_confidence is None)
        obs_occluded = Observation(
            observation_id="obs_probe_t1",
            session_id="sess_t1",
            entity_id="probe_1",
            timestamp="2026-10-10T11:00:00Z",
            image_reference="img1.jpg",
            visibility_status=VisibilityStatus.HEAVILY_OCCLUDED,
            detector_confidence=None,  # No manufactured score
            identity_match_score=None,
            alignment_quality=0.88,
            raw_evidence={"reason": "obscured by technician arm"},
        )
        self.store.add_observation(obs_occluded)

        # Entity in persistent storage must NOT be marked removed!
        current_entity = self.store.get_entity("probe_1")
        self.assertNotEqual(current_entity.current_state, EpistemicState.CONFIRMED_REMOVED)
        self.assertEqual(current_entity.current_state, EpistemicState.CONFIRMED_PRESENT)

        # If an epistemic reasoner evaluates this evidence, it transitions to INSUFFICIENT_EVIDENCE
        event = ChangeEvent(
            event_id="evt_t1_probe",
            entity_id="probe_1",
            from_session_id="sess_t0",
            to_session_id="sess_t1",
            timestamp="2026-10-10T11:00:00Z",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.INSUFFICIENT_EVIDENCE,
            supporting_evidence={"visibility": "HEAVILY_OCCLUDED"},
            explanation="Object region occluded; lack of detection cannot confirm removal.",
        )
        self.store.record_state_transition(
            entity_id="probe_1",
            new_state=EpistemicState.INSUFFICIENT_EVIDENCE,
            event=event,
        )

        audited_entity = self.store.get_entity("probe_1")
        self.assertEqual(audited_entity.current_state, EpistemicState.INSUFFICIENT_EVIDENCE)
        self.assertNotEqual(audited_entity.current_state, EpistemicState.CONFIRMED_REMOVED)

    def test_chronological_change_event_audit(self):
        """Verifies state transitions are retrieved in chronological order with tie-breakers."""
        sess1 = Session(session_id="s1", session_name="S1", timestamp="2026-10-10T09:00:00Z", image_path="1.jpg")
        sess2 = Session(session_id="s2", session_name="S2", timestamp="2026-10-10T10:00:00Z", image_path="2.jpg")
        sess3 = Session(session_id="s3", session_name="S3", timestamp="2026-10-10T11:00:00Z", image_path="3.jpg")
        self.store.add_session(sess1)
        self.store.add_session(sess2)
        self.store.add_session(sess3)

        ent = WorkbenchEntity(
            entity_id="breadboard_1",
            label="solderless_breadboard",
            created_at="2026-10-10T09:00:00Z",
            current_state=EpistemicState.CONFIRMED_PRESENT,
        )
        self.store.add_entity(ent)

        evt1 = ChangeEvent(
            event_id="ev1",
            entity_id="breadboard_1",
            from_session_id="s1",
            to_session_id="s2",
            timestamp="2026-10-10T10:00:00Z",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_MOVED,
            displacement_distance=8.2,
            explanation="Shifted left.",
        )
        self.store.record_state_transition(
            entity_id="breadboard_1",
            new_state=EpistemicState.CONFIRMED_MOVED,
            event=evt1,
        )

        evt2 = ChangeEvent(
            event_id="ev2",
            entity_id="breadboard_1",
            from_session_id="s2",
            to_session_id="s3",
            timestamp="2026-10-10T11:00:00Z",
            prior_state=EpistemicState.CONFIRMED_MOVED,
            new_state=EpistemicState.PENDING_REMOVAL,
            explanation="Not observed in location; pending verification.",
        )
        self.store.record_state_transition(
            entity_id="breadboard_1",
            new_state=EpistemicState.PENDING_REMOVAL,
            event=evt2,
        )

        events = self.store.list_change_events_for_entity("breadboard_1")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].new_state, EpistemicState.CONFIRMED_MOVED)
        self.assertEqual(events[1].new_state, EpistemicState.PENDING_REMOVAL)
        self.assertEqual(events[0].timestamp, "2026-10-10T10:00:00Z")
        self.assertEqual(events[1].timestamp, "2026-10-10T11:00:00Z")

    def test_pairwise_baseline_isolation(self):
        """Verifies pairwise baseline results are isolated from persistent entity state."""
        sess1 = Session(session_id="sA", session_name="SA", timestamp="2026-10-10T09:00:00Z", image_path="A.jpg")
        sess2 = Session(session_id="sB", session_name="SB", timestamp="2026-10-10T10:00:00Z", image_path="B.jpg")
        self.store.add_session(sess1)
        self.store.add_session(sess2)

        pairwise = PairwiseBaselineResult(
            result_id="pair_001",
            reference_session_id="sA",
            query_session_id="sB",
            entity_id=None,
            predicted_change=BaselineChangeType.REMOVED,
            confidence=0.74,
            details={"heuristic": "naive_iou_difference", "threshold": 0.5},
        )
        self.store.add_pairwise_result(pairwise)

        results = self.store.list_pairwise_results(reference_session_id="sA")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].predicted_change, BaselineChangeType.REMOVED)

        # Baseline records do not create or alter entities in memory
        entities = self.store.list_entities()
        self.assertEqual(len(entities), 0)


if __name__ == "__main__":
    unittest.main()
