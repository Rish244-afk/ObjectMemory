"""
SQLite repository implementation for ObjectMemory.

Provides complete CRUD operations, atomic transactions, foreign-key enforcement,
and JSON serialization for domain models.

Thread Confinement Notice:
SQLiteMemoryStore maintains a single sqlite3.Connection with default Python
thread confinement (check_same_thread=True). It is NOT thread-safe for concurrent
access across multiple threads. In multi-threaded applications (such as Streamlit),
each thread or request worker must instantiate its own repository instance or
serialize access through a mutual exclusion lock.
"""

from __future__ import annotations
import json
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from objectmemory.domain.models import (
    BoundingBox,
    ChangeEvent,
    Observation,
    PairwiseBaselineResult,
    RawDetection,
    Session,
    WorkbenchEntity,
)
from objectmemory.domain.states import (
    BaselineChangeType,
    EpistemicState,
    VisibilityStatus,
    is_valid_transition,
)
from objectmemory.persistence.schema import apply_migrations


def _bbox_to_json(bbox: Optional[BoundingBox]) -> Optional[str]:
    if bbox is None:
        return None
    return json.dumps(list(bbox.as_tuple()))


def _json_to_bbox(s: Optional[str]) -> Optional[BoundingBox]:
    if not s:
        return None
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Malformed JSON for bounding box: {s}") from exc

    if not isinstance(data, (list, tuple)) or len(data) != 4:
        raise ValueError(
            f"Serialized bounding box must be a 4-element sequence of numbers, got: {data}"
        )
    return BoundingBox.from_tuple(tuple(data))


def _dict_to_json(d: Optional[Dict[str, Any]]) -> Optional[str]:
    if d is None:
        return None
    return json.dumps(d)


def _json_to_dict(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Malformed JSON payload: {s}") from exc


class SQLiteMemoryStore:
    """
    Persistent SQLite storage for sessions, entities, observations, and events.
    Thread-confined: Instances must not be shared across threads without external locking.
    """

    def __init__(self, db_path: Union[str, Path], check_same_thread: bool = True) -> None:
        self.db_path = str(db_path)
        self.check_same_thread = check_same_thread
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._initialize_schema()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=self.check_same_thread)
        self._conn.row_factory = sqlite3.Row
        # Crucial safeguard: explicitly enforce foreign key constraints
        self._conn.execute("PRAGMA foreign_keys = ON;")

    def _initialize_schema(self) -> None:
        assert self._conn is not None
        apply_migrations(self._conn)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database connection is closed.")
        return self._conn

    def close(self) -> None:
        """Closes the active database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SQLiteMemoryStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------

    def add_session(self, session: Session) -> None:
        """Persists a new session within an atomic transaction."""
        query = """
            INSERT INTO sessions (session_id, session_name, timestamp, image_path, notes, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.conn:
            self.conn.execute(
                query,
                (
                    session.session_id,
                    session.session_name,
                    session.timestamp,
                    session.image_path,
                    session.notes,
                    _dict_to_json(session.metadata),
                ),
            )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieves a session by its unique ID."""
        cursor = self.conn.execute("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return Session(
            session_id=row["session_id"],
            session_name=row["session_name"],
            timestamp=row["timestamp"],
            image_path=row["image_path"],
            notes=row["notes"],
            metadata=_json_to_dict(row["metadata_json"]),
        )

    def list_sessions(self, order_asc: bool = True) -> List[Session]:
        """Lists all recorded sessions ordered by timestamp with session_id tie-breaker."""
        direction = "ASC" if order_asc else "DESC"
        cursor = self.conn.execute(
            f"SELECT * FROM sessions ORDER BY timestamp {direction}, session_id {direction};"
        )
        return [
            Session(
                session_id=row["session_id"],
                session_name=row["session_name"],
                timestamp=row["timestamp"],
                image_path=row["image_path"],
                notes=row["notes"],
                metadata=_json_to_dict(row["metadata_json"]),
            )
            for row in cursor.fetchall()
        ]

    # -------------------------------------------------------------------------
    # Persistent Entities
    # -------------------------------------------------------------------------

    def add_entity(self, entity: WorkbenchEntity) -> None:
        """Persists a new entity within an atomic transaction."""
        query = """
            INSERT INTO entities (
                entity_id, label, created_at, canonical_bbox_json,
                current_state, last_seen_session_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self.conn:
            self.conn.execute(
                query,
                (
                    entity.entity_id,
                    entity.label,
                    entity.created_at,
                    _bbox_to_json(entity.canonical_bbox),
                    entity.current_state.value,
                    entity.last_seen_session_id,
                    _dict_to_json(entity.metadata),
                ),
            )

    def get_entity(self, entity_id: str) -> Optional[WorkbenchEntity]:
        """Retrieves an entity by its ID."""
        cursor = self.conn.execute("SELECT * FROM entities WHERE entity_id = ?;", (entity_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return WorkbenchEntity(
            entity_id=row["entity_id"],
            label=row["label"],
            created_at=row["created_at"],
            canonical_bbox=_json_to_bbox(row["canonical_bbox_json"]),
            current_state=EpistemicState(row["current_state"]),
            last_seen_session_id=row["last_seen_session_id"],
            metadata=_json_to_dict(row["metadata_json"]),
        )

    def list_entities(self, state: Optional[EpistemicState] = None) -> List[WorkbenchEntity]:
        """Lists entities ordered by created_at with entity_id tie-breaker."""
        if state is not None:
            cursor = self.conn.execute(
                "SELECT * FROM entities WHERE current_state = ? ORDER BY created_at ASC, entity_id ASC;",
                (state.value,),
            )
        else:
            cursor = self.conn.execute("SELECT * FROM entities ORDER BY created_at ASC, entity_id ASC;")
        return [
            WorkbenchEntity(
                entity_id=row["entity_id"],
                label=row["label"],
                created_at=row["created_at"],
                canonical_bbox=_json_to_bbox(row["canonical_bbox_json"]),
                current_state=EpistemicState(row["current_state"]),
                last_seen_session_id=row["last_seen_session_id"],
                metadata=_json_to_dict(row["metadata_json"]),
            )
            for row in cursor.fetchall()
        ]

    # -------------------------------------------------------------------------
    # Atomic State Transitions & Events (Priority 1)
    # -------------------------------------------------------------------------

    def record_state_transition(
        self,
        entity_id: str,
        new_state: EpistemicState,
        event: ChangeEvent,
        canonical_bbox: Optional[BoundingBox] = None,
        last_seen_session_id: Optional[str] = None,
    ) -> None:
        """
        Atomically updates an entity's persistent state and records the supporting ChangeEvent.

        In a single transaction:
        1. Validates that the entity exists and retrieves its current state.
        2. Validates that the requested transition is permitted by policy.
        3. Verifies that event.entity_id matches entity_id.
        4. Verifies that event.prior_state matches the entity's current state.
        5. Verifies that event.new_state matches new_state.
        6. Updates the entity row.
        7. Inserts the ChangeEvent row.
        Any exception rolls back BOTH operations cleanly.
        """
        if not isinstance(new_state, EpistemicState):
            new_state = EpistemicState(new_state)

        with self.conn:
            # 1. Retrieve current entity
            entity = self.get_entity(entity_id)
            if not entity:
                raise KeyError(f"Entity with ID '{entity_id}' does not exist.")

            # 2. Validate transition policy
            if not is_valid_transition(entity.current_state, new_state):
                raise ValueError(
                    f"Invalid state transition from '{entity.current_state.value}' to '{new_state.value}' "
                    f"for entity '{entity_id}'."
                )

            # 3. Validate event attributes
            if event.entity_id != entity_id:
                raise ValueError(
                    f"ChangeEvent entity_id ('{event.entity_id}') does not match target entity_id ('{entity_id}')."
                )
            if event.prior_state != entity.current_state:
                raise ValueError(
                    f"ChangeEvent prior_state ('{event.prior_state.value}') does not match entity's "
                    f"actual current state ('{entity.current_state.value}')."
                )
            if event.new_state != new_state:
                raise ValueError(
                    f"ChangeEvent new_state ('{event.new_state.value}') does not match target "
                    f"new_state ('{new_state.value}')."
                )

            # 4. Prepare entity update
            updates = ["current_state = ?"]
            params: List[Any] = [new_state.value]

            if last_seen_session_id is not None:
                updates.append("last_seen_session_id = ?")
                params.append(last_seen_session_id)

            if canonical_bbox is not None:
                updates.append("canonical_bbox_json = ?")
                params.append(_bbox_to_json(canonical_bbox))

            params.append(entity_id)
            entity_query = f"UPDATE entities SET {', '.join(updates)} WHERE entity_id = ?;"
            self.conn.execute(entity_query, params)

            # 5. Insert ChangeEvent
            event_query = """
                INSERT INTO change_events (
                    event_id, entity_id, from_session_id, to_session_id, timestamp,
                    prior_state, new_state, source, displacement_distance,
                    supporting_evidence_json, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            self.conn.execute(
                event_query,
                (
                    event.event_id,
                    event.entity_id,
                    event.from_session_id,
                    event.to_session_id,
                    event.timestamp,
                    event.prior_state.value,
                    event.new_state.value,
                    event.source,
                    event.displacement_distance,
                    _dict_to_json(event.supporting_evidence),
                    event.explanation,
                ),
            )

    def update_entity_state(
        self,
        entity_id: str,
        new_state: EpistemicState,
        last_seen_session_id: Optional[str] = None,
        canonical_bbox: Optional[BoundingBox] = None,
    ) -> None:
        """
        Deprecated: Direct state modification without an associated ChangeEvent audit record.
        Use record_state_transition(...) instead to guarantee atomic audit trails.
        """
        warnings.warn(
            "update_entity_state is deprecated; use record_state_transition to ensure auditable transitions.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.get_entity(entity_id):
            raise KeyError(f"Entity with ID {entity_id} does not exist.")

        updates = ["current_state = ?"]
        params: List[Any] = [new_state.value]

        if last_seen_session_id is not None:
            updates.append("last_seen_session_id = ?")
            params.append(last_seen_session_id)

        if canonical_bbox is not None:
            updates.append("canonical_bbox_json = ?")
            params.append(_bbox_to_json(canonical_bbox))

        params.append(entity_id)
        query = f"UPDATE entities SET {', '.join(updates)} WHERE entity_id = ?;"
        with self.conn:
            self.conn.execute(query, params)

    # -------------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------------

    def add_observation(self, obs: Observation) -> None:
        """Persists a new observation within an atomic transaction."""
        query = """
            INSERT INTO observations (
                observation_id, session_id, entity_id, timestamp, image_reference,
                raw_bbox_json, canonical_bbox_json, visibility_status,
                detector_confidence, identity_match_score, alignment_quality,
                appearance_descriptor_ref, raw_evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.conn:
            self.conn.execute(
                query,
                (
                    obs.observation_id,
                    obs.session_id,
                    obs.entity_id,
                    obs.timestamp,
                    obs.image_reference,
                    _bbox_to_json(obs.raw_bbox),
                    _bbox_to_json(obs.canonical_bbox),
                    obs.visibility_status.value,
                    obs.detector_confidence,
                    obs.identity_match_score,
                    obs.alignment_quality,
                    obs.appearance_descriptor_ref,
                    _dict_to_json(obs.raw_evidence),
                ),
            )

    def get_observation(self, observation_id: str) -> Optional[Observation]:
        """Retrieves a single observation by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM observations WHERE observation_id = ?;", (observation_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Observation(
            observation_id=row["observation_id"],
            session_id=row["session_id"],
            entity_id=row["entity_id"],
            timestamp=row["timestamp"],
            image_reference=row["image_reference"],
            raw_bbox=_json_to_bbox(row["raw_bbox_json"]),
            canonical_bbox=_json_to_bbox(row["canonical_bbox_json"]),
            visibility_status=VisibilityStatus(row["visibility_status"]),
            detector_confidence=row["detector_confidence"],
            identity_match_score=row["identity_match_score"],
            alignment_quality=row["alignment_quality"],
            appearance_descriptor_ref=row["appearance_descriptor_ref"],
            raw_evidence=_json_to_dict(row["raw_evidence_json"]),
        )

    def list_observations_for_session(self, session_id: str) -> List[Observation]:
        """Lists observations in a session with observation_id tie-breaker."""
        cursor = self.conn.execute(
            "SELECT * FROM observations WHERE session_id = ? ORDER BY timestamp ASC, observation_id ASC;",
            (session_id,),
        )
        return [
            Observation(
                observation_id=row["observation_id"],
                session_id=row["session_id"],
                entity_id=row["entity_id"],
                timestamp=row["timestamp"],
                image_reference=row["image_reference"],
                raw_bbox=_json_to_bbox(row["raw_bbox_json"]),
                canonical_bbox=_json_to_bbox(row["canonical_bbox_json"]),
                visibility_status=VisibilityStatus(row["visibility_status"]),
                detector_confidence=row["detector_confidence"],
                identity_match_score=row["identity_match_score"],
                alignment_quality=row["alignment_quality"],
                appearance_descriptor_ref=row["appearance_descriptor_ref"],
                raw_evidence=_json_to_dict(row["raw_evidence_json"]),
            )
            for row in cursor.fetchall()
        ]

    def list_observations_for_entity(self, entity_id: str) -> List[Observation]:
        """Lists observations associated with an entity with observation_id tie-breaker."""
        cursor = self.conn.execute(
            "SELECT * FROM observations WHERE entity_id = ? ORDER BY timestamp ASC, observation_id ASC;",
            (entity_id,),
        )
        return [
            Observation(
                observation_id=row["observation_id"],
                session_id=row["session_id"],
                entity_id=row["entity_id"],
                timestamp=row["timestamp"],
                image_reference=row["image_reference"],
                raw_bbox=_json_to_bbox(row["raw_bbox_json"]),
                canonical_bbox=_json_to_bbox(row["canonical_bbox_json"]),
                visibility_status=VisibilityStatus(row["visibility_status"]),
                detector_confidence=row["detector_confidence"],
                identity_match_score=row["identity_match_score"],
                alignment_quality=row["alignment_quality"],
                appearance_descriptor_ref=row["appearance_descriptor_ref"],
                raw_evidence=_json_to_dict(row["raw_evidence_json"]),
            )
            for row in cursor.fetchall()
        ]

    # -------------------------------------------------------------------------
    # Change Events
    # -------------------------------------------------------------------------

    def add_change_event(self, event: ChangeEvent) -> None:
        """Persists a change event within an atomic transaction."""
        query = """
            INSERT INTO change_events (
                event_id, entity_id, from_session_id, to_session_id, timestamp,
                prior_state, new_state, source, displacement_distance,
                supporting_evidence_json, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.conn:
            self.conn.execute(
                query,
                (
                    event.event_id,
                    event.entity_id,
                    event.from_session_id,
                    event.to_session_id,
                    event.timestamp,
                    event.prior_state.value,
                    event.new_state.value,
                    event.source,
                    event.displacement_distance,
                    _dict_to_json(event.supporting_evidence),
                    event.explanation,
                ),
            )

    def list_change_events_for_entity(self, entity_id: str) -> List[ChangeEvent]:
        """Retrieves events for an entity with event_id tie-breaker."""
        cursor = self.conn.execute(
            "SELECT * FROM change_events WHERE entity_id = ? ORDER BY timestamp ASC, event_id ASC;",
            (entity_id,),
        )
        return [
            ChangeEvent(
                event_id=row["event_id"],
                entity_id=row["entity_id"],
                from_session_id=row["from_session_id"],
                to_session_id=row["to_session_id"],
                timestamp=row["timestamp"],
                prior_state=EpistemicState(row["prior_state"]),
                new_state=EpistemicState(row["new_state"]),
                source=row["source"],
                displacement_distance=row["displacement_distance"],
                supporting_evidence=_json_to_dict(row["supporting_evidence_json"]) or {},
                explanation=row["explanation"] or "",
            )
            for row in cursor.fetchall()
        ]

    def list_change_events_for_session(self, session_id: str) -> List[ChangeEvent]:
        """Retrieves events for a session with event_id tie-breaker."""
        cursor = self.conn.execute(
            "SELECT * FROM change_events WHERE to_session_id = ? ORDER BY timestamp ASC, event_id ASC;",
            (session_id,),
        )
        return [
            ChangeEvent(
                event_id=row["event_id"],
                entity_id=row["entity_id"],
                from_session_id=row["from_session_id"],
                to_session_id=row["to_session_id"],
                timestamp=row["timestamp"],
                prior_state=EpistemicState(row["prior_state"]),
                new_state=EpistemicState(row["new_state"]),
                source=row["source"],
                displacement_distance=row["displacement_distance"],
                supporting_evidence=_json_to_dict(row["supporting_evidence_json"]) or {},
                explanation=row["explanation"] or "",
            )
            for row in cursor.fetchall()
        ]

    # -------------------------------------------------------------------------
    # Pairwise Baseline Results
    # -------------------------------------------------------------------------

    def add_pairwise_result(self, res: PairwiseBaselineResult) -> None:
        """Stores a pairwise baseline comparison result within an atomic transaction."""
        query = """
            INSERT INTO pairwise_baseline_results (
                result_id, reference_session_id, query_session_id, entity_id,
                predicted_change, confidence, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self.conn:
            self.conn.execute(
                query,
                (
                    res.result_id,
                    res.reference_session_id,
                    res.query_session_id,
                    res.entity_id,
                    res.predicted_change.value,
                    res.confidence,
                    _dict_to_json(res.details),
                ),
            )

    def list_pairwise_results(
        self,
        reference_session_id: Optional[str] = None,
        query_session_id: Optional[str] = None,
    ) -> List[PairwiseBaselineResult]:
        """Lists pairwise comparisons ordered by result_id."""
        conditions = []
        params = []
        if reference_session_id is not None:
            conditions.append("reference_session_id = ?")
            params.append(reference_session_id)
        if query_session_id is not None:
            conditions.append("query_session_id = ?")
            params.append(query_session_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM pairwise_baseline_results {where_clause} ORDER BY result_id ASC;"
        cursor = self.conn.execute(query, params)
        return [
            PairwiseBaselineResult(
                result_id=row["result_id"],
                reference_session_id=row["reference_session_id"],
                query_session_id=row["query_session_id"],
                entity_id=row["entity_id"],
                predicted_change=BaselineChangeType(row["predicted_change"]),
                confidence=row["confidence"],
                details=_json_to_dict(row["details_json"]),
            )
            for row in cursor.fetchall()
        ]

    # -------------------------------------------------------------------------
    # Raw Detections
    # -------------------------------------------------------------------------

    def add_raw_detection(self, det: RawDetection) -> None:
        """Stores a raw sensor detection independently of registration success."""
        query = """
            INSERT INTO raw_detections (
                detection_id, session_id, timestamp, label, confidence,
                raw_bbox_json, canonical_bbox_json, registration_status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.conn:
            self.conn.execute(
                query,
                (
                    det.detection_id,
                    det.session_id,
                    det.timestamp,
                    det.label,
                    det.confidence,
                    _bbox_to_json(det.raw_bbox),
                    _bbox_to_json(det.canonical_bbox),
                    det.registration_status,
                    _dict_to_json(det.metadata),
                ),
            )

    def add_raw_detections(self, dets: Sequence[RawDetection]) -> None:
        """Stores multiple raw detections atomically in a single transaction."""
        if not dets:
            return
        query = """
            INSERT INTO raw_detections (
                detection_id, session_id, timestamp, label, confidence,
                raw_bbox_json, canonical_bbox_json, registration_status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        rows = [
            (
                d.detection_id,
                d.session_id,
                d.timestamp,
                d.label,
                d.confidence,
                _bbox_to_json(d.raw_bbox),
                _bbox_to_json(d.canonical_bbox),
                d.registration_status,
                _dict_to_json(d.metadata),
            )
            for d in dets
        ]
        with self.conn:
            self.conn.executemany(query, rows)

    def list_raw_detections_for_session(self, session_id: str) -> List[RawDetection]:
        """Retrieves all raw sensor detections captured in a given session."""
        query = "SELECT * FROM raw_detections WHERE session_id = ? ORDER BY detection_id ASC;"
        cursor = self.conn.execute(query, (session_id,))
        return [
            RawDetection(
                detection_id=row["detection_id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                label=row["label"],
                confidence=row["confidence"],
                raw_bbox=_json_to_bbox(row["raw_bbox_json"]),
                canonical_bbox=_json_to_bbox(row["canonical_bbox_json"]),
                registration_status=row["registration_status"],
                metadata=_json_to_dict(row["metadata_json"]),
            )
            for row in cursor.fetchall()
        ]

    def get_raw_detection(self, detection_id: str) -> Optional[RawDetection]:
        """Retrieves a single raw detection by ID."""
        cursor = self.conn.execute("SELECT * FROM raw_detections WHERE detection_id = ?;", (detection_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return RawDetection(
            detection_id=row["detection_id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            label=row["label"],
            confidence=row["confidence"],
            raw_bbox=_json_to_bbox(row["raw_bbox_json"]),
            canonical_bbox=_json_to_bbox(row["canonical_bbox_json"]),
            registration_status=row["registration_status"],
            metadata=_json_to_dict(row["metadata_json"]),
        )
