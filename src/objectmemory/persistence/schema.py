"""
SQLite schema definitions, constraints, and migration logic for ObjectMemory.

Enforces:
1. Foreign key constraints across sessions, entities, observations, and events.
2. ON DELETE RESTRICT on historical tables (change_events, observations, pairwise_baseline_results)
   to ensure historical audit trails and raw evidence cannot be silently erased.
3. Database-level CHECK constraints for finite unit-interval confidence scores.
4. Schema migrations via PRAGMA user_version.
"""

import sqlite3

CURRENT_SCHEMA_VERSION = 3

DDL_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    session_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    image_path TEXT NOT NULL,
    notes TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    canonical_bbox_json TEXT,
    current_state TEXT NOT NULL,
    last_seen_session_id TEXT,
    metadata_json TEXT,
    FOREIGN KEY (last_seen_session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    entity_id TEXT,
    timestamp TEXT NOT NULL,
    image_reference TEXT NOT NULL,
    raw_bbox_json TEXT,
    canonical_bbox_json TEXT,
    visibility_status TEXT NOT NULL,
    detector_confidence REAL CHECK (
        detector_confidence IS NULL OR (
            detector_confidence >= 0.0 AND detector_confidence <= 1.0 AND detector_confidence = detector_confidence
        )
    ),
    identity_match_score REAL CHECK (
        identity_match_score IS NULL OR (
            identity_match_score >= 0.0 AND identity_match_score <= 1.0 AND identity_match_score = identity_match_score
        )
    ),
    alignment_quality REAL CHECK (
        alignment_quality IS NULL OR (
            alignment_quality >= 0.0 AND alignment_quality <= 1.0 AND alignment_quality = alignment_quality
        )
    ),
    appearance_descriptor_ref TEXT,
    raw_evidence_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS raw_detections (
    detection_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL CHECK (
        confidence IS NULL OR (
            confidence >= 0.0 AND confidence <= 1.0 AND confidence = confidence
        )
    ),
    raw_bbox_json TEXT NOT NULL,
    canonical_bbox_json TEXT,
    registration_status TEXT NOT NULL DEFAULT 'UNREGISTERED',
    metadata_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS change_events (
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
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT,
    FOREIGN KEY (from_session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT,
    FOREIGN KEY (to_session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS pairwise_baseline_results (
    result_id TEXT PRIMARY KEY,
    reference_session_id TEXT NOT NULL,
    query_session_id TEXT NOT NULL,
    entity_id TEXT,
    predicted_change TEXT NOT NULL,
    confidence REAL CHECK (
        confidence IS NULL OR (
            confidence >= 0.0 AND confidence <= 1.0 AND confidence = confidence
        )
    ),
    details_json TEXT,
    FOREIGN KEY (reference_session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT,
    FOREIGN KEY (query_session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);
CREATE INDEX IF NOT EXISTS idx_entities_state ON entities(current_state);
CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);
CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);
CREATE INDEX IF NOT EXISTS idx_raw_detections_session ON raw_detections(session_id);
CREATE INDEX IF NOT EXISTS idx_change_events_entity ON change_events(entity_id);
CREATE INDEX IF NOT EXISTS idx_change_events_to_session ON change_events(to_session_id);
CREATE INDEX IF NOT EXISTS idx_change_events_source ON change_events(source);
CREATE INDEX IF NOT EXISTS idx_pairwise_ref_query ON pairwise_baseline_results(reference_session_id, query_session_id);
"""


def apply_migrations(conn: sqlite3.Connection) -> None:
    """
    Applies incremental schema migrations safely.
    Handles existing databases that had CASCADE foreign keys or older schema versions.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA user_version;")
    current_ver = cursor.fetchone()[0]

    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='change_events';")
    has_change_events = cursor.fetchone() is not None

    if not has_change_events:
        # Fresh database: execute full v2 DDL
        conn.executescript(DDL_SCHEMA)
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION};")
        conn.commit()
        return

    if current_ver < 2:
        # Migrate from v1 to v2:
        # Recreate change_events, observations, and pairwise_baseline_results with ON DELETE RESTRICT
        # and CHECK constraints.
        conn.execute("PRAGMA foreign_keys = OFF;")
        
        # 1. Migrate change_events
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrating_change_events (
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
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT,
                FOREIGN KEY (from_session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT,
                FOREIGN KEY (to_session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT
            );
        """)
        conn.execute("INSERT INTO _migrating_change_events SELECT * FROM change_events;")
        conn.execute("DROP TABLE change_events;")
        conn.execute("ALTER TABLE _migrating_change_events RENAME TO change_events;")

        # 2. Migrate observations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrating_observations (
                observation_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                entity_id TEXT,
                timestamp TEXT NOT NULL,
                image_reference TEXT NOT NULL,
                raw_bbox_json TEXT,
                canonical_bbox_json TEXT,
                visibility_status TEXT NOT NULL,
                detector_confidence REAL CHECK (
                    detector_confidence IS NULL OR (
                        detector_confidence >= 0.0 AND detector_confidence <= 1.0 AND detector_confidence = detector_confidence
                    )
                ),
                identity_match_score REAL CHECK (
                    identity_match_score IS NULL OR (
                        identity_match_score >= 0.0 AND identity_match_score <= 1.0 AND identity_match_score = identity_match_score
                    )
                ),
                alignment_quality REAL CHECK (
                    alignment_quality IS NULL OR (
                        alignment_quality >= 0.0 AND alignment_quality <= 1.0 AND alignment_quality = alignment_quality
                    )
                ),
                appearance_descriptor_ref TEXT,
                raw_evidence_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE SET NULL
            );
        """)
        conn.execute("INSERT INTO _migrating_observations SELECT * FROM observations;")
        conn.execute("DROP TABLE observations;")
        conn.execute("ALTER TABLE _migrating_observations RENAME TO observations;")

        # 3. Migrate pairwise_baseline_results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrating_pairwise (
                result_id TEXT PRIMARY KEY,
                reference_session_id TEXT NOT NULL,
                query_session_id TEXT NOT NULL,
                entity_id TEXT,
                predicted_change TEXT NOT NULL,
                confidence REAL CHECK (
                    confidence IS NULL OR (
                        confidence >= 0.0 AND confidence <= 1.0 AND confidence = confidence
                    )
                ),
                details_json TEXT,
                FOREIGN KEY (reference_session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT,
                FOREIGN KEY (query_session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE SET NULL
            );
        """)
        conn.execute("INSERT INTO _migrating_pairwise SELECT * FROM pairwise_baseline_results;")
        conn.execute("DROP TABLE pairwise_baseline_results;")
        conn.execute("ALTER TABLE _migrating_pairwise RENAME TO pairwise_baseline_results;")

        # Recreate indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_state ON entities(current_state);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_change_events_entity ON change_events(entity_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_change_events_to_session ON change_events(to_session_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_change_events_source ON change_events(source);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pairwise_ref_query ON pairwise_baseline_results(reference_session_id, query_session_id);")

        conn.execute("PRAGMA user_version = 2;")
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON;")

        # Validate that existing records respect integrity
        cursor.execute("PRAGMA foreign_key_check;")
        violations = cursor.fetchall()
        if violations:
            raise RuntimeError(f"Foreign key violations detected after schema migration: {violations}")

    # Check and migrate to v3: raw_detections table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_detections';")
    has_raw_detections = cursor.fetchone() is not None

    if not has_raw_detections or current_ver < 3:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_detections (
                detection_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL CHECK (
                    confidence IS NULL OR (
                        confidence >= 0.0 AND confidence <= 1.0 AND confidence = confidence
                    )
                ),
                raw_bbox_json TEXT NOT NULL,
                canonical_bbox_json TEXT,
                registration_status TEXT NOT NULL DEFAULT 'UNREGISTERED',
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_detections_session ON raw_detections(session_id);")
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION};")
        conn.commit()
