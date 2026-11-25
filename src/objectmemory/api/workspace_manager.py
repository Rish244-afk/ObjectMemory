"""
Workspace Manager for ObjectMemory backend.

Coordinates:
1. Workspace lifecycle and filesystem storage under a configurable root directory.
2. Complete workspace isolation (independent SQLite DB, dedicated images/ dir).
3. Thread-safe execution using per-workspace mutual exclusion locks.
4. Durability across restarts: automatically re-establishes reference frames from persisted sessions.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from objectmemory.datasets.real_dataset import RealVisitFrame
from objectmemory.detection.detector import ObjectDetector, OpenCVDNNDetector
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.models import Detection, DetectorConfiguration
from objectmemory.domain.models import BoundingBox, normalize_iso_timestamp
from objectmemory.domain.states import EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner, RealVisitProcessingResult
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig

logger = logging.getLogger(__name__)


def _sanitize_slug(slug: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug.strip())
    if not cleaned:
        raise ValueError("Invalid workspace or session identifier.")
    return cleaned


class WorkspaceManager:
    """
    Manages isolated physical workspaces, image storage, and pipeline lifecycle.
    """

    def __init__(
        self,
        base_dir: Path,
        detector: Optional[ObjectDetector] = None,
        weights_path: Optional[Path] = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = Path(weights_path) if weights_path else Path("models/yolov8m.onnx")
        self._detector = detector
        self._detector_instances: Dict[str, ObjectDetector] = {}
        self._global_lock = threading.Lock()
        self._workspace_locks: Dict[str, threading.RLock] = {}
        self._active_runners: Dict[str, RealImagePipelineRunner] = {}

    def _get_workspace_lock(self, workspace_id: str) -> threading.RLock:
        with self._global_lock:
            if workspace_id not in self._workspace_locks:
                self._workspace_locks[workspace_id] = threading.RLock()
            return self._workspace_locks[workspace_id]

    def _get_detector(self, detector_name: Optional[str] = None) -> ObjectDetector:
        if self._detector is not None:
            return self._detector
        default_name = DetectorRegistry.get_default_name() or "yolov8m"
        key = (detector_name or default_name).strip().lower()
        with self._global_lock:
            if key not in self._detector_instances:
                if key in ("yolov8m", "yolov8") and self.weights_path.exists():
                    self._detector_instances[key] = DetectorRegistry.get(key, weights_path=str(self.weights_path))
                else:
                    self._detector_instances[key] = DetectorRegistry.get(key)
            return self._detector_instances[key]

    # ---------------------------------------------------------------------------
    # Directory paths
    # ---------------------------------------------------------------------------

    def _workspace_dir(self, workspace_id: str) -> Path:
        return self.base_dir / _sanitize_slug(workspace_id)

    def _meta_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "workspace.json"

    def _db_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "memory.db"

    def _images_dir(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "images"

    # ---------------------------------------------------------------------------
    # Workspace Lifecycle
    # ---------------------------------------------------------------------------

    def create_workspace(self, workspace_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        slug = _sanitize_slug(workspace_id)
        wdir = self._workspace_dir(slug)
        if wdir.exists() and (wdir / "workspace.json").exists():
            raise ValueError(f"Workspace '{slug}' already exists.")

        wdir.mkdir(parents=True, exist_ok=True)
        self._images_dir(slug).mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta = {
            "workspace_id": slug,
            "name": name.strip(),
            "description": description.strip() if description else "",
            "created_at": now_utc,
            "reference_session_id": None,
        }
        with open(self._meta_path(slug), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Initialize SQLite DB schema immediately
        with SQLiteMemoryStore(self._db_path(slug), check_same_thread=False):
            pass

        return meta

    def get_workspace_meta(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        slug = _sanitize_slug(workspace_id)
        mpath = self._meta_path(slug)
        if not mpath.exists():
            return None
        with open(mpath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_workspaces(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for item in sorted(self.base_dir.iterdir()):
            if item.is_dir() and (item / "workspace.json").exists():
                try:
                    summary = self.get_workspace_summary(item.name)
                    if summary:
                        summaries.append(summary)
                except Exception as exc:
                    logger.warning("Failed to read workspace summary for %s: %s", item.name, exc)
        return summaries

    def get_workspace_summary(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        meta = self.get_workspace_meta(workspace_id)
        if not meta:
            return None
        slug = meta["workspace_id"]
        db_path = self._db_path(slug)
        if not db_path.exists():
            return None

        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            sessions = store.list_sessions(order_asc=True)
            entities = store.list_entities()
            cursor = store.conn.execute("SELECT COUNT(*) FROM change_events;")
            events_count = cursor.fetchone()[0]

            last_session_ts = sessions[-1].timestamp if sessions else None
            ref_session_id = sessions[0].session_id if sessions else None

            counts: Dict[str, int] = {}
            for e in entities:
                st = e.current_state.value
                counts[st] = counts.get(st, 0) + 1

        return {
            "workspace_id": slug,
            "name": meta["name"],
            "description": meta.get("description", ""),
            "created_at": meta["created_at"],
            "total_sessions": len(sessions),
            "total_entities": len(entities),
            "total_events": events_count,
            "reference_session_id": ref_session_id,
            "last_session_timestamp": last_session_ts,
            "entity_state_counts": counts,
        }

    def delete_workspace(self, workspace_id: str) -> bool:
        import shutil
        import gc
        import time

        slug = _sanitize_slug(workspace_id)
        lock = self._get_workspace_lock(slug)
        with lock:
            if slug in self._active_runners:
                runner = self._active_runners.pop(slug)
                try:
                    if hasattr(runner, "store") and runner.store is not None:
                        runner.store.close()
                except Exception as e:
                    logger.warning(f"Error closing store for {slug}: {e}")

            wdir = self._workspace_dir(slug)
            if not wdir.exists():
                return False

            gc.collect()

            # Attempt deletion with retry for Windows file handle release
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    shutil.rmtree(wdir)
                    return True
                except PermissionError:
                    if attempt < max_attempts - 1:
                        time.sleep(0.1)
                        gc.collect()
                    else:
                        raise
            return False

    # ---------------------------------------------------------------------------
    # Runner Recovery & Persistence
    # ---------------------------------------------------------------------------

    def _get_or_create_runner(
        self,
        workspace_id: str,
        detector_name: Optional[str] = None,
    ) -> RealImagePipelineRunner:
        slug = _sanitize_slug(workspace_id)
        if slug in self._active_runners:
            runner = self._active_runners[slug]
            if detector_name is not None:
                runner.detector = self._get_detector(detector_name)
            return runner

        detector = self._get_detector(detector_name)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise FileNotFoundError(f"Workspace database not found for '{slug}'.")

        store = SQLiteMemoryStore(db_path, check_same_thread=False)
        runner = RealImagePipelineRunner(
            store=store,
            detector=detector,
            registrar=PlanarRegistrar(),
            temporal_config=TemporalEngineConfig(),
        )

        # Durability recovery: Restore reference frame from existing sessions
        sessions = store.list_sessions(order_asc=True)
        if sessions:
            first_session = sessions[0]
            ref_path = Path(first_session.image_path)
            if not ref_path.is_absolute():
                ref_path = self._images_dir(slug) / ref_path.name
            if ref_path.exists():
                ref_img = cv2.imread(str(ref_path))
                if ref_img is not None:
                    runner.ref_image = ref_img
                    runner.ref_visit_id = first_session.session_id
                    runner.processed_visits_count = len(sessions)
                    logger.info(
                        "Restored reference frame for workspace '%s' from %s (total %d visits)",
                        slug,
                        ref_path.name,
                        len(sessions),
                    )

        self._active_runners[slug] = runner
        return runner

    # ---------------------------------------------------------------------------
    # Image Submission & Observation Processing
    # ---------------------------------------------------------------------------

    def process_image_observation(
        self,
        workspace_id: str,
        image_bytes: bytes,
        session_name: Optional[str] = None,
        timestamp: Optional[str] = None,
        notes: Optional[str] = None,
        confidence_threshold: float = 0.25,
        detector_name: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Accepts raw image bytes, persists the image safely, and processes the visit.
        """
        slug = _sanitize_slug(workspace_id)
        if not self._meta_path(slug).exists():
            raise KeyError(f"Workspace '{slug}' does not exist.")

        lock = self._get_workspace_lock(slug)
        with lock:
            # 1. Decode image bytes
            arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image. Ensure valid PNG or JPEG data.")

            h_img, w_img = img.shape[:2]
            if h_img < 32 or w_img < 32:
                raise ValueError("Image dimensions are too small (minimum 32x32).")

            # 2. Get runner
            runner = self._get_or_create_runner(slug, detector_name=detector_name)
            is_first_session = (runner.ref_image is None or runner.processed_visits_count == 0)

            # 3. Format timestamp and session_id
            if timestamp is None or not timestamp.strip():
                ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                ts_utc = normalize_iso_timestamp(timestamp)

            session_idx = runner.processed_visits_count + 1
            session_id = f"visit_{session_idx:02d}"
            name = session_name.strip() if session_name and session_name.strip() else f"Visit {session_idx:02d}"

            # 4. Save image to workspace disk
            filename = f"{session_id}.png"
            image_disk_path = self._images_dir(slug) / filename
            cv2.imwrite(str(image_disk_path), img)

            # 5. Build RealVisitFrame
            visit_frame = RealVisitFrame(
                visit_id=session_id,
                session_name=name,
                timestamp=ts_utc,
                image_path=image_disk_path,
                metadata={"notes": notes, "original_filename": filename},
            )

            # 6. Execute pipeline
            if target_classes:
                det_cfg = DetectorConfiguration(
                    confidence_threshold=confidence_threshold,
                    target_classes=target_classes,
                )
                result: RealVisitProcessingResult = runner.process_visit(
                    visit_frame,
                    detector_config=det_cfg,
                )
            else:
                result: RealVisitProcessingResult = runner.process_visit(
                    visit_frame,
                    confidence_threshold=confidence_threshold,
                )

            # 7. Update workspace meta if this was the reference frame
            if is_first_session:
                meta = self.get_workspace_meta(slug)
                if meta:
                    meta["reference_session_id"] = session_id
                    with open(self._meta_path(slug), "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

            # 8. Query updated entities and events for response payload
            entities = runner.store.list_entities()
            new_events = runner.store.list_change_events_for_session(session_id)

            # Format detection summaries
            detections_out = []
            for det in result.detection_batch.detections:
                raw_b = {
                    "x_min": det.raw_bbox.x_min,
                    "y_min": det.raw_bbox.y_min,
                    "x_max": det.raw_bbox.x_max,
                    "y_max": det.raw_bbox.y_max,
                }
                canon_b = None
                if det.canonical_bbox is not None:
                    canon_b = {
                        "x_min": det.canonical_bbox.x_min,
                        "y_min": det.canonical_bbox.y_min,
                        "x_max": det.canonical_bbox.x_max,
                        "y_max": det.canonical_bbox.y_max,
                    }
                is_clipped = runner.engine._is_bbox_clipped(det.raw_bbox, (w_img, h_img))
                detections_out.append({
                    "label": det.label,
                    "confidence": det.confidence,
                    "raw_bbox": raw_b,
                    "canonical_bbox": canon_b,
                    "is_clipped": is_clipped,
                })

            # Format belief updates
            updates_out = []
            for u in result.temporal_report.entity_updates:
                updates_out.append({
                    "entity_id": u.entity_id,
                    "label": u.label,
                    "prior_state": u.prior_state.value,
                    "new_state": u.new_state.value,
                    "transition_occurred": u.transition_occurred,
                    "visibility_status": u.visibility_status.value,
                    "explanation": u.explanation,
                    "event_id": u.event_id,
                    "displacement_px": (
                        u.supporting_evidence.get("displacement_px")
                        if isinstance(u.supporting_evidence, dict)
                        else None
                    ),
                })

            # Format change events
            events_out = []
            for ev in new_events:
                events_out.append({
                    "event_id": ev.event_id,
                    "entity_id": ev.entity_id,
                    "from_session_id": ev.from_session_id,
                    "to_session_id": ev.to_session_id,
                    "timestamp": ev.timestamp,
                    "prior_state": ev.prior_state.value,
                    "new_state": ev.new_state.value,
                    "source": ev.source,
                    "displacement_distance": ev.displacement_distance,
                    "explanation": ev.explanation,
                    "supporting_evidence": ev.supporting_evidence,
                })

            reg = result.registration
            reg_out = {
                "success": reg.success,
                "status": reg.status.value,
                "method_used": reg.method_used,
                "inlier_ratio": float(reg.inlier_ratio) if reg.inlier_ratio is not None else 0.0,
                "mean_reprojection_error": (
                    float(reg.reprojection_stats.mean)
                    if reg.reprojection_stats and reg.reprojection_stats.mean is not None
                    else None
                ),
                "inliers_count": reg.num_inliers,
                "ref_size": list(reg.ref_size),
                "query_size": list(reg.query_size),
                "failure_reason": reg.failure_reason,
                "selected_keyframe_id": reg.selected_keyframe_id,
                "is_direct_to_canonical": reg.is_direct_to_canonical,
                "spatial_coverage": reg.spatial_coverage,
                "decision_status": reg.decision_status or ("ACCEPTED" if reg.success else "REJECTED"),
                "decision_reason": reg.decision_reason or reg.failure_reason,
                "is_promoted": reg.is_promoted,
            }

            return {
                "session_id": session_id,
                "session_name": name,
                "timestamp": ts_utc,
                "image_url": f"/api/workspaces/{slug}/images/{filename}",
                "is_reference_frame": is_first_session,
                "registration": reg_out,
                "detections": detections_out,
                "belief_updates": updates_out,
                "events": events_out,
                "total_active_entities": len(entities),
                "latency_breakdown_ms": result.latency_breakdown_ms,
                "total_latency_ms": result.total_latency_ms,
            }

    # ---------------------------------------------------------------------------
    # Query Functions
    # ---------------------------------------------------------------------------

    def list_entities(self, workspace_id: str, state: Optional[str] = None) -> List[Dict[str, Any]]:
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise KeyError(f"Workspace '{slug}' not found.")

        target_state = EpistemicState(state) if state else None
        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            entities = store.list_entities(state=target_state)
            out = []
            for e in entities:
                canon = None
                if e.canonical_bbox:
                    canon = {
                        "x_min": e.canonical_bbox.x_min,
                        "y_min": e.canonical_bbox.y_min,
                        "x_max": e.canonical_bbox.x_max,
                        "y_max": e.canonical_bbox.y_max,
                    }
                out.append({
                    "entity_id": e.entity_id,
                    "label": e.label,
                    "created_at": e.created_at,
                    "current_state": e.current_state.value,
                    "canonical_bbox": canon,
                    "last_seen_session_id": e.last_seen_session_id,
                    "metadata": e.metadata,
                })
            return out

    def get_entity_detail(self, workspace_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            return None

        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            ent = store.get_entity(entity_id)
            if not ent:
                return None

            canon = None
            if ent.canonical_bbox:
                canon = {
                    "x_min": ent.canonical_bbox.x_min,
                    "y_min": ent.canonical_bbox.y_min,
                    "x_max": ent.canonical_bbox.x_max,
                    "y_max": ent.canonical_bbox.y_max,
                }
            entity_summary = {
                "entity_id": ent.entity_id,
                "label": ent.label,
                "created_at": ent.created_at,
                "current_state": ent.current_state.value,
                "canonical_bbox": canon,
                "last_seen_session_id": ent.last_seen_session_id,
                "metadata": ent.metadata,
            }

            observations = store.list_observations_for_entity(entity_id)
            obs_out = []
            for o in observations:
                lbl = (o.raw_evidence or {}).get("label") or ent.label
                obs_out.append({
                    "observation_id": o.observation_id,
                    "session_id": o.session_id,
                    "entity_id": o.entity_id,
                    "label": lbl,
                    "timestamp": o.timestamp,
                    "image_reference": o.image_reference,
                    "raw_bbox": (
                        {"x_min": o.raw_bbox.x_min, "y_min": o.raw_bbox.y_min, "x_max": o.raw_bbox.x_max, "y_max": o.raw_bbox.y_max}
                        if o.raw_bbox else None
                    ),
                    "canonical_bbox": (
                        {"x_min": o.canonical_bbox.x_min, "y_min": o.canonical_bbox.y_min, "x_max": o.canonical_bbox.x_max, "y_max": o.canonical_bbox.y_max}
                        if o.canonical_bbox else None
                    ),
                    "visibility_status": o.visibility_status.value,
                    "detector_confidence": o.detector_confidence,
                    "identity_match_score": o.identity_match_score,
                    "alignment_quality": o.alignment_quality,
                    "raw_evidence": o.raw_evidence,
                })

            events = store.list_change_events_for_entity(entity_id)
            ev_out = []
            for ev in events:
                ev_out.append({
                    "event_id": ev.event_id,
                    "entity_id": ev.entity_id,
                    "from_session_id": ev.from_session_id,
                    "to_session_id": ev.to_session_id,
                    "timestamp": ev.timestamp,
                    "prior_state": ev.prior_state.value,
                    "new_state": ev.new_state.value,
                    "source": ev.source,
                    "displacement_distance": ev.displacement_distance,
                    "explanation": ev.explanation,
                    "supporting_evidence": ev.supporting_evidence,
                })

            return {
                "entity": entity_summary,
                "observations": obs_out,
                "change_events": ev_out,
            }

    def list_sessions(self, workspace_id: str) -> List[Dict[str, Any]]:
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise KeyError(f"Workspace '{slug}' not found.")

        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            sessions = store.list_sessions(order_asc=True)
            out = []
            for s in sessions:
                obs_count = len(store.list_observations_for_session(s.session_id))
                evs_count = len(store.list_change_events_for_session(s.session_id))
                raw_count = len(store.list_raw_detections_for_session(s.session_id))
                img_name = Path(s.image_path).name
                out.append({
                    "session_id": s.session_id,
                    "session_name": s.session_name,
                    "timestamp": s.timestamp,
                    "image_path": s.image_path,
                    "image_url": f"/api/workspaces/{slug}/images/{img_name}",
                    "notes": s.notes,
                    "observations_count": obs_count,
                    "events_count": evs_count,
                    "raw_detections_count": raw_count,
                })
            return out

    def get_session_detail(self, workspace_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            return None

        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            sess = store.get_session(session_id)
            if not sess:
                return None

            obs = store.list_observations_for_session(session_id)
            evs = store.list_change_events_for_session(session_id)
            raw_dets = store.list_raw_detections_for_session(session_id)
            img_name = Path(sess.image_path).name

            session_summary = {
                "session_id": sess.session_id,
                "session_name": sess.session_name,
                "timestamp": sess.timestamp,
                "image_path": sess.image_path,
                "image_url": f"/api/workspaces/{slug}/images/{img_name}",
                "notes": sess.notes,
                "observations_count": len(obs),
                "events_count": len(evs),
                "raw_detections_count": len(raw_dets),
            }

            obs_out = []
            entity_label_cache: Dict[str, str] = {}
            for o in obs:
                lbl = (o.raw_evidence or {}).get("label")
                if not lbl and o.entity_id:
                    if o.entity_id not in entity_label_cache:
                        e = store.get_entity(o.entity_id)
                        entity_label_cache[o.entity_id] = e.label if e else ""
                    lbl = entity_label_cache[o.entity_id] or None

                obs_out.append({
                    "observation_id": o.observation_id,
                    "session_id": o.session_id,
                    "entity_id": o.entity_id,
                    "label": lbl,
                    "timestamp": o.timestamp,
                    "image_reference": o.image_reference,
                    "raw_bbox": (
                        {"x_min": o.raw_bbox.x_min, "y_min": o.raw_bbox.y_min, "x_max": o.raw_bbox.x_max, "y_max": o.raw_bbox.y_max}
                        if o.raw_bbox else None
                    ),
                    "canonical_bbox": (
                        {"x_min": o.canonical_bbox.x_min, "y_min": o.canonical_bbox.y_min, "x_max": o.canonical_bbox.x_max, "y_max": o.canonical_bbox.y_max}
                        if o.canonical_bbox else None
                    ),
                    "visibility_status": o.visibility_status.value,
                    "detector_confidence": o.detector_confidence,
                    "identity_match_score": o.identity_match_score,
                    "alignment_quality": o.alignment_quality,
                    "raw_evidence": o.raw_evidence,
                })

            ev_out = []
            for ev in evs:
                ev_out.append({
                    "event_id": ev.event_id,
                    "entity_id": ev.entity_id,
                    "from_session_id": ev.from_session_id,
                    "to_session_id": ev.to_session_id,
                    "timestamp": ev.timestamp,
                    "prior_state": ev.prior_state.value,
                    "new_state": ev.new_state.value,
                    "source": ev.source,
                    "displacement_distance": ev.displacement_distance,
                    "explanation": ev.explanation,
                    "supporting_evidence": ev.supporting_evidence,
                })

            raw_out = []
            for rd in raw_dets:
                raw_out.append({
                    "detection_id": rd.detection_id,
                    "session_id": rd.session_id,
                    "timestamp": rd.timestamp,
                    "label": rd.label,
                    "confidence": rd.confidence,
                    "raw_bbox": {
                        "x_min": rd.raw_bbox.x_min,
                        "y_min": rd.raw_bbox.y_min,
                        "x_max": rd.raw_bbox.x_max,
                        "y_max": rd.raw_bbox.y_max,
                    },
                    "canonical_bbox": (
                        {
                            "x_min": rd.canonical_bbox.x_min,
                            "y_min": rd.canonical_bbox.y_min,
                            "x_max": rd.canonical_bbox.x_max,
                            "y_max": rd.canonical_bbox.y_max,
                        }
                        if rd.canonical_bbox else None
                    ),
                    "registration_status": rd.registration_status,
                    "metadata": rd.metadata,
                })

            reg_data = (sess.metadata or {}).get("registration")
            reg_out = None
            if reg_data:
                reg_out = {
                    "success": bool(reg_data.get("success", False)),
                    "status": str(reg_data.get("status", "FAILED")),
                    "method_used": str(reg_data.get("method_used", "UNKNOWN")),
                    "inlier_ratio": float(reg_data.get("inlier_ratio", 0.0) or 0.0),
                    "mean_reprojection_error": reg_data.get("mean_reprojection_error"),
                    "inliers_count": int(reg_data.get("inliers_count", 0)),
                    "ref_size": list(reg_data.get("ref_size", [0, 0])),
                    "query_size": list(reg_data.get("query_size", [0, 0])),
                    "failure_reason": reg_data.get("failure_reason"),
                    "selected_keyframe_id": reg_data.get("selected_keyframe_id"),
                    "is_direct_to_canonical": bool(reg_data.get("is_direct_to_canonical", True)),
                    "spatial_coverage": reg_data.get("spatial_coverage"),
                    "decision_status": reg_data.get("decision_status"),
                    "decision_reason": reg_data.get("decision_reason"),
                    "is_promoted": bool(reg_data.get("is_promoted", False)),
                }

            return {
                "session": session_summary,
                "observations": obs_out,
                "events": ev_out,
                "raw_detections": raw_out,
                "registration": reg_out,
            }

    def list_change_events(self, workspace_id: str) -> List[Dict[str, Any]]:
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise KeyError(f"Workspace '{slug}' not found.")

        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            cursor = store.conn.execute("SELECT * FROM change_events ORDER BY timestamp ASC, event_id ASC;")
            rows = cursor.fetchall()
            out = []
            for row in rows:
                ev_supp = json.loads(row["supporting_evidence_json"]) if row["supporting_evidence_json"] else {}
                out.append({
                    "event_id": row["event_id"],
                    "entity_id": row["entity_id"],
                    "from_session_id": row["from_session_id"],
                    "to_session_id": row["to_session_id"],
                    "timestamp": row["timestamp"],
                    "prior_state": row["prior_state"],
                    "new_state": row["new_state"],
                    "source": row["source"],
                    "displacement_distance": row["displacement_distance"],
                    "explanation": row["explanation"] or "",
                    "supporting_evidence": ev_supp,
                })
            return out

    def get_diagnostics(self, workspace_id: str) -> Dict[str, Any]:
        slug = _sanitize_slug(workspace_id)
        summary = self.get_workspace_summary(slug)
        if not summary:
            raise KeyError(f"Workspace '{slug}' not found.")

        active_runner = self._active_runners.get(slug)
        detector = active_runner.detector if active_runner is not None else self._get_detector()
        model_info = detector.model_info()

        # Registration health from latest session observations
        db_path = self._db_path(slug)
        reg_health = {"status": "NO_OBSERVATIONS", "mean_inlier_ratio": None}
        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            cursor = store.conn.execute(
                "SELECT alignment_quality FROM observations WHERE alignment_quality IS NOT NULL ORDER BY timestamp DESC LIMIT 20;"
            )
            qualities = [r[0] for r in cursor.fetchall()]
            if qualities:
                reg_health = {
                    "status": "HEALTHY" if np.mean(qualities) >= 0.20 else "DEGRADED",
                    "recent_alignment_quality_mean": float(np.mean(qualities)),
                    "samples_evaluated": len(qualities),
                }

        return {
            "workspace_id": slug,
            "detector_info": model_info,
            "total_sessions": summary["total_sessions"],
            "total_entities": summary["total_entities"],
            "total_events": summary["total_events"],
            "entity_state_distribution": summary["entity_state_counts"],
            "registration_health": reg_health,
        }

    def compare_sessions(
        self,
        workspace_id: str,
        ref_session_id: Optional[str] = None,
        query_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise KeyError(f"Workspace '{slug}' not found.")

        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            sessions = store.list_sessions(order_asc=True)
            if len(sessions) < 2:
                raise ValueError("Comparison requires at least 2 sessions in the workspace.")

            # Default to comparing session 1 with latest session
            ref_id = ref_session_id or sessions[0].session_id
            query_id = query_session_id or sessions[-1].session_id

            ref_sess = store.get_session(ref_id)
            query_sess = store.get_session(query_id)
            if not ref_sess or not query_sess:
                raise KeyError(f"One or both sessions not found: {ref_id}, {query_id}")

            # Get observations in ref and query sessions
            ref_obs = {o.entity_id: o for o in store.list_observations_for_session(ref_id) if o.entity_id}
            query_obs = {o.entity_id: o for o in store.list_observations_for_session(query_id) if o.entity_id}
            query_events = {e.entity_id: e for e in store.list_change_events_for_session(query_id)}

            # Compute each entity's state as of query_id (instead of latest workspace state)
            session_order = {s.session_id: i for i, s in enumerate(sessions)}
            q_order = session_order.get(query_id, len(sessions) - 1)
            state_at_query: Dict[str, str] = {}
            cur_ev = store.conn.execute(
                "SELECT entity_id, to_session_id, new_state FROM change_events ORDER BY timestamp ASC, event_id ASC;"
            )
            for row in cur_ev.fetchall():
                if session_order.get(row["to_session_id"], -1) <= q_order:
                    state_at_query[row["entity_id"]] = row["new_state"]

            all_entity_ids = sorted(list(set(ref_obs.keys()).union(query_obs.keys())))
            items = []
            for eid in all_entity_ids:
                ent = store.get_entity(eid)
                label = ent.label if ent else "unknown"
                r_o = ref_obs.get(eid)
                q_o = query_obs.get(eid)
                ev = query_events.get(eid)
                fallback_state = state_at_query.get(eid) or (ent.current_state.value if ent else "CONFIRMED_PRESENT")

                r_box = (
                    {"x_min": r_o.canonical_bbox.x_min, "y_min": r_o.canonical_bbox.y_min, "x_max": r_o.canonical_bbox.x_max, "y_max": r_o.canonical_bbox.y_max}
                    if r_o and r_o.canonical_bbox else None
                )
                q_box = (
                    {"x_min": q_o.canonical_bbox.x_min, "y_min": q_o.canonical_bbox.y_min, "x_max": q_o.canonical_bbox.x_max, "y_max": q_o.canonical_bbox.y_max}
                    if q_o and q_o.canonical_bbox else None
                )

                if eid not in ref_obs and eid in query_obs:
                    status_change = "ADDED"
                    prior_st = "NONE"
                    new_st = fallback_state
                    expl = "Entity first observed in query session."
                elif eid in ref_obs and eid not in query_obs:
                    status_change = "MISSING_IN_QUERY"
                    prior_st = "CONFIRMED_PRESENT"
                    new_st = fallback_state
                    expl = "Entity not detected in query session."
                elif ev:
                    status_change = ev.new_state.value
                    prior_st = ev.prior_state.value
                    new_st = ev.new_state.value
                    expl = ev.explanation
                else:
                    status_change = "UNCHANGED"
                    prior_st = fallback_state
                    new_st = prior_st
                    expl = "No state transition triggered."

                disp = ev.displacement_distance if ev else None

                items.append({
                    "entity_id": eid,
                    "label": label,
                    "status_change": status_change,
                    "prior_state": prior_st,
                    "new_state": new_st,
                    "displacement_px": disp,
                    "ref_bbox": r_box,
                    "query_bbox": q_box,
                    "explanation": expl,
                })

            # Check alignment quality from query observations
            qualities = [o.alignment_quality for o in query_obs.values() if o.alignment_quality is not None]
            inlier_ratio = float(np.mean(qualities)) if qualities else 1.0

            return {
                "workspace_id": slug,
                "reference_session_id": ref_id,
                "query_session_id": query_id,
                "registration_status": "SUCCESS" if inlier_ratio >= 0.20 else "DEGRADED",
                "registration_inlier_ratio": inlier_ratio,
                "items": items,
            }

    def get_image_file_path(self, workspace_id: str, filename: str) -> Optional[Path]:
        slug = _sanitize_slug(workspace_id)
        safe_fn = Path(filename).name
        file_path = self._images_dir(slug) / safe_fn
        if file_path.exists() and file_path.is_file():
            return file_path
        return None

    def export_workspace(self, workspace_id: str) -> Dict[str, Any]:
        """
        Exports the complete workspace state, session history, entity timelines, and diagnostics
        as a structured dictionary suitable for experimental archiving and validation.
        """
        slug = _sanitize_slug(workspace_id)
        with self._get_workspace_lock(slug):
            meta = self.get_workspace_meta(slug)
            if not meta:
                raise KeyError(f"Workspace '{workspace_id}' not found.")

            summary = self.get_workspace_summary(slug)
            if not summary:
                raise KeyError(f"Workspace '{workspace_id}' summary unavailable.")

            sessions = self.list_sessions(slug)
            session_details = []
            for s in sessions:
                det = self.get_session_detail(slug, s["session_id"])
                if det:
                    session_details.append(det)

            entities = self.list_entities(slug)
            entity_details = []
            for e in entities:
                det = self.get_entity_detail(slug, e["entity_id"])
                if det:
                    entity_details.append(det)

            events = self.list_change_events(slug)
            diagnostics = self.get_diagnostics(slug)

            return {
                "workspace": summary,
                "entity_state_counts": summary.get("entity_state_counts", {}),
                "sessions": session_details,
                "entities": entity_details,
                "events": events,
                "diagnostics": diagnostics,
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

    def process_image_sequence(
        self,
        workspace_id: str,
        files: List[Tuple[str, bytes]],
        confidence_threshold: float = 0.25,
        detector_name: Optional[str] = None,
        prompts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes a sequence of image files chronologically in sorted filename/temporal order.
        Each image is processed through the common temporal memory pipeline.
        """
        slug = _sanitize_slug(workspace_id)
        existing_sessions = self.list_sessions(slug)
        start_idx = len(existing_sessions) + 1
        # Naturally sort by filename
        sorted_files = sorted(files, key=lambda x: x[0])
        results = []
        for idx, (fname, contents) in enumerate(sorted_files, start=start_idx):
            visit_stem = Path(fname).stem
            session_name = f"Visit {idx:02d} ({visit_stem})"
            res = self.process_image_observation(
                workspace_id=slug,
                image_bytes=contents,
                session_name=session_name,
                confidence_threshold=confidence_threshold,
                detector_name=detector_name,
                target_classes=prompts,
            )
            results.append(res)
        return results
