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
                # Evict other non-default heavy models when switching to avoid holding multiple large ONNX sessions
                for old_key in list(self._detector_instances.keys()):
                    if old_key != default_name and old_key != key:
                        old_inst = self._detector_instances.pop(old_key)
                        if hasattr(old_inst, "unload"):
                            try:
                                old_inst.unload()
                            except Exception:
                                pass
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

            # 6. Execute pipeline (normal frozen threshold)
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

            # 6b. Record raw label provenance & run evidence-gated sub-threshold recovery if needed
            self._sync_entity_raw_label_metadata(runner.store, session_id)
            if not is_first_session and not target_classes:
                self._attempt_subthreshold_recovery(
                    runner=runner,
                    img=img,
                    session_id=session_id,
                    ts_utc=ts_utc,
                    reg=result.registration,
                    accepted_detections=result.detection_batch.detections,
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

    def _format_entity_summary(self, e: Any) -> Dict[str, Any]:
        meta = dict(e.metadata or {})
        canon = None
        if e.canonical_bbox:
            canon = {
                "x_min": e.canonical_bbox.x_min,
                "y_min": e.canonical_bbox.y_min,
                "x_max": e.canonical_bbox.x_max,
                "y_max": e.canonical_bbox.y_max,
            }
        raw_labels = list(meta.get("raw_predicted_labels") or [])
        if not raw_labels:
            raw_labels = [meta.get("original_raw_label", e.label)]
        raw_label = meta.get("original_raw_label") or raw_labels[-1]
        raw_conf = meta.get("original_raw_confidence")
        if raw_conf is None:
            raw_conf = meta.get("initial_confidence")
        confirmed = meta.get("confirmed_label")
        provenance = meta.get("label_provenance") or ("USER_CONFIRMED" if confirmed else "MODEL_PREDICTED")
        confirmed_at = meta.get("label_confirmed_at")
        class_uncertain = (
            len(set(raw_labels)) > 1
            or (raw_conf is not None and float(raw_conf) < 0.55)
            or bool(confirmed and confirmed != raw_label)
        )
        return {
            "entity_id": e.entity_id,
            "label": e.label,
            "raw_label": raw_label,
            "raw_confidence": float(raw_conf) if raw_conf is not None else None,
            "confirmed_label": confirmed,
            "label_provenance": provenance,
            "label_confirmed_at": confirmed_at,
            "observed_labels": sorted(set(raw_labels)),
            "class_uncertain": class_uncertain,
            "created_at": e.created_at,
            "current_state": e.current_state.value,
            "canonical_bbox": canon,
            "last_seen_session_id": e.last_seen_session_id,
            "metadata": e.metadata,
        }

    def list_entities(self, workspace_id: str, state: Optional[str] = None) -> List[Dict[str, Any]]:
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise KeyError(f"Workspace '{slug}' not found.")

        target_state = EpistemicState(state) if state else None
        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            entities = store.list_entities(state=target_state)
            return [self._format_entity_summary(e) for e in entities]

    def get_entity_detail(self, workspace_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            return None

        with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
            ent = store.get_entity(entity_id)
            if not ent:
                return None

            entity_summary = self._format_entity_summary(ent)

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

            # Get observations, raw detections, and events in ref and query sessions
            ref_obs = {o.entity_id: o for o in store.list_observations_for_session(ref_id) if o.entity_id}
            query_obs = {o.entity_id: o for o in store.list_observations_for_session(query_id) if o.entity_id}
            ref_raws = store.list_raw_detections_for_session(ref_id)
            query_raws = store.list_raw_detections_for_session(query_id)
            query_events = {e.entity_id: e for e in store.list_change_events_for_session(query_id)}

            # Compute each entity's state at ref_id and query_id, plus intermediate movement/removal transitions
            session_order = {s.session_id: i for i, s in enumerate(sessions)}
            r_order = session_order.get(ref_id, 0)
            q_order = session_order.get(query_id, len(sessions) - 1)

            # Detect if ref and query span across a reset/restarted sequence (e.g. a second V01 upload)
            cross_sequence_warning: Optional[str] = None
            lo_ord, hi_ord = min(r_order, q_order), max(r_order, q_order)
            for idx_s in range(lo_ord + 1, hi_ord + 1):
                s_name = sessions[idx_s].session_name or ""
                if re.search(r"\b(V01|visit[_\s]?0?1)\b", s_name, re.IGNORECASE) and idx_s > 0:
                    cross_sequence_warning = (
                        f"Selected visits appear to span across two separate sequences "
                        f"('{sessions[idx_s].session_id}: {s_name}' restarts at V01). "
                        f"Select visits from the same scenario sequence for valid comparison."
                    )
                    break

            state_at_ref: Dict[str, str] = {}
            state_at_query: Dict[str, str] = {}
            moved_between: Dict[str, float] = {}
            events_in_window_entities: set[str] = set()
            cur_ev = store.conn.execute(
                "SELECT entity_id, to_session_id, new_state, displacement_distance FROM change_events ORDER BY timestamp ASC, rowid ASC;"
            )
            for row in cur_ev.fetchall():
                ev_ord = session_order.get(row["to_session_id"], -1)
                if ev_ord <= r_order:
                    state_at_ref[row["entity_id"]] = row["new_state"]
                if ev_ord <= q_order:
                    state_at_query[row["entity_id"]] = row["new_state"]
                if r_order < ev_ord <= q_order:
                    events_in_window_entities.add(row["entity_id"])
                    if row["new_state"] == "CONFIRMED_MOVED" and row["displacement_distance"] is not None:
                        moved_between[row["entity_id"]] = float(row["displacement_distance"])

            def _bbox_dict(b: Any) -> Optional[Dict[str, float]]:
                if b is None:
                    return None
                return {"x_min": b.x_min, "y_min": b.y_min, "x_max": b.x_max, "y_max": b.y_max}

            # Check alignment quality from query observations, raw detections, and temporal events
            qualities = [o.alignment_quality for o in query_obs.values() if o.alignment_quality is not None]
            evt_reg_failed = any(
                e.new_state == EpistemicState.INSUFFICIENT_EVIDENCE
                and "registration" in (e.explanation or "").lower()
                for e in query_events.values()
            )
            raw_reg_failed = bool(
                query_raws
                and all(r.registration_status != "SUCCESS" and r.canonical_bbox is None for r in query_raws)
            )
            if qualities:
                inlier_ratio = float(np.mean(qualities))
            elif evt_reg_failed or raw_reg_failed:
                inlier_ratio = 0.0
            else:
                inlier_ratio = 1.0
            reg_reliable = (query_id == sessions[0].session_id) or (
                inlier_ratio >= 0.20 and not evt_reg_failed and not raw_reg_failed
            )

            all_entity_ids = sorted(
                list(set(ref_obs.keys()).union(query_obs.keys()).union(events_in_window_entities))
            )
            items = []
            for eid in all_entity_ids:
                ent = store.get_entity(eid)
                label = ent.label if ent else "unknown"
                ent_meta = (ent.metadata or {}) if ent else {}
                created_sess = ent_meta.get("created_in_session")
                created_ord = session_order.get(created_sess, 0) if created_sess else 0

                r_o = ref_obs.get(eid)
                q_o = query_obs.get(eid)
                ev = query_events.get(eid)
                ref_st = state_at_ref.get(eid) or ("NONE" if created_ord > r_order else "CONFIRMED_PRESENT")
                query_st = state_at_query.get(eid) or (ent.current_state.value if ent else "CONFIRMED_PRESENT")

                r_box = _bbox_dict(r_o.canonical_bbox if r_o else None)
                q_box = _bbox_dict(q_o.canonical_bbox if q_o else None)
                r_raw_box = _bbox_dict(r_o.raw_bbox if r_o else None)
                q_raw_box = _bbox_dict(q_o.raw_bbox if q_o else None)

                # Compute actual canonical centroid displacement between ref and query when both exist
                pair_disp: Optional[float] = None
                if r_o and r_o.canonical_bbox and q_o and q_o.canonical_bbox:
                    rcx = (r_o.canonical_bbox.x_min + r_o.canonical_bbox.x_max) / 2.0
                    rcy = (r_o.canonical_bbox.y_min + r_o.canonical_bbox.y_max) / 2.0
                    qcx = (q_o.canonical_bbox.x_min + q_o.canonical_bbox.x_max) / 2.0
                    qcy = (q_o.canonical_bbox.y_min + q_o.canonical_bbox.y_max) / 2.0
                    pair_disp = float(np.hypot(qcx - rcx, qcy - rcy))

                if created_ord > r_order:
                    if query_st == "CONFIRMED_REMOVED":
                        status_change = query_st
                        movement_status = "REMOVED"
                        prior_st = "NONE"
                        new_st = query_st
                        expl = f"Entity introduced after {ref_id} and transitioned to {query_st} by {query_id}."
                    elif query_st == "PENDING_REMOVAL":
                        status_change = query_st
                        movement_status = "NOT_DETECTED"
                        prior_st = "NONE"
                        new_st = query_st
                        expl = f"Entity introduced after {ref_id} and not detected in {query_id}."
                    else:
                        status_change = "ADDED"
                        movement_status = "ADDED"
                        prior_st = "NONE"
                        new_st = query_st
                        expl = "Entity first observed after reference session."
                    disp = pair_disp if pair_disp is not None else (ev.displacement_distance if ev else moved_between.get(eid))
                elif not reg_reliable and eid in ref_obs and eid not in query_obs:
                    status_change = "INSUFFICIENT_EVIDENCE"
                    movement_status = "REGISTRATION_UNRELIABLE"
                    prior_st = ref_st
                    new_st = "INSUFFICIENT_EVIDENCE"
                    expl = f"Planar registration unreliable in {query_id}; positional comparison withheld."
                    disp = None
                elif query_st == "CONFIRMED_REMOVED" and ref_st not in ("CONFIRMED_REMOVED", "PENDING_REMOVAL"):
                    status_change = query_st
                    movement_status = "REMOVED"
                    prior_st = ref_st
                    new_st = query_st
                    expl = ev.explanation if ev else f"Entity transitioned to {query_st} by {query_id}."
                    disp = None
                elif query_st == "PENDING_REMOVAL" and ref_st not in ("CONFIRMED_REMOVED", "PENDING_REMOVAL"):
                    status_change = query_st
                    movement_status = "NOT_DETECTED"
                    prior_st = ref_st
                    new_st = query_st
                    expl = ev.explanation if ev else f"Entity temporarily not detected in {query_id} (pending confirmation)."
                    disp = None
                elif ev and ev.new_state == EpistemicState.CONFIRMED_MOVED:
                    status_change = "CONFIRMED_MOVED"
                    movement_status = "MOVED"
                    prior_st = ref_st
                    new_st = "CONFIRMED_MOVED"
                    expl = ev.explanation
                    disp = pair_disp if pair_disp is not None else ev.displacement_distance
                elif query_st == "CONFIRMED_MOVED" and ref_st != "CONFIRMED_MOVED":
                    status_change = "CONFIRMED_MOVED"
                    movement_status = "MOVED"
                    prior_st = ref_st
                    new_st = "CONFIRMED_MOVED"
                    disp = pair_disp if pair_disp is not None else moved_between.get(eid)
                    disp_str = f" ({disp:.1f} px)" if disp is not None else ""
                    expl = f"Entity relocated between {ref_id} and {query_id}{disp_str}."
                elif query_st == "PENDING_MOVEMENT" or (ev and ev.new_state.value == "PENDING_MOVEMENT"):
                    status_change = "PENDING_MOVEMENT"
                    movement_status = "PENDING_MOVEMENT"
                    prior_st = ref_st
                    new_st = "PENDING_MOVEMENT"
                    expl = ev.explanation if ev else f"Candidate movement in {query_id} awaiting multi-visit confirmation."
                    disp = pair_disp if pair_disp is not None else (ev.displacement_distance if ev else None)
                elif query_st == "AMBIGUOUS_IDENTITY" or (ev and ev.new_state.value == "AMBIGUOUS_IDENTITY"):
                    status_change = "AMBIGUOUS_IDENTITY"
                    movement_status = "IDENTITY_AMBIGUOUS"
                    prior_st = ref_st
                    new_st = "AMBIGUOUS_IDENTITY"
                    expl = ev.explanation if ev else f"Multiple competing association candidates in {query_id}."
                    disp = None
                elif query_st == "INSUFFICIENT_EVIDENCE" or (ev and ev.new_state.value == "INSUFFICIENT_EVIDENCE"):
                    status_change = "INSUFFICIENT_EVIDENCE"
                    expl = ev.explanation if ev else f"Insufficient positional or visual evidence in {query_id}."
                    movement_status = "REGISTRATION_UNRELIABLE" if ("registration" in expl.lower() or not reg_reliable) else "INSUFFICIENT_EVIDENCE"
                    prior_st = ref_st
                    new_st = "INSUFFICIENT_EVIDENCE"
                    disp = None
                elif eid in ref_obs and eid not in query_obs:
                    status_change = "MISSING_IN_QUERY"
                    movement_status = "NOT_DETECTED" if reg_reliable else "REGISTRATION_UNRELIABLE"
                    prior_st = ref_st
                    new_st = query_st
                    expl = "Entity not detected in query session." if reg_reliable else "Registration unreliable; entity could not be verified."
                    disp = None
                elif ev:
                    status_change = ev.new_state.value
                    prior_st = ev.prior_state.value
                    new_st = ev.new_state.value
                    expl = ev.explanation
                    disp = pair_disp if pair_disp is not None else ev.displacement_distance
                    if not reg_reliable:
                        movement_status = "REGISTRATION_UNRELIABLE"
                        disp = None
                    elif pair_disp is not None:
                        movement_status = "NOT_MOVED"
                    else:
                        movement_status = "INSUFFICIENT_EVIDENCE"
                else:
                    status_change = "UNCHANGED"
                    prior_st = ref_st
                    new_st = query_st
                    disp = pair_disp
                    if not reg_reliable:
                        movement_status = "REGISTRATION_UNRELIABLE"
                        expl = f"Planar registration unreliable in {query_id}; cannot confirm stationary vs. moved."
                        disp = None
                    elif pair_disp is not None:
                        movement_status = "NOT_MOVED"
                        expl = f"Verified stationary in canonical frame (displacement {pair_disp:.1f} px)."
                    else:
                        movement_status = "INSUFFICIENT_EVIDENCE"
                        expl = "Canonical bounding boxes unavailable for both visits."

                # Surface raw model labels, confirmed label provenance, and evidence source honestly
                raw_labels = list(ent_meta.get("raw_predicted_labels") or [])
                if not raw_labels:
                    raw_labels = [ent_meta.get("original_raw_label", label)]
                raw_label = ent_meta.get("original_raw_label") or (raw_labels[-1] if raw_labels else label)
                confirmed_lbl = ent_meta.get("confirmed_label")
                lbl_prov = ent_meta.get("label_provenance") or ("USER_CONFIRMED" if confirmed_lbl else "MODEL_PREDICTED")
                ev_src = "MODEL_DETECTION"
                if q_o and isinstance(q_o.raw_evidence, dict) and q_o.raw_evidence.get("evidence_source"):
                    ev_src = str(q_o.raw_evidence["evidence_source"])
                elif ev and ev.source in ("HUMAN_CONFIRMED", "SUB_THRESHOLD_RECOVERY"):
                    ev_src = ev.source

                r_conf = r_o.detector_confidence if r_o else None
                q_conf = q_o.detector_confidence if q_o else None
                min_conf = min([c for c in (r_conf, q_conf) if c is not None], default=1.0)
                class_uncertain = (len(set(raw_labels)) > 1) or (min_conf < 0.55) or bool(confirmed_lbl and confirmed_lbl != raw_label)

                items.append({
                    "entity_id": eid,
                    "label": label,
                    "raw_label": raw_label,
                    "confirmed_label": confirmed_lbl,
                    "label_provenance": lbl_prov,
                    "evidence_source": ev_src,
                    "observed_labels": sorted(set(raw_labels)),
                    "class_uncertain": class_uncertain,
                    "status_change": status_change,
                    "movement_status": movement_status,
                    "prior_state": prior_st,
                    "new_state": new_st,
                    "displacement_px": disp,
                    "ref_confidence": r_conf,
                    "query_confidence": q_conf,
                    "ref_bbox": r_box,
                    "query_bbox": q_box,
                    "ref_raw_bbox": r_raw_box,
                    "query_raw_bbox": q_raw_box,
                    "explanation": expl,
                })

            return {
                "workspace_id": slug,
                "reference_session_id": ref_id,
                "query_session_id": query_id,
                "registration_status": "SUCCESS" if reg_reliable else "DEGRADED",
                "registration_inlier_ratio": inlier_ratio,
                "ref_detections_count": len(ref_raws),
                "query_detections_count": len(query_raws),
                "cross_sequence_warning": cross_sequence_warning,
                "items": items,
            }

    def _sync_entity_raw_label_metadata(self, store: SQLiteMemoryStore, session_id: str) -> None:
        """Persists original raw model label/confidence and observed label history without overwriting confirmed labels."""
        for obs in store.list_observations_for_session(session_id):
            if not obs.entity_id:
                continue
            ent = store.get_entity(obs.entity_id)
            if not ent:
                continue
            meta = dict(ent.metadata or {})
            ev_dict = obs.raw_evidence if isinstance(obs.raw_evidence, dict) else {}
            det_lbl = ev_dict.get("detected_label") or ev_dict.get("raw_label") or ent.label
            changed = False
            if "original_raw_label" not in meta:
                meta["original_raw_label"] = det_lbl
                changed = True
            if "original_raw_confidence" not in meta and obs.detector_confidence is not None:
                meta["original_raw_confidence"] = float(obs.detector_confidence)
                changed = True
            raw_list = list(meta.get("raw_predicted_labels") or [])
            if not raw_list:
                raw_list = [meta["original_raw_label"]]
                changed = True
            if det_lbl and det_lbl not in raw_list:
                raw_list.append(det_lbl)
                changed = True
            meta["raw_predicted_labels"] = raw_list
            if "label_provenance" not in meta:
                meta["label_provenance"] = "USER_CONFIRMED" if meta.get("confirmed_label") else "MODEL_PREDICTED"
                changed = True
            if changed:
                store.conn.execute(
                    "UPDATE entities SET metadata_json = ? WHERE entity_id = ?;",
                    (json.dumps(meta), ent.entity_id),
                )
        store.conn.commit()

    def _attempt_subthreshold_recovery(
        self,
        runner: RealImagePipelineRunner,
        img: np.ndarray,
        session_id: str,
        ts_utc: str,
        reg: Any,
        accepted_detections: List[Any],
        confidence_threshold: float,
    ) -> None:
        """
        Separate, strictly evidence-gated candidate recovery for unmatched entities.
        Never alters raw_detections or frozen production confidence_threshold (0.25).
        Requires reliable registration, 0.12 <= conf < threshold, no overlap with accepted detections,
        appearance cosine similarity >= 0.85, and unambiguous one-to-one separation.
        """
        if not reg or not reg.success or reg.homography is None:
            return
        if reg.inlier_ratio is not None and reg.inlier_ratio < runner.temporal_config.min_registration_inlier_ratio:
            return

        store = runner.store
        obs_in_sess = {o.entity_id for o in store.list_observations_for_session(session_id) if o.entity_id}
        unmatched_ents = [
            e for e in store.list_entities()
            if e.entity_id not in obs_in_sess and e.current_state != EpistemicState.CONFIRMED_REMOVED
        ]
        if not unmatched_ents:
            return

        try:
            low_batch = runner.detector.detect(img, confidence_threshold=0.10)
        except Exception:
            return

        from objectmemory.baseline.comparator import calculate_bbox_iou
        from objectmemory.detection.models import BoundingBox as DetBBox
        from objectmemory.domain.models import Observation as DomObs, ChangeEvent as DomEvt
        from objectmemory.domain.states import VisibilityStatus
        from objectmemory.detection.pipeline import transform_bbox_to_canonical

        sub_cands = []
        for c in low_batch.detections:
            if c.confidence < 0.12 or c.confidence >= confidence_threshold:
                continue
            overlaps_accepted = any(
                calculate_bbox_iou(c.raw_bbox, a.raw_bbox) >= 0.20
                for a in accepted_detections
            )
            if not overlaps_accepted:
                sub_cands.append(c)
        if not sub_cands:
            return

        # Load entity latest signature vectors
        ent_sigs: Dict[str, np.ndarray] = {}
        for ent in unmatched_ents:
            for o in reversed(store.list_observations_for_entity(ent.entity_id)):
                vec = (o.raw_evidence or {}).get("signature_vector") if isinstance(o.raw_evidence, dict) else None
                if vec:
                    ent_sigs[ent.entity_id] = np.asarray(vec, dtype=np.float32)
                    break
        if not ent_sigs:
            return

        cand_sigs = [runner.signature_extractor.extract(img, c.raw_bbox) for c in sub_cands]
        sim_mat = np.zeros((len(unmatched_ents), len(sub_cands)), dtype=float)
        for i, ent in enumerate(unmatched_ents):
            evec = ent_sigs.get(ent.entity_id)
            if evec is None:
                continue
            enorm = float(np.linalg.norm(evec))
            if enorm <= 1e-8:
                continue
            for j, csig in enumerate(cand_sigs):
                if csig is None or not csig.is_valid or csig.combined_vector is None:
                    continue
                cvec = csig.combined_vector
                cnorm = float(np.linalg.norm(cvec))
                if cnorm > 1e-8:
                    sim_mat[i, j] = float(np.dot(evec, cvec) / (enorm * cnorm))

        used_cands: set[int] = set()
        for i, ent in enumerate(unmatched_ents):
            row_sims = sim_mat[i]
            best_j = int(np.argmax(row_sims))
            best_sim = min(1.0, max(0.0, float(row_sims[best_j])))
            if best_sim < 0.85 or best_j in used_cands:
                continue
            # Ambiguity check: no competing candidate within 0.05 or competing entity >= 0.80
            second_cand_sim = max([float(row_sims[k]) for k in range(len(sub_cands)) if k != best_j], default=0.0)
            other_ent_sim = max([float(sim_mat[r, best_j]) for r in range(len(unmatched_ents)) if r != i], default=0.0)
            if best_sim - second_cand_sim < 0.05 or other_ent_sim >= 0.80:
                continue

            used_cands.add(best_j)
            cand = sub_cands[best_j]
            csig = cand_sigs[best_j]
            canon_box = transform_bbox_to_canonical(cand.raw_bbox, reg.homography, ref_size=reg.ref_size)
            if canon_box is None:
                continue

            meta = dict(ent.metadata or {})
            raw_list = list(meta.get("raw_predicted_labels") or [meta.get("original_raw_label", ent.label)])
            if cand.label not in raw_list:
                raw_list.append(cand.label)
            meta["raw_predicted_labels"] = raw_list
            meta["vacant_streak"] = 0
            meta["unobserved_streak"] = 0
            meta["last_reliable_canonical_bbox"] = [canon_box.x_min, canon_box.y_min, canon_box.x_max, canon_box.y_max]

            init_list = meta.get("initial_canonical_bbox")
            init_box = DetBBox(*init_list) if init_list else ent.canonical_bbox
            disp = 0.0
            if init_box is not None:
                icx, icy = (init_box.x_min + init_box.x_max) / 2.0, (init_box.y_min + init_box.y_max) / 2.0
                ccx, ccy = (canon_box.x_min + canon_box.x_max) / 2.0, (canon_box.y_min + canon_box.y_max) / 2.0
                disp = float(np.hypot(ccx - icx, ccy - icy))

            target_state = (
                EpistemicState.CONFIRMED_MOVED
                if disp > runner.engine.config.movement_displacement_threshold
                else EpistemicState.CONFIRMED_PRESENT
            )
            obs_id = f"obs_{session_id}_{ent.entity_id}_rec"
            sig_list = csig.combined_vector.tolist() if csig and csig.combined_vector is not None else None
            store.add_observation(
                DomObs(
                    observation_id=obs_id,
                    session_id=session_id,
                    timestamp=ts_utc,
                    image_reference=f"memory://{session_id}.png",
                    entity_id=ent.entity_id,
                    raw_bbox=cand.raw_bbox,
                    canonical_bbox=canon_box,
                    visibility_status=VisibilityStatus.CLEARLY_VISIBLE,
                    detector_confidence=float(cand.confidence),
                    identity_match_score=best_sim,
                    alignment_quality=float(reg.inlier_ratio or 1.0),
                    raw_evidence={
                        "evidence_source": "SUB_THRESHOLD_RECOVERY",
                        "raw_label": cand.label,
                        "raw_confidence": float(cand.confidence),
                        "appearance_similarity": round(best_sim, 4),
                        "normal_detector_threshold": confidence_threshold,
                        "signature_vector": sig_list,
                    },
                )
            )
            store.conn.execute("UPDATE entities SET metadata_json = ? WHERE entity_id = ?;", (json.dumps(meta), ent.entity_id))
            store.conn.commit()

            expl = (
                f"Sub-threshold candidate recovered via appearance similarity ({best_sim:.3f}, "
                f"raw '{cand.label}' conf={cand.confidence:.3f} < {confidence_threshold:.2f}); "
                f"canonical displacement {disp:.1f}px."
            )
            if target_state != ent.current_state or target_state == EpistemicState.CONFIRMED_MOVED:
                evt = DomEvt(
                    event_id=f"evt_{session_id}_{ent.entity_id}_rec_{target_state.value.lower()}",
                    entity_id=ent.entity_id,
                    from_session_id=ent.last_seen_session_id,
                    to_session_id=session_id,
                    timestamp=ts_utc,
                    prior_state=ent.current_state,
                    new_state=target_state,
                    source="SUB_THRESHOLD_RECOVERY",
                    displacement_distance=disp,
                    supporting_evidence={
                        "evidence_source": "SUB_THRESHOLD_RECOVERY",
                        "raw_label": cand.label,
                        "raw_confidence": float(cand.confidence),
                        "appearance_similarity": round(best_sim, 4),
                        "displacement": disp,
                    },
                    explanation=expl,
                )
                store.record_state_transition(
                    entity_id=ent.entity_id,
                    new_state=target_state,
                    event=evt,
                    canonical_bbox=canon_box,
                    last_seen_session_id=session_id,
                )

    def update_entity_label(
        self,
        workspace_id: str,
        entity_id: str,
        confirmed_label: str,
    ) -> Dict[str, Any]:
        """Allows user confirmation of an entity's semantic label while preserving raw detector predictions."""
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise KeyError(f"Workspace '{slug}' not found.")
        cleaned = confirmed_label.strip()
        if not cleaned:
            raise ValueError("Confirmed label cannot be empty.")

        with self._get_workspace_lock(slug):
            with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
                ent = store.get_entity(entity_id)
                if not ent:
                    raise KeyError(f"Entity '{entity_id}' not found in workspace '{slug}'.")
                meta = dict(ent.metadata or {})
                if "original_raw_label" not in meta:
                    meta["original_raw_label"] = ent.label
                raw_list = list(meta.get("raw_predicted_labels") or [meta["original_raw_label"]])
                meta["raw_predicted_labels"] = raw_list
                meta["confirmed_label"] = cleaned
                meta["label_provenance"] = "USER_CONFIRMED"
                meta["label_confirmed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                store.conn.execute(
                    "UPDATE entities SET label = ?, metadata_json = ? WHERE entity_id = ?;",
                    (cleaned, json.dumps(meta), entity_id),
                )
                store.conn.commit()
                updated = store.get_entity(entity_id)
                return self._format_entity_summary(updated)

    def clear_entity_label(
        self,
        workspace_id: str,
        entity_id: str,
    ) -> Dict[str, Any]:
        """Clears user-confirmed label and restores model-derived labelling."""
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise KeyError(f"Workspace '{slug}' not found.")

        with self._get_workspace_lock(slug):
            with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
                ent = store.get_entity(entity_id)
                if not ent:
                    raise KeyError(f"Entity '{entity_id}' not found in workspace '{slug}'.")
                meta = dict(ent.metadata or {})
                restored_label = meta.get("original_raw_label") or (meta.get("raw_predicted_labels") or [ent.label])[0]
                meta["confirmed_label"] = None
                meta["label_provenance"] = "MODEL_PREDICTED"
                meta["label_confirmed_at"] = None
                store.conn.execute(
                    "UPDATE entities SET label = ?, metadata_json = ? WHERE entity_id = ?;",
                    (restored_label, json.dumps(meta), entity_id),
                )
                store.conn.commit()
                updated = store.get_entity(entity_id)
                return self._format_entity_summary(updated)

    def confirm_entity_location(
        self,
        workspace_id: str,
        session_id: str,
        entity_id: str,
        bbox_dict: Dict[str, float],
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Human-assisted location confirmation for an existing entity in a visit.
        Never modifies raw_detections; gates movement decisions on registration reliability.
        """
        slug = _sanitize_slug(workspace_id)
        db_path = self._db_path(slug)
        if not db_path.exists():
            raise KeyError(f"Workspace '{slug}' not found.")

        from objectmemory.detection.models import BoundingBox as DetBBox
        from objectmemory.domain.models import Observation as DomObs, ChangeEvent as DomEvt
        from objectmemory.domain.states import VisibilityStatus
        from objectmemory.detection.pipeline import transform_bbox_to_canonical
        from objectmemory.registration.registrar import PlanarRegistrar
        from objectmemory.temporal.models import TemporalEngineConfig

        with self._get_workspace_lock(slug):
            with SQLiteMemoryStore(db_path, check_same_thread=False) as store:
                ent = store.get_entity(entity_id)
                if not ent:
                    raise KeyError(f"Entity '{entity_id}' not found in workspace '{slug}'.")
                sessions = store.list_sessions(order_asc=True)
                sess = store.get_session(session_id)
                if not sess:
                    raise KeyError(f"Session '{session_id}' not found in workspace '{slug}'.")

                q_img_path = self._images_dir(slug) / f"{session_id}.png"
                r_sess_id = sessions[0].session_id
                r_img_path = self._images_dir(slug) / f"{r_sess_id}.png"
                if not q_img_path.exists() or not r_img_path.exists():
                    raise ValueError("Stored session images required for human location confirmation are missing.")

                q_img = cv2.imread(str(q_img_path))
                r_img = cv2.imread(str(r_img_path))
                if q_img is None or r_img is None:
                    raise ValueError("Failed to decode stored session images.")

                h_img, w_img = q_img.shape[:2]
                x1, y1, x2, y2 = float(bbox_dict["x_min"]), float(bbox_dict["y_min"]), float(bbox_dict["x_max"]), float(bbox_dict["y_max"])
                if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img or x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Bounding box [{x1}, {y1}, {x2}, {y2}] is invalid for image size {w_img}x{h_img}.")
                raw_bbox = DetBBox(x1, y1, x2, y2)

                registrar = PlanarRegistrar()
                reg = registrar.register(r_img, q_img) if session_id != r_sess_id else registrar.register(r_img, r_img)
                tcfg = TemporalEngineConfig()
                is_reg_reliable = bool(
                    reg.success
                    and reg.homography is not None
                    and (reg.inlier_ratio is None or reg.inlier_ratio >= tcfg.min_registration_inlier_ratio)
                )

                now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                prior_state = ent.current_state
                canon_box: Optional[DetBBox] = None
                disp: Optional[float] = None

                if is_reg_reliable:
                    canon_box = transform_bbox_to_canonical(raw_bbox, reg.homography, ref_size=reg.ref_size)
                    meta = dict(ent.metadata or {})
                    init_list = meta.get("initial_canonical_bbox")
                    init_box = DetBBox(*init_list) if init_list else ent.canonical_bbox
                    if canon_box is not None and init_box is not None:
                        icx, icy = (init_box.x_min + init_box.x_max) / 2.0, (init_box.y_min + init_box.y_max) / 2.0
                        ccx, ccy = (canon_box.x_min + canon_box.x_max) / 2.0, (canon_box.y_min + canon_box.y_max) / 2.0
                        disp = float(np.hypot(ccx - icx, ccy - icy))
                        new_state = EpistemicState.CONFIRMED_MOVED if disp > tcfg.movement_displacement_threshold else EpistemicState.CONFIRMED_PRESENT
                    else:
                        new_state = EpistemicState.CONFIRMED_PRESENT
                    meta["vacant_streak"] = 0
                    meta["unobserved_streak"] = 0
                    if canon_box is not None:
                        meta["last_reliable_canonical_bbox"] = [canon_box.x_min, canon_box.y_min, canon_box.x_max, canon_box.y_max]
                    expl = f"Human-confirmed location in {session_id}" + (f" (displacement: {disp:.1f}px)." if disp is not None else ".")
                    if notes:
                        expl += f" Note: {notes}"
                else:
                    new_state = EpistemicState.INSUFFICIENT_EVIDENCE
                    expl = f"Human-confirmed location in {session_id} withheld from movement claim due to unreliable registration ({reg.status.value})."

                obs_id = f"obs_{session_id}_{entity_id}_human_{int(time.time() * 1000)}"
                store.add_observation(
                    DomObs(
                        observation_id=obs_id,
                        session_id=session_id,
                        timestamp=sess.timestamp,
                        image_reference=f"memory://{session_id}.png",
                        entity_id=entity_id,
                        raw_bbox=raw_bbox,
                        canonical_bbox=canon_box,
                        visibility_status=VisibilityStatus.CLEARLY_VISIBLE if is_reg_reliable else VisibilityStatus.UNCERTAIN,
                        detector_confidence=None,
                        identity_match_score=1.0,
                        alignment_quality=float(reg.inlier_ratio or 0.0),
                        raw_evidence={
                            "evidence_source": "HUMAN_CONFIRMED",
                            "coordinate_system": "raw_pixel",
                            "image_size": [w_img, h_img],
                            "confirmed_at": now_utc,
                            "notes": notes,
                            "registration_reliable": is_reg_reliable,
                        },
                    )
                )
                if is_reg_reliable:
                    store.conn.execute("UPDATE entities SET metadata_json = ? WHERE entity_id = ?;", (json.dumps(meta), entity_id))
                    store.conn.commit()

                evt = DomEvt(
                    event_id=f"evt_{session_id}_{entity_id}_human_{int(time.time() * 1000)}",
                    entity_id=entity_id,
                    from_session_id=ent.last_seen_session_id,
                    to_session_id=session_id,
                    timestamp=sess.timestamp,
                    prior_state=prior_state,
                    new_state=new_state,
                    source="HUMAN_CONFIRMED",
                    displacement_distance=disp,
                    supporting_evidence={
                        "evidence_source": "HUMAN_CONFIRMED",
                        "confirmed_at": now_utc,
                        "registration_reliable": is_reg_reliable,
                        "displacement": disp,
                    },
                    explanation=expl,
                )
                store.record_state_transition(
                    entity_id=entity_id,
                    new_state=new_state,
                    event=evt,
                    canonical_bbox=canon_box,
                    last_seen_session_id=session_id if is_reg_reliable else ent.last_seen_session_id,
                )

                return {
                    "workspace_id": slug,
                    "session_id": session_id,
                    "entity_id": entity_id,
                    "evidence_source": "HUMAN_CONFIRMED",
                    "registration_reliable": is_reg_reliable,
                    "prior_state": prior_state.value,
                    "new_state": new_state.value,
                    "displacement_px": disp,
                    "canonical_bbox": {"x_min": canon_box.x_min, "y_min": canon_box.y_min, "x_max": canon_box.x_max, "y_max": canon_box.y_max} if canon_box else None,
                    "raw_bbox": bbox_dict,
                    "explanation": expl,
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
