"""
Multi-Visit Temporal Memory Engine for ObjectMemory.

Implements multi-visit belief updating over persistent workbench entities.

Key Design & Principles:
1. Chronological processing: Visits are ordered strictly by timestamp.
2. Idempotency: Duplicate reprocessing of an already processed visit returns the
   existing audit report without duplicating observations, entities, or change events.
3. Out-of-order rejection: Visits with timestamps earlier than the latest processed visit
   are rejected with clear ValueError to preserve causal monotonicity.
4. Preserves domain transition policy: Uses store.record_state_transition() atomically.
5. Intended removal progression:
   CONFIRMED_PRESENT -> PENDING_REMOVAL -> CONFIRMED_REMOVED
   - Direct jump from CONFIRMED_PRESENT to CONFIRMED_REMOVED is prevented by policy.
   - Non-detection alone does not confirm removal.
   - Requires verified vacant patch across configurable consecutive visits (default 2).
   - Occlusion or out-of-frame canonical bbox prevents removal progression.
6. Redetection restoration:
   - If entity in PENDING_REMOVAL or CONFIRMED_REMOVED is re-observed, transitions to
     CONFIRMED_PRESENT (or CONFIRMED_MOVED if relocated).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from objectmemory.baseline.comparator import (
    calculate_bbox_iou,
    calculate_centroid_distance,
    check_bbox_overlap_fraction,
)
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.detection.pipeline import transform_bbox_to_canonical
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import (
    BoundingBox,
    ChangeEvent,
    Observation,
    RawDetection,
    Session,
    WorkbenchEntity,
    normalize_iso_timestamp,
)
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.temporal.associator import CrossVisitEntityAssociator
from objectmemory.temporal.models import (
    AssociationMatch,
    EntityBeliefUpdate,
    TemporalEngineConfig,
    VisitInput,
    VisitProcessingReport,
)


class TemporalMemoryEngine:
    """
    Maintains and updates beliefs about persistent workbench entities across repeated visits.
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        config: Optional[TemporalEngineConfig] = None,
        associator: Optional[CrossVisitEntityAssociator] = None,
        signature_extractor: Optional[VisualSignatureExtractor] = None,
    ) -> None:
        self.store = store
        self.config = config or TemporalEngineConfig()
        self.associator = associator or CrossVisitEntityAssociator(config=self.config)
        self.signature_extractor = signature_extractor or VisualSignatureExtractor()

    def _is_registration_reliable(self, registration: RegistrationResult) -> bool:
        """
        Determines whether registration succeeded and meets the minimum inlier
        ratio and reprojection error required to trust canonical spatial coordinates.
        """
        if not registration.success or registration.homography is None:
            return False
        if (
            registration.inlier_ratio is not None
            and registration.inlier_ratio < self.config.min_registration_inlier_ratio
        ):
            return False
        if registration.reprojection_stats is not None:
            if (
                registration.reprojection_stats.mean is not None
                and registration.reprojection_stats.mean > self.config.max_reprojection_error
            ):
                return False
        return True

    def compute_effective_movement_threshold(
        self,
        registration: Optional[RegistrationResult],
        base_threshold: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Computes uncertainty-aware effective movement threshold:
            T_effective = T_base + Delta_U

        Where Delta_U = f(reproj_err, inlier_ratio, spatial_coverage, scale).
        """
        t_base = base_threshold if base_threshold is not None else self.config.movement_displacement_threshold
        if registration is None or not registration.success:
            return t_base, 0.0, {
                "base_threshold_px": round(t_base, 2),
                "delta_u_px": 0.0,
                "effective_threshold_px": round(t_base, 2),
            }

        # 1. Reprojection error term: higher reprojection error increases coordinate uncertainty
        u_reproj = 0.0
        if registration.reprojection_stats and registration.reprojection_stats.mean is not None:
            mean_err = float(registration.reprojection_stats.mean)
            u_reproj = min(15.0, max(0.0, (mean_err - 1.0) * 3.5))

        # 2. Inlier ratio penalty: lower ratio indicates weaker geometric consensus
        u_inlier = 0.0
        if registration.inlier_ratio is not None:
            r_inlier = float(registration.inlier_ratio)
            if r_inlier < 0.60:
                u_inlier = min(15.0, max(0.0, (0.60 - r_inlier) * 25.0))

        # 3. Spatial feature coverage penalty: clustered features cause high geometric variance away from cluster
        u_cov = 0.0
        spatial_cov = float(registration.spatial_coverage) if registration.spatial_coverage is not None else 0.50
        if spatial_cov < 0.50:
            u_cov = min(15.0, max(0.0, (0.50 - spatial_cov) * 25.0))

        # 4. Scale distortion penalty: resolution changes scale coordinate errors
        u_scale = 0.0
        try:
            qw, qh = registration.query_size
            rw, rh = registration.ref_size
            if qw > 0 and qh > 0 and rw > 0 and rh > 0:
                scale_ratio = math.sqrt((qw * qh) / float(rw * rh))
                u_scale = min(10.0, abs(math.log(max(scale_ratio, 0.01))) * 6.0)
        except Exception:
            u_scale = 0.0

        delta_u = min(45.0, u_reproj + u_inlier + u_cov + u_scale)
        t_effective = t_base + delta_u

        breakdown = {
            "base_threshold_px": round(t_base, 2),
            "delta_u_px": round(delta_u, 2),
            "effective_threshold_px": round(t_effective, 2),
            "u_reproj_px": round(u_reproj, 2),
            "u_inlier_px": round(u_inlier, 2),
            "u_coverage_px": round(u_cov, 2),
            "u_scale_px": round(u_scale, 2),
        }
        return t_effective, delta_u, breakdown

    def _is_bbox_clipped(
        self,
        bbox: BoundingBox,
        image_size: Tuple[int, int],
    ) -> bool:
        """Determines whether a bounding box touches or crosses the image border within border_margin_px."""
        margin = self.config.border_margin_px
        qw, qh = image_size
        return (
            bbox.x_min <= margin
            or bbox.y_min <= margin
            or bbox.x_max >= float(qw) - margin
            or bbox.y_max >= float(qh) - margin
        )

    def _is_spatial_comparison_reliable(
        self,
        entity: WorkbenchEntity,
        det: Detection,
        visit: VisitInput,
        initial_bbox: Optional[BoundingBox],
        last_reliable_bbox: Optional[BoundingBox] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluates whether global registration and field-of-view overlap are sufficient
        to project and compare spatial coordinates.
        """
        if not self._is_registration_reliable(visit.registration):
            return False, "Registration unverified or failed"

        if det.canonical_bbox is None:
            return False, "Detection lacks canonical bounding box projection"

        ref_box = last_reliable_bbox or initial_bbox
        if ref_box is None:
            return False, "Entity lacks baseline canonical bounding box"

        # Overlap fraction check with registered field of view
        if visit.registration.overlap_mask is not None:
            if initial_bbox is not None:
                init_overlap = check_bbox_overlap_fraction(
                    initial_bbox, visit.registration.overlap_mask
                )
                if init_overlap < self.config.min_overlap_fraction:
                    return False, (
                        f"Initial canonical position is outside registered overlap "
                        f"({init_overlap:.1%} < {self.config.min_overlap_fraction:.1%})"
                    )

            ref_overlap = check_bbox_overlap_fraction(
                ref_box, visit.registration.overlap_mask
            )
            if ref_overlap < self.config.min_overlap_fraction:
                return False, (
                    f"Reference canonical position is outside registered overlap "
                    f"({ref_overlap:.1%} < {self.config.min_overlap_fraction:.1%})"
                )

            cur_overlap = check_bbox_overlap_fraction(
                det.canonical_bbox, visit.registration.overlap_mask
            )
            if cur_overlap < self.config.min_overlap_fraction:
                return False, (
                    f"Observed canonical position is outside registered overlap "
                    f"({cur_overlap:.1%} < {self.config.min_overlap_fraction:.1%})"
                )

        return True, "Spatial comparison reliable"

    def process_visit(self, visit: VisitInput) -> VisitProcessingReport:
        """
        Processes a single visit into the temporal memory store.
        Enforces chronological order, idempotency, atomic transitions, and belief updates.
        """
        visit_ts = normalize_iso_timestamp(visit.timestamp)

        # 1. Check idempotency: If session already exists, check whether already processed
        existing_session = self.store.get_session(visit.session_id)
        if existing_session is not None:
            # Check existing observations or raw detections for this session
            existing_obs = self.store.list_observations_for_session(visit.session_id)
            existing_raw = self.store.list_raw_detections_for_session(visit.session_id)
            if existing_obs or existing_raw:
                # Idempotent replay: return cached summary without mutating state
                return self._build_reprocessed_report(visit, existing_obs)

        # 2. Check chronological monotonic order against existing sessions
        cursor = self.store.conn.execute("SELECT timestamp FROM sessions ORDER BY timestamp DESC LIMIT 1;")
        row = cursor.fetchone()
        if row is not None:
            latest_ts = row[0]
            if visit_ts < latest_ts:
                raise ValueError(
                    f"Out-of-order visit rejected: Visit timestamp '{visit_ts}' is earlier than "
                    f"latest persisted session timestamp '{latest_ts}'."
                )

        # 3. Persist session if not already stored
        is_reg_reliable = self._is_registration_reliable(visit.registration)
        reg_status_str = "REGISTERED" if is_reg_reliable else "UNREGISTERED"

        if existing_session is None:
            sess_meta = dict(visit.metadata or {})
            if visit.registration is not None:
                sess_meta["registration"] = {
                    "success": visit.registration.success,
                    "status": visit.registration.status.value,
                    "method_used": visit.registration.method_used,
                    "inlier_ratio": float(visit.registration.inlier_ratio or 0.0),
                    "mean_reprojection_error": (
                        visit.registration.reprojection_stats.mean
                        if getattr(visit.registration, "reprojection_stats", None) is not None
                        else getattr(visit.registration, "mean_reprojection_error", None)
                    ),
                    "inliers_count": getattr(
                        visit.registration, "num_inliers", getattr(visit.registration, "inliers_count", 0)
                    ),
                    "ref_size": list(visit.registration.ref_size) if visit.registration.ref_size else [0, 0],
                    "query_size": list(visit.registration.query_size) if visit.registration.query_size else [0, 0],
                    "failure_reason": visit.registration.failure_reason,
                    "selected_keyframe_id": visit.registration.selected_keyframe_id,
                    "is_direct_to_canonical": visit.registration.is_direct_to_canonical,
                    "spatial_coverage": visit.registration.spatial_coverage,
                    "decision_status": visit.registration.decision_status,
                    "decision_reason": visit.registration.decision_reason,
                    "is_promoted": visit.registration.is_promoted,
                }
            session_rec = Session(
                session_id=visit.session_id,
                session_name=visit.session_name,
                timestamp=visit_ts,
                image_path=f"memory://{visit.session_id}.png",
                notes=visit.notes,
                metadata=sess_meta,
            )
            self.store.add_session(session_rec)

        # 4. Canonicalize bounding boxes for query detections if registration succeeded and is reliable
        aligned_detections: List[Detection] = []
        raw_dets_to_store: List[RawDetection] = []
        for idx, det in enumerate(visit.detections):
            canon_box = det.canonical_bbox
            if (
                canon_box is None
                and is_reg_reliable
                and visit.registration.homography is not None
            ):
                canon_box = transform_bbox_to_canonical(
                    det.raw_bbox,
                    visit.registration.homography,
                    ref_size=visit.registration.ref_size,
                )
            elif not is_reg_reliable:
                # Do not trust pre-populated or unverified projected coordinates from unreliable registration
                canon_box = None

            aligned_detections.append(
                Detection(
                    label=det.label,
                    confidence=det.confidence,
                    raw_bbox=det.raw_bbox,
                    canonical_bbox=canon_box,
                    signature=det.signature,
                    metadata=det.metadata,
                )
            )

            raw_dets_to_store.append(
                RawDetection(
                    detection_id=f"raw_{visit.session_id}_{idx + 1:03d}",
                    session_id=visit.session_id,
                    timestamp=visit_ts,
                    label=det.label,
                    confidence=det.confidence,
                    raw_bbox=det.raw_bbox,
                    canonical_bbox=canon_box if is_reg_reliable else None,
                    registration_status=reg_status_str,
                    metadata=det.metadata,
                )
            )

        # Persist raw sensor detections independently of registration success
        self.store.add_raw_detections(raw_dets_to_store)

        # 5. Retrieve all active entities from memory
        active_entities = self.store.list_entities()

        # Build signature cache for entities from their last known observation
        entity_sigs = self._load_entity_latest_signatures(active_entities)

        # 6. Associate detections with existing entities one-to-one
        matches, unmatched_detections = self.associator.associate(
            entities=active_entities,
            detections=aligned_detections,
            registration=visit.registration,
            entity_signatures=entity_sigs,
        )

        match_by_entity = {m.entity_id: m for m in matches if m.entity_id}

        # 7. Update beliefs for existing entities
        entity_updates: List[EntityBeliefUpdate] = []
        transitions_count = 0

        for ent in active_entities:
            assoc = match_by_entity.get(ent.entity_id)
            update = self._update_entity_belief(
                entity=ent,
                assoc=assoc,
                visit=visit,
                aligned_detections=aligned_detections,
            )
            entity_updates.append(update)
            if update.transition_occurred:
                transitions_count += 1

        # 8. Create new entities for unmatched candidate detections
        new_entities_count = 0
        for det in unmatched_detections:
            # We create a new persistent entity only if registration succeeded and is reliable,
            # or if this is the very first reference visit
            is_initial_ref = len(active_entities) == 0 and is_reg_reliable
            if (is_reg_reliable and det.canonical_bbox is not None) or is_initial_ref:
                new_ent_id = f"ent_{visit.session_id}_{new_entities_count + 1}"
                new_canon = det.canonical_bbox or (det.raw_bbox if is_initial_ref else None)
                canon_list = [new_canon.x_min, new_canon.y_min, new_canon.x_max, new_canon.y_max] if new_canon else None
                new_ent = WorkbenchEntity(
                    entity_id=new_ent_id,
                    label=det.label,
                    created_at=visit_ts,
                    canonical_bbox=new_canon,
                    current_state=EpistemicState.CONFIRMED_PRESENT,
                    last_seen_session_id=visit.session_id,
                    metadata={
                        "created_in_session": visit.session_id,
                        "initial_confidence": det.confidence,
                        "initial_canonical_bbox": canon_list,
                        "last_reliable_canonical_bbox": canon_list,
                        "is_clipped": det.metadata.get("is_clipped", False),
                        "vacant_streak": 0,
                    },
                )
                self.store.add_entity(new_ent)
                new_entities_count += 1

                sig_vec = None
                if det.signature is not None and det.signature.combined_vector is not None:
                    sig_vec = det.signature.combined_vector.tolist()

                # Record observation
                obs_id = f"obs_{visit.session_id}_{new_ent_id}"
                obs = Observation(
                    observation_id=obs_id,
                    session_id=visit.session_id,
                    entity_id=new_ent_id,
                    timestamp=visit_ts,
                    image_reference=f"memory://{visit.session_id}.png",
                    raw_bbox=det.raw_bbox,
                    canonical_bbox=new_canon,
                    visibility_status=VisibilityStatus.CLEARLY_VISIBLE,
                    detector_confidence=det.confidence,
                    identity_match_score=1.0,
                    alignment_quality=visit.registration.inlier_ratio,
                    appearance_descriptor_ref=f"sig_{obs_id}",
                    raw_evidence={
                        "is_initial_detection": True,
                        "signature_vector": sig_vec,
                    },
                )
                self.store.add_observation(obs)

                # Record initial discovery event
                event_id = f"evt_{visit.session_id}_{new_ent_id}_discovery"
                event = ChangeEvent(
                    event_id=event_id,
                    entity_id=new_ent_id,
                    from_session_id=None,
                    to_session_id=visit.session_id,
                    timestamp=visit_ts,
                    prior_state=EpistemicState.INSUFFICIENT_EVIDENCE,
                    new_state=EpistemicState.CONFIRMED_PRESENT,
                    source="TEMPORAL_MEMORY",
                    explanation=f"New object '{det.label}' discovered on workbench.",
                    supporting_evidence={"confidence": det.confidence},
                )
                # The entity was inserted with CONFIRMED_PRESENT, so we record event as audit
                self.store.add_change_event(event)

        report = VisitProcessingReport(
            session_id=visit.session_id,
            session_name=visit.session_name,
            timestamp=visit_ts,
            registration_status=visit.registration.status.value,
            total_detections=len(visit.detections),
            matched_detections=len(aligned_detections) - len(unmatched_detections),
            new_entities_created=new_entities_count,
            state_transitions_count=transitions_count,
            entity_updates=entity_updates,
            unassociated_detections=unmatched_detections,
            is_reprocessed=False,
            details={"registration_success": visit.registration.success},
        )
        return report

    # -------------------------------------------------------------------------
    # Internal Belief Update Logic
    # -------------------------------------------------------------------------

    def _evaluate_displacement(
        self,
        entity: WorkbenchEntity,
        det: Detection,
        visit: VisitInput,
        initial_bbox: Optional[BoundingBox],
        last_reliable_bbox: Optional[BoundingBox],
        det_canon: Optional[BoundingBox],
    ) -> Tuple[Optional[float], Optional[float], bool, bool, bool, EpistemicState, str]:
        """
        Evaluates spatial displacement separating consecutive movement and baseline displacement.

        Returns:
            Tuple of:
            (
                consecutive_disp,
                baseline_disp,
                is_consecutive_moved,
                is_return_to_baseline,
                is_spatial_reliable,
                target_state,
                explanation,
            )
        """
        # 1. Global registration check
        if not self._is_registration_reliable(visit.registration):
            return None, None, False, False, False, entity.current_state, "Registration unverified or failed"

        if det_canon is None:
            return None, None, False, False, False, entity.current_state, "Detection lacks canonical bounding box projection"

        ref_box = last_reliable_bbox or initial_bbox
        if ref_box is None:
            return None, None, False, False, False, entity.current_state, "Entity lacks baseline canonical bounding box"

        # 2. Check overlap fraction with registered field of view
        is_spatial_ok, spatial_reason = self._is_spatial_comparison_reliable(
            entity=entity,
            det=det,
            visit=visit,
            initial_bbox=initial_bbox,
            last_reliable_bbox=last_reliable_bbox,
        )
        if not is_spatial_ok:
            return None, None, False, False, False, entity.current_state, spatial_reason

        # 3. Check background / environmental support class exclusion
        if (
            entity.label in self.config.excluded_background_classes
            or det.label in self.config.excluded_background_classes
        ):
            c_disp = calculate_centroid_distance(ref_box, det_canon)
            b_disp = calculate_centroid_distance(initial_bbox, det_canon) if initial_bbox else c_disp
            return (
                c_disp,
                b_disp,
                False,
                False,
                True,
                EpistemicState.CONFIRMED_PRESENT,
                f"Background/support class '{det.label}' excluded from movement state transitions",
            )

        # 4. Compute boundary clipping flags
        qw, qh = (
            (visit.image.shape[1], visit.image.shape[0])
            if hasattr(visit, "image") and visit.image is not None and visit.image.ndim >= 2
            else visit.registration.query_size
        )
        is_current_clipped = self._is_bbox_clipped(det.raw_bbox, (qw, qh))

        rw, rh = visit.registration.ref_size
        is_ref_clipped = self._is_bbox_clipped(ref_box, (rw, rh)) or bool((entity.metadata or {}).get("is_clipped", False))
        is_init_clipped = (self._is_bbox_clipped(initial_bbox, (rw, rh)) if initial_bbox else False) or bool((entity.metadata or {}).get("is_clipped", False))

        # A boundary truncation artifact in the current visit can only occur if the
        # current detection is clipped by the image edge. If the current detection is
        # fully inside the interior, its centroid is not truncated by camera framing.
        is_boundary_affected = is_current_clipped

        # 5. Compute displacement metrics and dynamic uncertainty threshold
        t_effective, delta_u, u_breakdown = self.compute_effective_movement_threshold(
            visit.registration,
            self.config.movement_displacement_threshold,
        )

        disp_initial = calculate_centroid_distance(initial_bbox, det_canon) if initial_bbox else None
        disp_last = calculate_centroid_distance(last_reliable_bbox, det_canon) if last_reliable_bbox else None
        iou_initial = calculate_bbox_iou(initial_bbox, det_canon) if initial_bbox else 0.0
        iou_last = calculate_bbox_iou(last_reliable_bbox, det_canon) if last_reliable_bbox else 0.0

        consecutive_disp = disp_last if disp_last is not None else (disp_initial if disp_initial is not None else 0.0)
        baseline_disp = disp_initial if disp_initial is not None else consecutive_disp

        # 6. Evaluate Return to Baseline
        was_previously_moved = (entity.current_state == EpistemicState.CONFIRMED_MOVED) or (
            last_reliable_bbox is not None
            and initial_bbox is not None
            and ((calculate_centroid_distance(initial_bbox, last_reliable_bbox) or 0.0) > self.config.movement_displacement_threshold)
        )
        is_back_at_baseline = (baseline_disp <= t_effective) or (iou_initial >= 0.40)
        is_step_moved = consecutive_disp > t_effective

        if was_previously_moved and is_back_at_baseline and is_step_moved:
            expl = (
                f"Object returned to initial baseline position "
                f"(moved {consecutive_disp:.1f}px since last observation; baseline displacement: {baseline_disp:.1f}px <= {t_effective:.1f}px)."
            )
            return consecutive_disp, baseline_disp, True, True, True, EpistemicState.CONFIRMED_PRESENT, expl

        # 7. Evaluate boundary clipping branch
        if is_boundary_affected:
            if (
                baseline_disp < self.config.clipped_movement_displacement_threshold
                and consecutive_disp < self.config.clipped_movement_displacement_threshold
            ):
                if entity.current_state == EpistemicState.CONFIRMED_REMOVED:
                    expl = f"Object re-introduced to workbench (re-detected with confidence {det.confidence:.2f})."
                    return consecutive_disp, baseline_disp, False, False, True, EpistemicState.CONFIRMED_PRESENT, expl

                if entity.current_state == EpistemicState.PENDING_REMOVAL:
                    expl = f"Re-detected near prior position (step: {consecutive_disp:.1f}px) despite boundary contact; restored to CONFIRMED_PRESENT."
                    return consecutive_disp, baseline_disp, False, False, True, EpistemicState.CONFIRMED_PRESENT, expl

                expl = (
                    f"Detection clipped by image boundary ([{det.raw_bbox.x_min:.0f}, {det.raw_bbox.y_min:.0f}, "
                    f"{det.raw_bbox.x_max:.0f}, {det.raw_bbox.y_max:.0f}] within {self.config.border_margin_px:.0f}px "
                    f"of {qw}x{qh}); displacement ({baseline_disp:.1f}px < {self.config.clipped_movement_displacement_threshold:.1f}px) "
                    f"treated as boundary truncation artifact."
                )
                return consecutive_disp, baseline_disp, False, False, False, entity.current_state, expl
            else:
                # Genuine large movement exceeding clipped threshold
                if entity.current_state == EpistemicState.CONFIRMED_REMOVED:
                    expl = f"Object re-introduced to workbench (re-detected with confidence {det.confidence:.2f})."
                    return consecutive_disp, baseline_disp, False, False, True, EpistemicState.CONFIRMED_PRESENT, expl

                expl = (
                    f"Object relocated by {baseline_disp:.1f}px to new canonical position "
                    f"(large displacement >= {self.config.clipped_movement_displacement_threshold:.1f}px despite boundary contact)."
                )
                return consecutive_disp, baseline_disp, True, False, True, EpistemicState.CONFIRMED_MOVED, expl

        # 8. Unclipped interior object branch
        # Case A: Returning from CONFIRMED_REMOVED (must re-enter as CONFIRMED_PRESENT per transition policy)
        if entity.current_state == EpistemicState.CONFIRMED_REMOVED:
            expl = f"Object re-introduced to workbench (re-detected with confidence {det.confidence:.2f}; baseline displacement: {baseline_disp:.1f}px)."
            return consecutive_disp, baseline_disp, False, (baseline_disp <= t_effective), True, EpistemicState.CONFIRMED_PRESENT, expl

        # Case B: Returning from PENDING_REMOVAL
        if entity.current_state == EpistemicState.PENDING_REMOVAL:
            if (
                baseline_disp <= t_effective
                or consecutive_disp <= t_effective
                or iou_last >= 0.40
            ):
                expl = f"Re-detected near prior position (step displacement: {consecutive_disp:.1f}px <= {t_effective:.1f}px)."
                return consecutive_disp, baseline_disp, False, False, True, EpistemicState.CONFIRMED_PRESENT, expl
            else:
                expl = f"Re-detected at new location (displacement from last: {consecutive_disp:.1f}px, baseline: {baseline_disp:.1f}px)."
                return consecutive_disp, baseline_disp, True, False, True, EpistemicState.CONFIRMED_MOVED, expl

        # Case C: Object currently in CONFIRMED_MOVED
        if entity.current_state == EpistemicState.CONFIRMED_MOVED:
            if consecutive_disp > t_effective:
                expl = f"Object moved to another relocated position (step: {consecutive_disp:.1f}px, baseline: {baseline_disp:.1f}px)."
                return consecutive_disp, baseline_disp, True, False, True, EpistemicState.CONFIRMED_MOVED, expl
            else:
                expl = f"Object remains at relocated position (baseline displacement: {baseline_disp:.1f}px, step: {consecutive_disp:.1f}px)."
                return consecutive_disp, baseline_disp, False, False, True, EpistemicState.CONFIRMED_MOVED, expl

        # Case D: Object in CONFIRMED_PRESENT or CONFIRMED_ADDED
        if baseline_disp <= self.config.movement_displacement_threshold:
            expl = f"Object remains present (displacement {baseline_disp:.1f}px <= {self.config.movement_displacement_threshold:.1f}px)."
            return consecutive_disp, baseline_disp, False, False, True, EpistemicState.CONFIRMED_PRESENT, expl

        if baseline_disp <= t_effective:
            expl = (
                f"Displacement ({baseline_disp:.1f}px) within registration uncertainty margin "
                f"(effective threshold: {t_effective:.1f}px = {self.config.movement_displacement_threshold:.1f}px "
                f"+ {delta_u:.1f}px uncertainty); preserved as CONFIRMED_PRESENT."
            )
            return consecutive_disp, baseline_disp, False, False, True, EpistemicState.CONFIRMED_PRESENT, expl

        # Check for gradual coordinate drift without physical movement
        if (
            iou_initial >= self.config.drift_iou_threshold
            and consecutive_disp <= self.config.max_drift_step_displacement
        ):
            expl = (
                f"Coordinate drift without physical movement "
                f"(accumulated {baseline_disp:.1f}px, step {consecutive_disp:.1f}px <= {self.config.max_drift_step_displacement:.1f}px, "
                f"IoU {iou_initial:.2f} >= {self.config.drift_iou_threshold:.2f})."
            )
            return consecutive_disp, baseline_disp, False, False, True, EpistemicState.CONFIRMED_PRESENT, expl

        # Genuine relocation from baseline
        expl = f"Object relocated by {baseline_disp:.1f}px to new canonical position (exceeds effective threshold {t_effective:.1f}px)."
        return consecutive_disp, baseline_disp, True, False, True, EpistemicState.CONFIRMED_MOVED, expl

    def _process_matched_detection(
        self,
        entity: WorkbenchEntity,
        assoc: AssociationMatch,
        visit: VisitInput,
        aligned_detections: List[Detection],
        ent_meta: Dict[str, Any],
    ) -> EntityBeliefUpdate:
        """
        Processes an associated detection for an existing entity, evaluates displacement,
        and applies the epistemic state transition policy.
        """
        visit_ts = normalize_iso_timestamp(visit.timestamp)
        det = assoc.detection
        assert det is not None

        initial_bbox = None
        if "initial_canonical_bbox" in ent_meta and ent_meta["initial_canonical_bbox"]:
            initial_bbox = BoundingBox(*ent_meta["initial_canonical_bbox"])
        elif entity.canonical_bbox is not None:
            initial_bbox = entity.canonical_bbox

        last_reliable_bbox = None
        if "last_reliable_canonical_bbox" in ent_meta and ent_meta["last_reliable_canonical_bbox"]:
            last_reliable_bbox = BoundingBox(*ent_meta["last_reliable_canonical_bbox"])
        elif entity.canonical_bbox is not None:
            last_reliable_bbox = entity.canonical_bbox

        is_reg_reliable = self._is_registration_reliable(visit.registration)

        (
            consecutive_disp,
            baseline_disp,
            is_consecutive_moved,
            is_return_to_baseline,
            is_spatial_reliable,
            target_state,
            explanation,
        ) = self._evaluate_displacement(
            entity=entity,
            det=det,
            visit=visit,
            initial_bbox=initial_bbox,
            last_reliable_bbox=last_reliable_bbox,
            det_canon=det.canonical_bbox,
        )

        det_canon_to_store = det.canonical_bbox if is_spatial_reliable else None

        sig_vec = None
        if det.signature is not None and det.signature.combined_vector is not None:
            sig_vec = det.signature.combined_vector.tolist()

        obs_id = f"obs_{visit.session_id}_{entity.entity_id}"
        vis_status = (
            VisibilityStatus.CLEARLY_VISIBLE
            if is_spatial_reliable
            else VisibilityStatus.UNCERTAIN
        )

        # Dynamic uncertainty evaluation
        _, _, u_breakdown = self.compute_effective_movement_threshold(
            visit.registration,
            self.config.movement_displacement_threshold,
        )

        # Primary displacement distance for audit and observations
        if is_return_to_baseline:
            displacement = consecutive_disp
        elif target_state == EpistemicState.CONFIRMED_MOVED:
            displacement = baseline_disp if baseline_disp is not None else consecutive_disp
        elif is_consecutive_moved:
            displacement = consecutive_disp
        else:
            displacement = baseline_disp if baseline_disp is not None else consecutive_disp

        obs = Observation(
            observation_id=obs_id,
            session_id=visit.session_id,
            entity_id=entity.entity_id,
            timestamp=visit_ts,
            image_reference=f"memory://{visit.session_id}.png",
            raw_bbox=det.raw_bbox,
            canonical_bbox=det_canon_to_store,
            visibility_status=vis_status,
            detector_confidence=det.confidence,
            identity_match_score=assoc.confidence,
            alignment_quality=visit.registration.inlier_ratio,
            appearance_descriptor_ref=f"sig_{obs_id}",
            raw_evidence={
                "association_details": assoc.details,
                "signature_vector": sig_vec,
                "registration_reliable": is_reg_reliable,
                "spatial_comparison_reliable": is_spatial_reliable,
                "spatial_unreliability_reason": explanation if not is_spatial_reliable else None,
                "displacement": displacement,
                "consecutive_displacement": consecutive_disp,
                "baseline_displacement": baseline_disp,
                "is_consecutive_moved": is_consecutive_moved,
                "is_return_to_baseline": is_return_to_baseline,
                "is_clipped": det.metadata.get("is_clipped", False),
                "is_cross_class": assoc.is_cross_class,
                "detected_label": det.label,
                "established_label": entity.label,
                "uncertainty_breakdown": u_breakdown,
            },
        )
        self.store.add_observation(obs)

        # Reset vacant streak and unobserved streak on detection
        ent_meta["vacant_streak"] = 0
        ent_meta["unobserved_streak"] = 0
        if is_spatial_reliable and det.canonical_bbox is not None:
            ent_meta["last_reliable_canonical_bbox"] = [
                det.canonical_bbox.x_min,
                det.canonical_bbox.y_min,
                det.canonical_bbox.x_max,
                det.canonical_bbox.y_max,
            ]
            ent_meta["is_clipped"] = det.metadata.get("is_clipped", False)

        if assoc.is_cross_class:
            mismatches = ent_meta.setdefault("classification_mismatches", [])
            mismatches.append({
                "session_id": visit.session_id,
                "detected_label": det.label,
                "established_label": entity.label,
                "confidence": det.confidence,
            })

        self._update_entity_metadata(entity.entity_id, ent_meta)

        if target_state != entity.current_state or is_consecutive_moved or is_return_to_baseline:
            event_id = f"evt_{visit.session_id}_{entity.entity_id}_{target_state.value.lower()}"
            if is_return_to_baseline:
                event_disp = consecutive_disp
            elif is_consecutive_moved or target_state == EpistemicState.CONFIRMED_MOVED:
                event_disp = consecutive_disp if is_consecutive_moved else baseline_disp
            else:
                event_disp = 0.0 if is_spatial_reliable else None

            event = ChangeEvent(
                event_id=event_id,
                entity_id=entity.entity_id,
                from_session_id=entity.last_seen_session_id,
                to_session_id=visit.session_id,
                timestamp=visit_ts,
                prior_state=entity.current_state,
                new_state=target_state,
                source="TEMPORAL_MEMORY",
                displacement_distance=event_disp,
                supporting_evidence={
                    "displacement": event_disp,
                    "consecutive_displacement": consecutive_disp,
                    "baseline_displacement": baseline_disp,
                    "is_consecutive_moved": is_consecutive_moved,
                    "is_return_to_baseline": is_return_to_baseline,
                    "confidence": det.confidence,
                    "match_confidence": assoc.confidence,
                    "registration_reliable": is_reg_reliable,
                    "spatial_comparison_reliable": is_spatial_reliable,
                    "is_cross_class": assoc.is_cross_class,
                    "detected_label": det.label,
                    "established_label": entity.label,
                    "uncertainty_breakdown": u_breakdown,
                },
                explanation=explanation,
            )
            self.store.record_state_transition(
                entity_id=entity.entity_id,
                new_state=target_state,
                event=event,
                canonical_bbox=det_canon_to_store,
                last_seen_session_id=visit.session_id,
            )
            return EntityBeliefUpdate(
                entity_id=entity.entity_id,
                label=entity.label,
                prior_state=entity.current_state,
                new_state=target_state,
                transition_occurred=(target_state != entity.current_state),
                observation_id=obs_id,
                event_id=event_id,
                visibility_status=vis_status,
                explanation=explanation,
                consecutive_vacant_visits=0,
                supporting_evidence={
                    "consecutive_displacement": consecutive_disp,
                    "baseline_displacement": baseline_disp,
                    "is_consecutive_moved": is_consecutive_moved,
                    "is_return_to_baseline": is_return_to_baseline,
                },
            )
        else:
            # Update last seen session and canonical bbox (only if reliable)
            self._touch_entity(entity.entity_id, visit.session_id, det_canon_to_store)
            diag_event_id: Optional[str] = None
            if assoc.is_cross_class:
                diag_event_id = f"evt_{visit.session_id}_{entity.entity_id}_class_mismatch"
                sim_str = f"{assoc.signature_similarity:.2f}" if assoc.signature_similarity is not None else "N/A"
                diag_event = ChangeEvent(
                    event_id=diag_event_id,
                    entity_id=entity.entity_id,
                    from_session_id=entity.last_seen_session_id,
                    to_session_id=visit.session_id,
                    timestamp=visit_ts,
                    prior_state=entity.current_state,
                    new_state=entity.current_state,
                    source="TEMPORAL_MEMORY",
                    displacement_distance=displacement if displacement is not None else 0.0,
                    supporting_evidence={
                        "classification_mismatch": True,
                        "is_cross_class": True,
                        "detected_label": det.label,
                        "established_label": entity.label,
                        "confidence": det.confidence,
                        "signature_similarity": assoc.signature_similarity,
                        "spatial_iou": assoc.spatial_iou,
                    },
                    explanation=(
                        f"Classification mismatch observed: detected as '{det.label}' "
                        f"(confidence {det.confidence:.2f}) but matched to established entity '{entity.label}' "
                        f"(similarity: {sim_str}, IoU: {assoc.spatial_iou if assoc.spatial_iou is not None else 0.0:.2f})."
                    ),
                )
                self.store.add_change_event(diag_event)

            return EntityBeliefUpdate(
                entity_id=entity.entity_id,
                label=entity.label,
                prior_state=entity.current_state,
                new_state=entity.current_state,
                transition_occurred=False,
                observation_id=obs_id,
                event_id=diag_event_id,
                visibility_status=vis_status,
                explanation=explanation,
                consecutive_vacant_visits=0,
            )

    def _update_entity_belief(
        self,
        entity: WorkbenchEntity,
        assoc: Optional[AssociationMatch],
        visit: VisitInput,
        aligned_detections: List[Detection],
    ) -> EntityBeliefUpdate:
        """
        Evaluates observational evidence in current visit and performs atomic state
        transition if mandated by policy.
        """
        visit_ts = normalize_iso_timestamp(visit.timestamp)
        ent_meta = entity.metadata or {}
        vacant_streak = int(ent_meta.get("vacant_streak", 0))

        # CASE 1: Object successfully detected and associated in this visit
        if assoc is not None and assoc.is_associated and assoc.detection is not None:
            return self._process_matched_detection(
                entity=entity,
                assoc=assoc,
                visit=visit,
                aligned_detections=aligned_detections,
                ent_meta=ent_meta,
            )

        # CASE 2: Ambiguous identity association
        if assoc is not None and assoc.is_ambiguous:
            # Ambiguous association: transition to AMBIGUOUS_IDENTITY if not already
            target_state = EpistemicState.AMBIGUOUS_IDENTITY
            if target_state != entity.current_state:
                event_id = f"evt_{visit.session_id}_{entity.entity_id}_ambiguous"
                expl = assoc.explanation
                event = ChangeEvent(
                    event_id=event_id,
                    entity_id=entity.entity_id,
                    from_session_id=entity.last_seen_session_id,
                    to_session_id=visit.session_id,
                    timestamp=visit_ts,
                    prior_state=entity.current_state,
                    new_state=target_state,
                    source="TEMPORAL_MEMORY",
                    supporting_evidence=assoc.details,
                    explanation=expl,
                )
                self.store.record_state_transition(
                    entity_id=entity.entity_id,
                    new_state=target_state,
                    event=event,
                )
                return EntityBeliefUpdate(
                    entity_id=entity.entity_id,
                    label=entity.label,
                    prior_state=entity.current_state,
                    new_state=target_state,
                    transition_occurred=True,
                    event_id=event_id,
                    visibility_status=VisibilityStatus.UNCERTAIN,
                    explanation=expl,
                    consecutive_vacant_visits=vacant_streak,
                )
            return EntityBeliefUpdate(
                entity_id=entity.entity_id,
                label=entity.label,
                prior_state=entity.current_state,
                new_state=entity.current_state,
                transition_occurred=False,
                visibility_status=VisibilityStatus.UNCERTAIN,
                explanation="Ambiguous identity persists.",
                consecutive_vacant_visits=vacant_streak,
            )

        # CASE 3: Object NOT detected in current visit -> Evaluate evidence of absence
        # Safeguards:
        # A. Registration failure or low inlier ratio -> Cannot trust spatial evidence; do not progress removal!
        if (
            not self._is_registration_reliable(visit.registration)
            or visit.registration.overlap_mask is None
        ):
            return EntityBeliefUpdate(
                entity_id=entity.entity_id,
                label=entity.label,
                prior_state=entity.current_state,
                new_state=entity.current_state,
                transition_occurred=False,
                visibility_status=VisibilityStatus.UNOBSERVED,
                explanation="Registration unverified or inlier ratio below threshold; absence cannot be confirmed.",
                consecutive_vacant_visits=vacant_streak,
            )

        canon_box = entity.canonical_bbox
        if canon_box is None:
            return EntityBeliefUpdate(
                entity_id=entity.entity_id,
                label=entity.label,
                prior_state=entity.current_state,
                new_state=entity.current_state,
                transition_occurred=False,
                visibility_status=VisibilityStatus.UNOBSERVED,
                explanation="Entity has no canonical bounding box.",
                consecutive_vacant_visits=vacant_streak,
            )

        # B. Check valid registration overlap
        overlap_frac = check_bbox_overlap_fraction(canon_box, visit.registration.overlap_mask)
        if overlap_frac < self.config.min_overlap_fraction:
            # Canonical region is out-of-frame or clipped
            return EntityBeliefUpdate(
                entity_id=entity.entity_id,
                label=entity.label,
                prior_state=entity.current_state,
                new_state=entity.current_state,
                transition_occurred=False,
                visibility_status=VisibilityStatus.OUT_OF_FRAME,
                explanation=f"Canonical location is outside overlap ({overlap_frac:.1%}); cannot confirm absence.",
                consecutive_vacant_visits=vacant_streak,
            )

        # C. Check occlusion by other detected objects
        for det in aligned_detections:
            d_box = det.canonical_bbox
            if d_box is not None:
                iou = calculate_bbox_iou(canon_box, d_box)
                if iou > 0.20:
                    # Partially occluded by another object!
                    return EntityBeliefUpdate(
                        entity_id=entity.entity_id,
                        label=entity.label,
                        prior_state=entity.current_state,
                        new_state=entity.current_state,
                        transition_occurred=False,
                        visibility_status=VisibilityStatus.PARTIALLY_OCCLUDED,
                        explanation=f"Canonical location occluded by detection '{det.label}' (IoU: {iou:.2f}).",
                        consecutive_vacant_visits=vacant_streak,
                    )

        # D. Vacancy verification on aligned image patch
        is_vacant = False
        aligned_img = visit.registration.aligned_image
        if aligned_img is not None:
            # Check patch appearance
            patch_sig = self.signature_extractor.extract(aligned_img, canon_box)
            ent_sigs = self._load_entity_latest_signatures([entity])
            ent_sig = ent_sigs.get(entity.entity_id)

            if patch_sig.is_valid and ent_sig is not None and ent_sig.is_valid:
                patch_sim = ent_sig.cosine_similarity(patch_sig)
                if patch_sim < self.config.vacancy_similarity_max:
                    is_vacant = True

        unobserved_streak = int(ent_meta.get("unobserved_streak", 0)) + 1
        ent_meta["unobserved_streak"] = unobserved_streak

        if is_vacant:
            vacant_streak += 1
            ent_meta["vacant_streak"] = vacant_streak
            ent_meta["unobserved_streak"] = int(ent_meta.get("unobserved_streak", 0)) + 1
            self._update_entity_metadata(entity.entity_id, ent_meta)

            # Evaluate removal progression with verified vacant patch:
            # Visit 1: CONFIRMED_PRESENT -> PENDING_REMOVAL
            # Visit 2+: PENDING_REMOVAL -> CONFIRMED_REMOVED
            if entity.current_state in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED):
                target_state = EpistemicState.PENDING_REMOVAL
            elif entity.current_state == EpistemicState.PENDING_REMOVAL:
                if vacant_streak >= self.config.vacancy_visits_for_removal:
                    target_state = EpistemicState.CONFIRMED_REMOVED
                else:
                    target_state = EpistemicState.PENDING_REMOVAL
            else:
                target_state = entity.current_state

            expl = (
                f"Vacant patch verified for {vacant_streak} consecutive visit(s); "
                f"transitioning to {target_state.value}."
            )
        else:
            # Patch unverified (patch similarity high or image patch unavailable):
            self._update_entity_metadata(entity.entity_id, ent_meta)
            unobserved_thresh = getattr(self.config, "unobserved_visits_for_removal", 3)
            if unobserved_streak == 1:
                # First missed detection with object-like patch appearance:
                # Preserve current state (suppress false removal on detector false negative)
                target_state = entity.current_state
                expl = (
                    "Non-detection without verified vacancy cannot confirm removal; "
                    "patch appearance still resembles object."
                )
            elif unobserved_streak == 2:
                # Second consecutive missed visit: transition to PENDING_REMOVAL
                if entity.current_state in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED):
                    target_state = EpistemicState.PENDING_REMOVAL
                    expl = (
                        f"Object not detected for {unobserved_streak} consecutive reliably registered visits "
                        f"(patch unverified); transitioning to PENDING_REMOVAL."
                    )
                else:
                    target_state = entity.current_state
                    expl = f"Object remains in {entity.current_state.value}."
            else:
                # 3+ consecutive missed visits: transition to CONFIRMED_REMOVED
                if entity.current_state == EpistemicState.PENDING_REMOVAL and unobserved_streak >= unobserved_thresh:
                    target_state = EpistemicState.CONFIRMED_REMOVED
                    expl = (
                        f"Object unobserved for {unobserved_streak} consecutive reliably registered visits; "
                        f"confirmed removed."
                    )
                else:
                    target_state = entity.current_state
                    expl = f"Object remains in {entity.current_state.value}."

        if target_state != entity.current_state:
            event_id = f"evt_{visit.session_id}_{entity.entity_id}_{target_state.value.lower()}"
            event = ChangeEvent(
                event_id=event_id,
                entity_id=entity.entity_id,
                from_session_id=entity.last_seen_session_id,
                to_session_id=visit.session_id,
                timestamp=visit_ts,
                prior_state=entity.current_state,
                new_state=target_state,
                source="TEMPORAL_MEMORY",
                supporting_evidence={
                    "consecutive_vacant_visits": vacant_streak,
                    "unobserved_streak": unobserved_streak,
                    "vacancy_verified": is_vacant,
                },
                explanation=expl,
            )
            self.store.record_state_transition(
                entity_id=entity.entity_id,
                new_state=target_state,
                event=event,
            )
            return EntityBeliefUpdate(
                entity_id=entity.entity_id,
                label=entity.label,
                prior_state=entity.current_state,
                new_state=target_state,
                transition_occurred=True,
                event_id=event_id,
                visibility_status=VisibilityStatus.CLEARLY_VISIBLE if is_vacant else VisibilityStatus.UNCERTAIN,
                explanation=expl,
                consecutive_vacant_visits=vacant_streak,
            )
        else:
            return EntityBeliefUpdate(
                entity_id=entity.entity_id,
                label=entity.label,
                prior_state=entity.current_state,
                new_state=entity.current_state,
                transition_occurred=False,
                visibility_status=VisibilityStatus.CLEARLY_VISIBLE if is_vacant else VisibilityStatus.UNCERTAIN,
                explanation=expl,
                consecutive_vacant_visits=vacant_streak,
            )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _load_entity_latest_signatures(
        self, entities: List[WorkbenchEntity]
    ) -> Dict[str, VisualSignature]:
        """Loads the most recent visual signature for each entity from its latest observation."""
        signatures: Dict[str, VisualSignature] = {}
        for ent in entities:
            # Query latest observation with an appearance vector
            cursor = self.store.conn.execute(
                """
                SELECT raw_evidence_json FROM observations 
                WHERE entity_id = ? 
                ORDER BY timestamp DESC LIMIT 1;
                """,
                (ent.entity_id,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                import json
                try:
                    data = json.loads(row[0])
                    # If we stored signature vector in metadata/evidence
                    sig_vec = data.get("signature_vector")
                    if sig_vec and len(sig_vec) == 148:
                        arr = np.array(sig_vec, dtype=np.float32)
                        signatures[ent.entity_id] = VisualSignature(
                            color_histogram=arr[:128],
                            texture_descriptor=arr[128:],
                            combined_vector=arr,
                        )
                except Exception:
                    pass
        return signatures

    def _update_entity_metadata(self, entity_id: str, metadata: Dict[str, Any]) -> None:
        """Helper updating an entity's metadata column safely."""
        import json
        with self.store.conn:
            self.store.conn.execute(
                "UPDATE entities SET metadata_json = ? WHERE entity_id = ?;",
                (json.dumps(metadata), entity_id),
            )

    def _touch_entity(
        self,
        entity_id: str,
        session_id: str,
        canonical_bbox: Optional[BoundingBox],
    ) -> None:
        """Updates last_seen_session_id and canonical_bbox without changing state."""
        import json
        updates = ["last_seen_session_id = ?"]
        params: List[Any] = [session_id]
        if canonical_bbox is not None:
            updates.append("canonical_bbox_json = ?")
            params.append(json.dumps([canonical_bbox.x_min, canonical_bbox.y_min, canonical_bbox.x_max, canonical_bbox.y_max]))
        params.append(entity_id)
        with self.store.conn:
            self.store.conn.execute(
                f"UPDATE entities SET {', '.join(updates)} WHERE entity_id = ?;",
                params,
            )

    def _build_reprocessed_report(
        self, visit: VisitInput, existing_obs: List[Observation]
    ) -> VisitProcessingReport:
        """Builds idempotent reprocessed report when visit was already ingested."""
        events = self.store.conn.execute(
            "SELECT * FROM change_events WHERE to_session_id = ?;",
            (visit.session_id,),
        ).fetchall()
        return VisitProcessingReport(
            session_id=visit.session_id,
            session_name=visit.session_name,
            timestamp=normalize_iso_timestamp(visit.timestamp),
            registration_status=visit.registration.status.value,
            total_detections=len(existing_obs),
            matched_detections=len(existing_obs),
            new_entities_created=0,
            state_transitions_count=len(events),
            is_reprocessed=True,
            details={"reprocessed": True},
        )
