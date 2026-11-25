"""
Cross-visit entity association layer for ObjectMemory.

Associates newly detected objects in a visit with existing persistent entities.

Requirements:
1. One-to-one assignment within a visit.
2. Category compatibility enforcement.
3. Decoupled visual-signature similarity and canonical spatial overlap/distance.
4. Gated on signature validity: invalid signatures never yield valid identity matches.
5. Gated on registration quality: unaligned frames do not trust canonical spatial coordinates.
6. Ambiguous matches (near-identical candidates within ambiguity margin) produce explicit
   unresolved/ambiguous associations rather than false identity mergers.
7. Reuses optimal assignment (Kuhn-Munkres) with deterministic tie-breaking.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from objectmemory.baseline.comparator import (
    calculate_bbox_iou,
    calculate_centroid_distance,
    solve_optimal_assignment,
)
from objectmemory.detection.models import Detection
from objectmemory.domain.models import BoundingBox, WorkbenchEntity
from objectmemory.domain.states import EpistemicState
from objectmemory.registration.models import RegistrationResult
from objectmemory.temporal.models import AssociationMatch, TemporalEngineConfig


class CrossVisitEntityAssociator:
    """
    Solves one-to-one identity association between existing active workbench entities
    and newly observed detections in a visit frame.
    """

    def __init__(self, config: Optional[TemporalEngineConfig] = None) -> None:
        self.config = config or TemporalEngineConfig()

    def associate(
        self,
        entities: List[WorkbenchEntity],
        detections: List[Detection],
        registration: RegistrationResult,
        entity_signatures: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[AssociationMatch], List[Detection]]:
        """
        Associates candidates one-to-one.
        Returns:
        - List of AssociationMatch objects (one per active entity).
        - List of unmatched candidate detections (eligible for new entity creation).
        """
        entity_signatures = entity_signatures or {}
        n_ent = len(entities)
        n_det = len(detections)

        if n_ent == 0:
            return [], list(detections)

        if n_det == 0:
            # All entities are unobserved in this visit
            unmatched_matches = [
                AssociationMatch(
                    entity_id=ent.entity_id,
                    detection=None,
                    is_associated=False,
                    is_ambiguous=False,
                    confidence=0.0,
                    explanation="No detections present in visit.",
                )
                for ent in entities
            ]
            return unmatched_matches, []

        can_use_spatial = registration.success and (
            registration.inlier_ratio is None
            or registration.inlier_ratio >= self.config.min_registration_inlier_ratio
        )

        # 1. Compute pairwise affinity matrix: [n_ent, n_det]
        affinity_matrix = np.zeros((n_ent, n_det), dtype=float)
        iou_matrix = np.zeros((n_ent, n_det), dtype=float)
        dist_matrix = np.full((n_ent, n_det), np.inf, dtype=float)
        sim_matrix = np.zeros((n_ent, n_det), dtype=float)

        for i, ent in enumerate(entities):
            ent_sig = entity_signatures.get(ent.entity_id)

            for j, det in enumerate(detections):
                # A. Category compatibility: allow matching against confirmed label or known raw predicted labels
                ent_meta = ent.metadata or {}
                known_labels = set(ent_meta.get("raw_predicted_labels") or [])
                if ent.label:
                    known_labels.add(ent.label)
                if ent_meta.get("original_raw_label"):
                    known_labels.add(ent_meta["original_raw_label"])
                if known_labels and det.label and det.label not in known_labels:
                    continue

                # B. Visual signature similarity
                sim = 0.0
                has_valid_sig = False
                if ent_sig is not None and det.signature is not None:
                    if ent_sig.is_valid and det.signature.is_valid:
                        sim = float(ent_sig.cosine_similarity(det.signature))
                        has_valid_sig = True

                sim_matrix[i, j] = sim

                # C. Spatial alignment (only if registration succeeded)
                iou = 0.0
                dist = np.inf
                if can_use_spatial and ent.canonical_bbox is not None:
                    det_box = det.canonical_bbox
                    if det_box is not None:
                        iou = calculate_bbox_iou(ent.canonical_bbox, det_box)
                        d = calculate_centroid_distance(ent.canonical_bbox, det_box)
                        if d is not None:
                            dist = d

                iou_matrix[i, j] = iou
                dist_matrix[i, j] = dist

                # D. Composite affinity score:
                # If valid signatures exist, appearance is primary evidence.
                # Spatial proximity acts as corroboration or fallback.
                if has_valid_sig:
                    if sim >= self.config.signature_similarity_threshold:
                        # High appearance similarity: bonus if also collocated
                        spatial_bonus = 0.10 * iou if iou > 0.0 else 0.0
                        affinity_matrix[i, j] = min(1.0, sim + spatial_bonus)
                    else:
                        # Low appearance similarity: reject association
                        affinity_matrix[i, j] = 0.0
                else:
                    # No valid signature (e.g. initial visit or degenerate crop):
                    # Spatial proximity alone requires strong overlap (IoU >= threshold)
                    # and registration must be valid
                    if can_use_spatial and (
                        iou >= self.config.iou_match_threshold
                        or dist <= self.config.distance_match_threshold
                    ):
                        # Bounded confidence when appearance evidence is absent
                        affinity_matrix[i, j] = max(0.40, min(0.70, iou))
                    else:
                        affinity_matrix[i, j] = 0.0

        # 2. Check for ambiguity (multiple competing candidates within ambiguity margin)
        ambiguous_entities: Set[int] = set()
        for i in range(n_ent):
            affs = [affinity_matrix[i, j] for j in range(n_det) if affinity_matrix[i, j] > 0.0]
            if len(affs) >= 2:
                affs.sort(reverse=True)
                if (affs[0] - affs[1]) < self.config.ambiguity_margin:
                    ambiguous_entities.add(i)

        ambiguous_detections: Set[int] = set()
        for j in range(n_det):
            affs = [affinity_matrix[i, j] for i in range(n_ent) if affinity_matrix[i, j] > 0.0]
            if len(affs) >= 2:
                affs.sort(reverse=True)
                if (affs[0] - affs[1]) < self.config.ambiguity_margin:
                    ambiguous_detections.add(j)

        # 3. Solve one-to-one assignment
        matched_ent: Set[int] = set()
        matched_det: Set[int] = set()
        entity_match_map: Dict[int, AssociationMatch] = {}

        if self.config.assignment_strategy == "optimal":
            # Kuhn-Munkres minimum cost on (1.0 - affinity)
            cost_matrix = 1.0 - affinity_matrix
            optimal_pairs = solve_optimal_assignment(cost_matrix)
            candidate_pairs = [
                (i, j)
                for i, j in optimal_pairs
                if affinity_matrix[i, j] > 0.0
            ]
        else:
            # Greedy matching
            candidates = []
            for i in range(n_ent):
                for j in range(n_det):
                    aff = affinity_matrix[i, j]
                    if aff > 0.0:
                        candidates.append((aff, iou_matrix[i, j], -dist_matrix[i, j], i, j))
            candidates.sort(key=lambda t: (t[0], t[1], t[2], -t[3], -t[4]), reverse=True)
            candidate_pairs = [(i, j) for _, _, _, i, j in candidates]

        for i, j in candidate_pairs:
            if i in matched_ent or j in matched_det:
                continue

            ent = entities[i]
            det = detections[j]
            aff = float(affinity_matrix[i, j])
            iou = float(iou_matrix[i, j])
            dist = float(dist_matrix[i, j]) if math.isfinite(dist_matrix[i, j]) else None
            sim = float(sim_matrix[i, j])

            if i in ambiguous_entities or j in ambiguous_detections:
                # Declared ambiguous: do not merge identity silently!
                entity_match_map[i] = AssociationMatch(
                    entity_id=ent.entity_id,
                    detection=det,
                    is_associated=False,
                    is_ambiguous=True,
                    is_cross_class=False,
                    confidence=aff,
                    signature_similarity=sim,
                    spatial_iou=iou,
                    spatial_distance=dist,
                    explanation=(
                        f"Ambiguous identity between competing candidates (margin < "
                        f"{self.config.ambiguity_margin:.2f}); association unresolved."
                    ),
                    details={"competing_candidates": True, "affinity": aff},
                )
            else:
                entity_match_map[i] = AssociationMatch(
                    entity_id=ent.entity_id,
                    detection=det,
                    is_associated=True,
                    is_ambiguous=False,
                    is_cross_class=False,
                    confidence=aff,
                    signature_similarity=sim,
                    spatial_iou=iou,
                    spatial_distance=dist,
                    explanation=(
                        f"Identity matched to entity '{ent.entity_id}' (aff: {aff:.2f}, "
                        f"sim: {sim:.2f}, IoU: {iou:.2f})."
                    ),
                    details={"affinity": aff, "sim": sim, "iou": iou, "dist": dist},
                )

            matched_ent.add(i)
            matched_det.add(j)

        # 3b. Conservative cross-class fallback for unmatched candidates
        if self.config.enable_cross_class_association and can_use_spatial:
            unmatched_ent_indices = [i for i in range(n_ent) if i not in matched_ent]
            unmatched_det_indices = [j for j in range(n_det) if j not in matched_det]

            if unmatched_ent_indices and unmatched_det_indices:
                n_ue = len(unmatched_ent_indices)
                n_ud = len(unmatched_det_indices)
                cross_affinity_matrix = np.zeros((n_ue, n_ud), dtype=float)
                cross_iou_matrix = np.zeros((n_ue, n_ud), dtype=float)
                cross_dist_matrix = np.full((n_ue, n_ud), np.inf, dtype=float)
                cross_sim_matrix = np.full((n_ue, n_ud), np.nan, dtype=float)

                for u_i, ent_idx in enumerate(unmatched_ent_indices):
                    ent = entities[ent_idx]
                    if ent.canonical_bbox is None:
                        continue
                    ent_sig = entity_signatures.get(ent.entity_id)

                    for u_j, det_idx in enumerate(unmatched_det_indices):
                        det = detections[det_idx]
                        if det.canonical_bbox is None:
                            continue

                        # Check appearance signature
                        sim_val: Optional[float] = None
                        has_valid_sig = False
                        if ent_sig is not None and det.signature is not None:
                            if ent_sig.is_valid and det.signature.is_valid:
                                sim_val = float(ent_sig.cosine_similarity(det.signature))
                                has_valid_sig = True

                        if has_valid_sig and sim_val is not None:
                            cross_sim_matrix[u_i, u_j] = sim_val

                        # Check spatial overlap and distance
                        iou = calculate_bbox_iou(ent.canonical_bbox, det.canonical_bbox)
                        d = calculate_centroid_distance(ent.canonical_bbox, det.canonical_bbox)
                        dist = d if d is not None else np.inf

                        cross_iou_matrix[u_i, u_j] = iou
                        cross_dist_matrix[u_i, u_j] = dist

                        # Conservative cross-class qualification rules:
                        # Case 1: Appearance evidence available
                        if has_valid_sig and sim_val is not None:
                            if (
                                sim_val >= self.config.cross_class_signature_similarity_threshold
                                and (
                                    iou >= self.config.cross_class_iou_threshold
                                    or dist <= self.config.distance_match_threshold
                                )
                            ):
                                aff = min(0.95, sim_val * 0.70 + iou * 0.30)
                                cross_affinity_matrix[u_i, u_j] = aff
                            else:
                                cross_affinity_matrix[u_i, u_j] = 0.0
                        else:
                            # Case 2: Appearance evidence unavailable.
                            # Do NOT invent similarity scores. Requires high spatial coincidence.
                            if iou >= self.config.cross_class_fallback_iou_threshold:
                                aff = min(0.60, iou * 0.60)
                                cross_affinity_matrix[u_i, u_j] = aff
                            else:
                                cross_affinity_matrix[u_i, u_j] = 0.0

                # Ambiguity checks among cross-class candidates
                cross_ambig_ents: Set[int] = set()
                for u_i in range(n_ue):
                    affs = [cross_affinity_matrix[u_i, u_j] for u_j in range(n_ud) if cross_affinity_matrix[u_i, u_j] > 0.0]
                    if len(affs) >= 2:
                        affs.sort(reverse=True)
                        if (affs[0] - affs[1]) < self.config.cross_class_ambiguity_margin:
                            cross_ambig_ents.add(u_i)

                cross_ambig_dets: Set[int] = set()
                for u_j in range(n_ud):
                    affs = [cross_affinity_matrix[u_i, u_j] for u_i in range(n_ue) if cross_affinity_matrix[u_i, u_j] > 0.0]
                    if len(affs) >= 2:
                        affs.sort(reverse=True)
                        if (affs[0] - affs[1]) < self.config.cross_class_ambiguity_margin:
                            cross_ambig_dets.add(u_j)

                # Solve assignment on cross-class candidates
                if self.config.assignment_strategy == "optimal":
                    cross_cost = 1.0 - cross_affinity_matrix
                    cross_pairs = solve_optimal_assignment(cross_cost)
                    cross_candidates = [
                        (u_i, u_j)
                        for u_i, u_j in cross_pairs
                        if cross_affinity_matrix[u_i, u_j] > 0.0
                    ]
                else:
                    cross_cand_list = []
                    for u_i in range(n_ue):
                        for u_j in range(n_ud):
                            aff = cross_affinity_matrix[u_i, u_j]
                            if aff > 0.0:
                                cross_cand_list.append((aff, cross_iou_matrix[u_i, u_j], -cross_dist_matrix[u_i, u_j], u_i, u_j))
                    cross_cand_list.sort(key=lambda t: (t[0], t[1], t[2], -t[3], -t[4]), reverse=True)
                    cross_candidates = [(u_i, u_j) for _, _, _, u_i, u_j in cross_cand_list]

                for u_i, u_j in cross_candidates:
                    ent_idx = unmatched_ent_indices[u_i]
                    det_idx = unmatched_det_indices[u_j]

                    if ent_idx in matched_ent or det_idx in matched_det:
                        continue

                    ent = entities[ent_idx]
                    det = detections[det_idx]
                    aff = float(cross_affinity_matrix[u_i, u_j])
                    iou = float(cross_iou_matrix[u_i, u_j])
                    dist = float(cross_dist_matrix[u_i, u_j]) if math.isfinite(cross_dist_matrix[u_i, u_j]) else None
                    raw_sim = cross_sim_matrix[u_i, u_j]
                    sim_opt = float(raw_sim) if not np.isnan(raw_sim) else None

                    if u_i in cross_ambig_ents or u_j in cross_ambig_dets:
                        # Ambiguous cross-class candidate: do not merge!
                        entity_match_map[ent_idx] = AssociationMatch(
                            entity_id=ent.entity_id,
                            detection=det,
                            is_associated=False,
                            is_ambiguous=True,
                            is_cross_class=True,
                            confidence=aff,
                            signature_similarity=sim_opt,
                            spatial_iou=iou,
                            spatial_distance=dist,
                            explanation=(
                                f"Ambiguous cross-class identity candidate for entity '{ent.entity_id}' "
                                f"(detected as '{det.label}', established as '{ent.label}', margin < "
                                f"{self.config.cross_class_ambiguity_margin:.2f}); rejected."
                            ),
                            details={
                                "competing_candidates": True,
                                "affinity": aff,
                                "is_cross_class": True,
                                "detected_label": det.label,
                                "established_label": ent.label,
                            },
                        )
                    else:
                        # Accepted conservative cross-class match
                        sim_str = f"sim: {sim_opt:.2f}, " if sim_opt is not None else "sim: N/A, "
                        entity_match_map[ent_idx] = AssociationMatch(
                            entity_id=ent.entity_id,
                            detection=det,
                            is_associated=True,
                            is_ambiguous=False,
                            is_cross_class=True,
                            confidence=aff,
                            signature_similarity=sim_opt,
                            spatial_iou=iou,
                            spatial_distance=dist,
                            explanation=(
                                f"Cross-class identity match to entity '{ent.entity_id}' "
                                f"(detected as '{det.label}', established as '{ent.label}', "
                                f"aff: {aff:.2f}, {sim_str}IoU: {iou:.2f})."
                            ),
                            details={
                                "affinity": aff,
                                "sim": sim_opt,
                                "iou": iou,
                                "dist": dist,
                                "is_cross_class": True,
                                "detected_label": det.label,
                                "established_label": ent.label,
                            },
                        )

                    matched_ent.add(ent_idx)
                    matched_det.add(det_idx)

        # 4. Assemble association outputs for all entities
        results: List[AssociationMatch] = []
        for i, ent in enumerate(entities):
            if i in entity_match_map:
                results.append(entity_match_map[i])
            else:
                results.append(
                    AssociationMatch(
                        entity_id=ent.entity_id,
                        detection=None,
                        is_associated=False,
                        is_ambiguous=False,
                        confidence=0.0,
                        explanation=f"Entity '{ent.entity_id}' not observed in current visit.",
                    )
                )

        # 5. Collect unmatched candidate detections
        unmatched_detections = [
            detections[j] for j in range(n_det) if j not in matched_det
        ]

        return results, unmatched_detections
