"""
Independent Pairwise Change-Detection Baseline Comparator.

Compares a reference session with a query session without multi-visit temporal memory.
Serves as the experimental control for the ObjectMemory epistemic reasoning engine.

Decision Rules & Epistemic Safeguards:
1. Spatial coordinate comparison relies on canonical reference coordinates estimated via
   verified planar registration. If alignment failed or was rejected by quality gating,
   all spatial predictions are assigned UNKNOWN.
2. Deterministic one-to-one assignment: every reference detection can match at most one
   query detection, and vice versa. Ties are broken deterministically.
3. Ambiguity gating: if multiple candidates have near-identical visual signature similarity
   (difference < ambiguity_margin), identity assignment is flagged as ambiguous (UNKNOWN).
4. Non-detection is not removal: an unmatched reference object is NOT automatically classified
   as REMOVED. It is confirmed as REMOVED only if its canonical patch in the query image is
   clearly visible within the valid overlap mask, unoccluded, and its appearance significantly
   differs from the object (similarity < vacancy_similarity_max). If occlusion, low overlap,
   or insufficient evidence exists, it is classified as UNKNOWN.
5. All confidence scores are strictly bounded in [0.0, 1.0] and separate from detector
   confidence and alignment quality.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from objectmemory.baseline.models import (
    PairwiseComparisonReport,
    PairwiseMatch,
    PairwiseMatchConfig,
)
from objectmemory.detection.models import Detection
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import BaselineChangeType
from objectmemory.registration.models import AlignmentStatus, RegistrationResult


def calculate_bbox_iou(box1: Optional[BoundingBox], box2: Optional[BoundingBox]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes in [0.0, 1.0]."""
    if box1 is None or box2 is None:
        return 0.0

    inter_xmin = max(box1.x_min, box2.x_min)
    inter_ymin = max(box1.y_min, box2.y_min)
    inter_xmax = min(box1.x_max, box2.x_max)
    inter_ymax = min(box1.y_max, box2.y_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
    area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    iou = inter_area / union_area
    return float(max(0.0, min(1.0, iou)))


def calculate_centroid_distance(box1: Optional[BoundingBox], box2: Optional[BoundingBox]) -> Optional[float]:
    """Computes Euclidean distance between the centroids of two bounding boxes in pixels."""
    if box1 is None or box2 is None:
        return None

    c1_x = (box1.x_min + box1.x_max) * 0.5
    c1_y = (box1.y_min + box1.y_max) * 0.5
    c2_x = (box2.x_min + box2.x_max) * 0.5
    c2_y = (box2.y_min + box2.y_max) * 0.5

    dist = math.hypot(c1_x - c2_x, c1_y - c2_y)
    return float(dist)


def check_bbox_overlap_fraction(
    bbox: Optional[BoundingBox],
    overlap_mask: Optional[np.ndarray],
) -> float:
    """
    Computes the fraction of pixels of bbox that lie within the valid overlap mask (in [0.0, 1.0]).
    Safeguard: Returns 0.0 if overlap_mask is not provided or bbox is None.
    """
    if bbox is None or overlap_mask is None:
        return 0.0

    h, w = overlap_mask.shape[:2]
    x0 = max(0, min(w - 1, int(round(bbox.x_min))))
    y0 = max(0, min(h - 1, int(round(bbox.y_min))))
    x1 = max(0, min(w, int(round(bbox.x_max))))
    y1 = max(0, min(h, int(round(bbox.y_max))))

    box_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
    if box_area <= 0.0 or x1 <= x0 or y1 <= y0:
        return 0.0

    patch = overlap_mask[y0:y1, x0:x1]
    valid_pixels = np.count_nonzero(patch > 0)
    fraction = valid_pixels / box_area
    return float(max(0.0, min(1.0, fraction)))


def solve_optimal_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solves minimum-weight bipartite matching in O(N^3) time using the Kuhn-Munkres algorithm.
    Works for rectangular cost matrices without external dependencies.
    """
    cost = np.array(cost_matrix, dtype=float)
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    dim = max(n, m)
    padded = np.zeros((dim, dim), dtype=float)
    padded[:n, :m] = cost
    u = np.zeros(dim, dtype=float)
    v = np.zeros(dim, dtype=float)
    p = np.zeros(dim + 1, dtype=int)
    way = np.zeros(dim + 1, dtype=int)
    for i in range(1, dim + 1):
        p[0] = i
        j0 = 0
        minv = np.full(dim + 1, np.inf, dtype=float)
        used = np.zeros(dim + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, dim + 1):
                if not used[j]:
                    cur = padded[i0 - 1, j - 1] - u[i0 - 1] - v[j - 1]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(dim + 1):
                if used[j]:
                    u[p[j] - 1] += delta
                    v[j - 1] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    ans = [-1] * n
    for j in range(1, dim + 1):
        if p[j] <= n and j <= m:
            ans[p[j] - 1] = j - 1
    return [(i, ans[i]) for i in range(n) if ans[i] >= 0]


class PairwiseComparator:
    """
    Independent pairwise change detector comparing reference and query observations.
    Operates strictly pairwise with zero multi-session temporal state.
    """

    def __init__(
        self,
        config: Optional[PairwiseMatchConfig] = None,
        signature_extractor: Optional[VisualSignatureExtractor] = None,
    ) -> None:
        self.config = config or PairwiseMatchConfig()
        self.signature_extractor = signature_extractor or VisualSignatureExtractor()

    def compare(
        self,
        ref_detections: List[Detection],
        query_detections: List[Detection],
        registration: Optional[RegistrationResult] = None,
        ref_image: Optional[np.ndarray] = None,
        query_image_aligned: Optional[np.ndarray] = None,
        ref_session_id: str = "ref_session",
        query_session_id: str = "query_session",
        is_synthetic: bool = False,
    ) -> PairwiseComparisonReport:
        """
        Executes deterministic one-to-one pairwise change comparison.

        Args:
            ref_detections: List of detections from reference visit.
            query_detections: List of detections from query visit (with canonical_bbox populated).
            registration: Phase 2 RegistrationResult.
            ref_image: Reference visit image array (optional, for diagnostics).
            query_image_aligned: Query image warped to reference frame (for vacancy checks).
            ref_session_id: Reference session identifier.
            query_session_id: Query session identifier.
            is_synthetic: Provenance flag for synthetic benchmarking.

        Returns:
            PairwiseComparisonReport with auditable matches and summary counts.
        """
        # 1. Filter detections by detector confidence threshold
        active_ref = [
            d for d in ref_detections
            if d.confidence is None or d.confidence >= self.config.min_detector_confidence
        ]
        active_query = [
            d for d in query_detections
            if d.confidence is None or d.confidence >= self.config.min_detector_confidence
        ]

        alignment_status = registration.status if registration else AlignmentStatus.INSUFFICIENT_KEYPOINTS
        overlap_mask = registration.overlap_mask if registration else None

        # 2. Alignment Quality Gating
        # If alignment failed or was not verified, spatial coordinates cannot be compared.
        if registration is None or registration.status != AlignmentStatus.SUCCESS:
            return self._handle_unaligned_comparison(
                active_ref,
                active_query,
                alignment_status,
                ref_session_id,
                query_session_id,
                is_synthetic,
            )

        # 3. Compute pairwise affinity matrix and detect ambiguities
        n_ref = len(active_ref)
        n_query = len(active_query)

        affinity_matrix = np.zeros((n_ref, n_query), dtype=np.float64)
        iou_matrix = np.zeros((n_ref, n_query), dtype=np.float64)
        dist_matrix = np.zeros((n_ref, n_query), dtype=np.float64)

        for i, r_det in enumerate(active_ref):
            r_box = r_det.canonical_bbox or r_det.raw_bbox
            r_sig = r_det.signature

            for j, q_det in enumerate(active_query):
                q_box = q_det.canonical_bbox or q_det.raw_bbox
                q_sig = q_det.signature

                # Spatial metrics in canonical reference coordinates
                iou = calculate_bbox_iou(r_box, q_box)
                iou_matrix[i, j] = iou

                dist = calculate_centroid_distance(r_box, q_box)
                dist_matrix[i, j] = dist if dist is not None else 9999.0

                # Appearance signature similarity
                sim = 0.0
                if r_sig is not None and q_sig is not None:
                    sim = r_sig.cosine_similarity(q_sig)
                    sim = max(0.0, min(1.0, sim))

                # Category compatibility check
                label_match = 1.0
                if r_det.label and q_det.label and r_det.label != q_det.label:
                    label_match = 0.0

                if label_match > 0.0 and sim >= self.config.signature_similarity_threshold:
                    affinity_matrix[i, j] = sim
                else:
                    affinity_matrix[i, j] = 0.0

        # Ambiguity detection: check if multiple candidates have near-identical similarities
        ambiguous_ref: Set[int] = set()
        for i in range(n_ref):
            sims = [affinity_matrix[i, j] for j in range(n_query) if affinity_matrix[i, j] > 0.0]
            if len(sims) >= 2:
                sims.sort(reverse=True)
                if (sims[0] - sims[1]) < self.config.ambiguity_margin:
                    ambiguous_ref.add(i)

        ambiguous_query: Set[int] = set()
        for j in range(n_query):
            sims = [affinity_matrix[i, j] for i in range(n_ref) if affinity_matrix[i, j] > 0.0]
            if len(sims) >= 2:
                sims.sort(reverse=True)
                if (sims[0] - sims[1]) < self.config.ambiguity_margin:
                    ambiguous_query.add(j)

        # 4. One-to-one bipartite matching (greedy or optimal)
        matched_r: Set[int] = set()
        matched_q: Set[int] = set()
        matches: List[PairwiseMatch] = []

        if self.config.assignment_strategy == "optimal" and n_ref > 0 and n_query > 0:
            # Maximum-weight bipartite matching via Kuhn-Munkres
            max_aff = float(np.max(affinity_matrix)) if np.max(affinity_matrix) > 0.0 else 1.0
            cost_matrix = max_aff - affinity_matrix
            optimal_pairs = solve_optimal_assignment(cost_matrix)
            candidate_matches = [
                (affinity_matrix[i, j], iou_matrix[i, j], -dist_matrix[i, j], i, j)
                for i, j in optimal_pairs
                if affinity_matrix[i, j] >= self.config.signature_similarity_threshold
            ]
        else:
            # Deterministic greedy matching
            # Sort candidates deterministically: descending affinity, descending IoU, ascending distance, i, j
            candidate_matches = []
            for i in range(n_ref):
                for j in range(n_query):
                    aff = affinity_matrix[i, j]
                    if aff > 0.0:
                        candidate_matches.append((aff, iou_matrix[i, j], -dist_matrix[i, j], i, j))

            candidate_matches.sort(
                key=lambda t: (t[0], t[1], t[2], -t[3], -t[4]),
                reverse=True,
            )

        for aff, iou, neg_dist, i, j in candidate_matches:
            if i in matched_r or j in matched_q:
                continue

            r_det = active_ref[i]
            q_det = active_query[j]
            dist = -neg_dist

            # Ambiguity check
            if i in ambiguous_ref or j in ambiguous_query:
                matches.append(
                    PairwiseMatch(
                        predicted_change=BaselineChangeType.UNKNOWN,
                        ref_detection=r_det,
                        query_detection=q_det,
                        match_confidence=float(aff),
                        spatial_distance=float(dist),
                        spatial_iou=float(iou),
                        signature_similarity=float(aff),
                        explanation=(
                            f"Ambiguous identity match between candidates (difference < "
                            f"{self.config.ambiguity_margin:.2f}); cannot resolve one-to-one."
                        ),
                        details={"ambiguity_flag": True, "affinity": float(aff)},
                    )
                )
                matched_r.add(i)
                matched_q.add(j)
                continue

            # Classify spatial change
            is_unchanged = (
                iou >= self.config.iou_unchanged_threshold
                or dist <= self.config.distance_unchanged_threshold
            )

            if is_unchanged:
                outcome = BaselineChangeType.UNCHANGED
                explanation = (
                    f"Object matched at same location (IoU: {iou:.2f}, dist: {dist:.1f}px, "
                    f"sig_sim: {aff:.2f})."
                )
            else:
                outcome = BaselineChangeType.MOVED
                explanation = (
                    f"Object re-identified at new location (dist: {dist:.1f}px > "
                    f"{self.config.distance_unchanged_threshold:.1f}px, IoU: {iou:.2f})."
                )

            matches.append(
                PairwiseMatch(
                    predicted_change=outcome,
                    ref_detection=r_det,
                    query_detection=q_det,
                    match_confidence=float(aff),
                    spatial_distance=float(dist),
                    spatial_iou=float(iou),
                    signature_similarity=float(aff),
                    explanation=explanation,
                    details={"iou": float(iou), "distance": float(dist), "affinity": float(aff)},
                )
            )
            matched_r.add(i)
            matched_q.add(j)

        # 4b. Collocated unconfirmed pairings (Safeguard against spurious ADDED under lighting/appearance shift):
        # If an unmatched query detection overlaps an unmatched reference detection with matching category
        # at the same canonical position (IoU >= iou_unchanged_threshold or dist <= distance_unchanged_threshold),
        # but visual signature similarity fell below threshold, pair them as an unconfirmed change (UNKNOWN)
        # rather than emitting a false ADDED query detection and an occluded reference detection.
        for j in range(n_query):
            if j in matched_q:
                continue
            q_det = active_query[j]
            q_box = q_det.canonical_bbox or q_det.raw_bbox

            for i in range(n_ref):
                if i in matched_r:
                    continue
                r_det = active_ref[i]
                r_box = r_det.canonical_bbox or r_det.raw_bbox

                # Check same category
                if r_det.label and q_det.label and r_det.label != q_det.label:
                    continue

                iou = calculate_bbox_iou(r_box, q_box)
                dist = calculate_centroid_distance(r_box, q_box)
                is_collocated = (
                    iou >= self.config.iou_unchanged_threshold
                    or (dist is not None and dist <= self.config.distance_unchanged_threshold)
                )

                if is_collocated:
                    sim = 0.0
                    if r_det.signature is not None and q_det.signature is not None:
                        sim = float(r_det.signature.cosine_similarity(q_det.signature))
                    matches.append(
                        PairwiseMatch(
                            predicted_change=BaselineChangeType.UNKNOWN,
                            ref_detection=r_det,
                            query_detection=q_det,
                            match_confidence=float(max(0.0, min(1.0, sim))),
                            spatial_distance=dist,
                            spatial_iou=iou,
                            signature_similarity=sim,
                            explanation=(
                                f"Object detected at canonical location with matching label '{r_det.label}' "
                                f"(IoU: {iou:.2f}), but visual appearance similarity ({sim:.2f}) is below "
                                f"match threshold ({self.config.signature_similarity_threshold:.2f}); "
                                f"possible illumination change, reflection, or replacement."
                            ),
                            details={
                                "collocated_appearance_mismatch": True,
                                "iou": iou,
                                "similarity": sim,
                            },
                        )
                    )
                    matched_r.add(i)
                    matched_q.add(j)
                    break

        # 5. Handle unmatched query detections -> Candidate ADDED
        for j in range(n_query):
            if j in matched_q:
                continue

            q_det = active_query[j]
            q_box = q_det.canonical_bbox or q_det.raw_bbox

            # Verify overlap fraction in aligned reference frame
            overlap_frac = check_bbox_overlap_fraction(q_box, overlap_mask)
            if overlap_frac < self.config.min_overlap_fraction:
                matches.append(
                    PairwiseMatch(
                        predicted_change=BaselineChangeType.UNKNOWN,
                        query_detection=q_det,
                        match_confidence=q_det.confidence,
                        explanation=(
                            f"Unmatched query detection lies outside or on boundary of valid "
                            f"registration overlap (overlap: {overlap_frac:.1%})."
                        ),
                        details={"overlap_fraction": overlap_frac},
                    )
                )
            else:
                det_conf = q_det.confidence if q_det.confidence is not None else 0.8
                matches.append(
                    PairwiseMatch(
                        predicted_change=BaselineChangeType.ADDED,
                        query_detection=q_det,
                        match_confidence=det_conf,
                        explanation=(
                            f"New object detected in query visit with no reference counterpart "
                            f"(det_conf: {det_conf:.2f})."
                        ),
                        details={"detector_confidence": q_det.confidence},
                    )
                )

        # 6. Handle unmatched reference detections -> Evaluate REMOVED vs UNKNOWN
        # Safeguard: Non-detection is NOT proof of removal.
        for i in range(n_ref):
            if i in matched_r:
                continue

            r_det = active_ref[i]
            r_box = r_det.canonical_bbox or r_det.raw_bbox

            removal_match = self._evaluate_unmatched_reference_object(
                r_det=r_det,
                r_box=r_box,
                query_detections=active_query,
                query_image_aligned=query_image_aligned,
                overlap_mask=overlap_mask,
            )
            matches.append(removal_match)

        # 7. Summary counts
        counts = {
            BaselineChangeType.UNCHANGED.value: 0,
            BaselineChangeType.MOVED.value: 0,
            BaselineChangeType.ADDED.value: 0,
            BaselineChangeType.REMOVED.value: 0,
            BaselineChangeType.UNKNOWN.value: 0,
        }
        for m in matches:
            counts[m.predicted_change.value] += 1

        return PairwiseComparisonReport(
            reference_session_id=ref_session_id,
            query_session_id=query_session_id,
            alignment_status=alignment_status,
            matches=matches,
            summary_counts=counts,
            is_synthetic=is_synthetic,
            metadata={"num_ref": n_ref, "num_query": n_query},
        )

    def _evaluate_unmatched_reference_object(
        self,
        r_det: Detection,
        r_box: Optional[BoundingBox],
        query_detections: List[Detection],
        query_image_aligned: Optional[np.ndarray],
        overlap_mask: Optional[np.ndarray],
    ) -> PairwiseMatch:
        """
        Evaluates whether an unobserved reference object is REMOVED or UNKNOWN.
        Safeguards:
        - Out-of-frame -> UNKNOWN
        - Partial occlusion by another query object -> UNKNOWN
        - Missing detection without vacancy confirmation -> UNKNOWN
        - Confirmed vacant query patch -> REMOVED
        """
        # A. Check overlap mask coverage
        overlap_frac = check_bbox_overlap_fraction(r_box, overlap_mask)
        if overlap_frac < self.config.min_overlap_fraction:
            return PairwiseMatch(
                predicted_change=BaselineChangeType.UNKNOWN,
                ref_detection=r_det,
                explanation=(
                    f"Reference object canonical region is outside or partially out-of-frame in "
                    f"query image (overlap fraction: {overlap_frac:.1%} < "
                    f"{self.config.min_overlap_fraction:.1%})."
                ),
                details={"overlap_fraction": overlap_frac, "reason": "OUT_OF_FRAME"},
            )

        # B. Check for occlusion by any detected query object
        for q_det in query_detections:
            q_box = q_det.canonical_bbox or q_det.raw_bbox
            iou = calculate_bbox_iou(r_box, q_box)
            if iou > 0.20:
                return PairwiseMatch(
                    predicted_change=BaselineChangeType.UNKNOWN,
                    ref_detection=r_det,
                    query_detection=q_det,
                    explanation=(
                        f"Reference object canonical region is partially occluded by query detection "
                        f"'{q_det.label}' (IoU: {iou:.2f}); cannot verify removal."
                    ),
                    details={"occlusion_iou": iou, "reason": "PARTIALLY_OCCLUDED"},
                )

        # C. Vacancy verification on aligned query patch
        if (
            self.config.vacancy_check_enabled
            and query_image_aligned is not None
            and r_box is not None
            and r_det.signature is not None
            and r_det.signature.is_valid
        ):
            box_w = r_box.x_max - r_box.x_min
            box_h = r_box.y_max - r_box.y_min
            if box_w >= 5 and box_h >= 5:
                patch_sig = self.signature_extractor.extract(query_image_aligned, r_box)
                if patch_sig.is_valid:
                    patch_sim = float(r_det.signature.cosine_similarity(patch_sig))
                    patch_sim = max(0.0, min(1.0, patch_sim))

                    if patch_sim < self.config.vacancy_similarity_max:
                        # Aligned patch appearance differs significantly from the object -> Vacancy confirmed
                        conf = float(max(0.0, min(1.0, 1.0 - patch_sim)))
                        return PairwiseMatch(
                            predicted_change=BaselineChangeType.REMOVED,
                            ref_detection=r_det,
                            match_confidence=conf,
                            signature_similarity=patch_sim,
                            explanation=(
                                f"Object not detected and canonical patch verifies vacancy (patch "
                                f"similarity: {patch_sim:.2f} < {self.config.vacancy_similarity_max:.2f})."
                            ),
                            details={"patch_similarity": patch_sim, "vacancy_verified": True},
                        )
                    else:
                        # Patch still exhibits object-like visual attributes -> Detector miss or reflection
                        return PairwiseMatch(
                            predicted_change=BaselineChangeType.UNKNOWN,
                            ref_detection=r_det,
                            signature_similarity=patch_sim,
                            explanation=(
                                f"Object not detected by detector, but canonical patch appearance retains "
                                f"similarity ({patch_sim:.2f}); non-detection cannot confirm removal."
                            ),
                            details={"patch_similarity": patch_sim, "vacancy_verified": False},
                        )
                else:
                    return PairwiseMatch(
                        predicted_change=BaselineChangeType.UNKNOWN,
                        ref_detection=r_det,
                        explanation=(
                            f"Object not detected and vacancy verification aborted: "
                            f"{patch_sig.failure_reason or 'invalid patch signature'}."
                        ),
                        details={"patch_signature_invalid": True},
                    )

        # D. Fallback if vacancy check is disabled or image is unavailable
        if not self.config.require_vacancy_evidence:
            return PairwiseMatch(
                predicted_change=BaselineChangeType.REMOVED,
                ref_detection=r_det,
                match_confidence=0.50,
                explanation="Object not detected in query visit (naive detection-only mode).",
                details={"naive_removal": True},
            )

        return PairwiseMatch(
            predicted_change=BaselineChangeType.UNKNOWN,
            ref_detection=r_det,
            explanation=(
                "Object not detected in query visit; non-detection without vacancy verification "
                "cannot confirm removal."
            ),
            details={"reason": "MISSING_DETECTION_UNVERIFIED"},
        )

    def _handle_unaligned_comparison(
        self,
        ref_detections: List[Detection],
        query_detections: List[Detection],
        alignment_status: AlignmentStatus,
        ref_session_id: str,
        query_session_id: str,
        is_synthetic: bool,
    ) -> PairwiseComparisonReport:
        """Produces UNKNOWN predictions when registration is rejected or failed."""
        matches: List[PairwiseMatch] = []
        explanation = (
            f"Image co-registration status is '{alignment_status.value}'; without verified planar "
            f"alignment, spatial change cannot be determined."
        )

        for r in ref_detections:
            matches.append(
                PairwiseMatch(
                    predicted_change=BaselineChangeType.UNKNOWN,
                    ref_detection=r,
                    explanation=explanation,
                    details={"alignment_status": alignment_status.value},
                )
            )

        for q in query_detections:
            matches.append(
                PairwiseMatch(
                    predicted_change=BaselineChangeType.UNKNOWN,
                    query_detection=q,
                    explanation=explanation,
                    details={"alignment_status": alignment_status.value},
                )
            )

        counts = {
            BaselineChangeType.UNCHANGED.value: 0,
            BaselineChangeType.MOVED.value: 0,
            BaselineChangeType.ADDED.value: 0,
            BaselineChangeType.REMOVED.value: 0,
            BaselineChangeType.UNKNOWN.value: len(matches),
        }

        return PairwiseComparisonReport(
            reference_session_id=ref_session_id,
            query_session_id=query_session_id,
            alignment_status=alignment_status,
            matches=matches,
            summary_counts=counts,
            is_synthetic=is_synthetic,
            metadata={"alignment_failure": True},
        )
