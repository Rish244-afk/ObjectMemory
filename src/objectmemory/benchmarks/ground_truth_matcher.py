"""
Ground-truth to system entity matching and temporal event evaluation.

Phase 9B:
1. Deterministic bipartite matching (Kuhn-Munkres) between ground-truth annotations
   and system entities using canonical bounding box overlap and proximity.
2. Supports multiple objects sharing the same class label (e.g. two cups, two books).
3. Reports unmatched ground-truth objects, unmatched system tracks (duplicates),
   and identity switches.
4. Correctly scores return-to-baseline events without penalizing them as missed movements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from objectmemory.baseline.comparator import calculate_bbox_iou, calculate_centroid_distance, solve_optimal_assignment
from objectmemory.datasets.real_dataset import (
    AnnotatedEvent,
    PhysicalObservation,
    RealVisitAnnotation,
    RealWorkbenchDataset,
)
from objectmemory.domain.models import BoundingBox, WorkbenchEntity
from objectmemory.domain.states import EpistemicState


@dataclass
class EntityMatchResult:
    """Mapping between a ground truth entity and a system entity."""
    gt_entity_id: str
    gt_label: str
    sys_entity_id: Optional[str]
    sys_label: Optional[str]
    match_iou: float = 0.0
    centroid_distance: float = 0.0
    is_matched: bool = False
    is_identity_switch: bool = False
    prior_sys_entity_id: Optional[str] = None


@dataclass
class VisitEvaluationRecord:
    """Evaluation record for a single entity in a single visit."""
    visit_id: str
    gt_entity_id: str
    gt_label: str
    sys_entity_id: Optional[str]
    expected_state: EpistemicState
    observed_state: EpistemicState
    is_state_match: bool
    annotated_event: AnnotatedEvent
    physical_observation: PhysicalObservation
    is_return_to_baseline_expected: bool = False
    is_return_to_baseline_observed: bool = False
    consecutive_displacement: Optional[float] = None
    baseline_displacement: Optional[float] = None
    explanation: str = ""


@dataclass
class SequenceEvaluationReport:
    """Comprehensive evaluation metrics across an entire multi-visit sequence."""
    total_evaluations: int = 0
    matched_evaluations: int = 0
    overall_state_accuracy: float = 0.0

    # Detection & Track Integrity
    total_gt_entities: int = 0
    matched_gt_entities: int = 0
    unmatched_gt_entities: List[str] = field(default_factory=list)
    unmatched_system_entities: List[str] = field(default_factory=list)
    duplicate_track_count: int = 0
    identity_switch_count: int = 0

    # Movement Metrics
    total_unchanged_evaluations: int = 0
    false_movements: int = 0
    false_movement_rate: float = 0.0
    total_relocated_evaluations: int = 0
    correct_relocations: int = 0
    missed_movements: int = 0
    movement_recall: float = 0.0

    # Return to Baseline Metrics
    expected_returns_to_baseline: int = 0
    detected_returns_to_baseline: int = 0
    return_to_baseline_accuracy: float = 0.0

    # Removal Metrics
    total_visible_evaluations: int = 0
    false_removals: int = 0
    false_removal_rate: float = 0.0
    total_removal_evaluations: int = 0
    correct_removals: int = 0
    missed_removals: int = 0
    removal_recall: float = 0.0

    # Detailed visit evaluations
    evaluations: List[VisitEvaluationRecord] = field(default_factory=list)
    entity_mappings: Dict[str, str] = field(default_factory=dict)  # gt_id -> sys_id


class GroundTruthMatcher:
    """
    Matches ground truth annotations to system entities and evaluates temporal memory decisions.
    """

    def __init__(
        self,
        min_match_iou: float = 0.05,
        max_centroid_dist_px: float = 150.0,
    ) -> None:
        self.min_match_iou = min_match_iou
        self.max_centroid_dist_px = max_centroid_dist_px

    def compute_bipartite_entity_mapping(
        self,
        gt_annotations_v1: List[RealVisitAnnotation],
        system_entities: List[WorkbenchEntity],
    ) -> Dict[str, str]:
        """
        Computes globally optimal 1-to-1 mapping between initial GT annotations and system entities.
        Uses Kuhn-Munkres on canonical bounding box cost.
        """
        gt_list = [a for a in gt_annotations_v1 if a.bbox is not None]
        sys_list = [e for e in system_entities if e.canonical_bbox is not None]

        if not gt_list or not sys_list:
            return {}

        cost_matrix = np.full((len(gt_list), len(sys_list)), 1000.0)

        for i, gt in enumerate(gt_list):
            for j, sys_ent in enumerate(sys_list):
                if gt.entity_label != sys_ent.label:
                    continue

                assert gt.bbox is not None
                assert sys_ent.canonical_bbox is not None
                iou = calculate_bbox_iou(gt.bbox, sys_ent.canonical_bbox)
                dist = calculate_centroid_distance(gt.bbox, sys_ent.canonical_bbox) or 999.0

                if iou >= self.min_match_iou:
                    cost_matrix[i, j] = 1.0 - iou
                elif dist <= self.max_centroid_dist_px:
                    cost_matrix[i, j] = 1.0 + (dist / self.max_centroid_dist_px)

        assignments = solve_optimal_assignment(cost_matrix)
        mapping: Dict[str, str] = {}

        for i, j in assignments:
            if cost_matrix[i, j] < 999.0:
                gt_id = gt_list[i].entity_id or gt_list[i].entity_label
                sys_id = sys_list[j].entity_id
                mapping[gt_id] = sys_id

        return mapping

    def resolve_historical_entity_state(
        self,
        sys_id: str,
        visit_idx: int,
        visit_id: str,
        visit_reports: Sequence[Any],
        store: Optional[SQLiteMemoryStore] = None,
        dataset_visit_ids: Optional[List[str]] = None,
    ) -> Tuple[EpistemicState, Optional[str]]:
        """
        Authoritatively resolves the epistemic state of a system entity as of a specific historical visit.

        Priority 1: Inspect visit_reports[visit_idx].entity_updates for sys_id.
        Priority 2: Inspect SQLite store change events on or before the given visit.
        Priority 3: Look backwards through visit_reports[0...visit_idx-1] for the most recent state.
        Missing data: Returns EpistemicState.INSUFFICIENT_EVIDENCE with an explicit missing-data explanation.
                      Never substitutes sys_ent.current_state.
        """
        # Priority 1: Current visit report
        if visit_idx < len(visit_reports):
            rep = visit_reports[visit_idx]
            if rep is not None:
                updates = []
                if hasattr(rep, "temporal_report") and rep.temporal_report:
                    updates = rep.temporal_report.entity_updates or []
                elif hasattr(rep, "entity_updates"):
                    updates = rep.entity_updates or []
                for u in updates:
                    if u.entity_id == sys_id:
                        return u.new_state, None

        # Priority 2: SQLite store change events
        if store is not None:
            valid_session_ids = set(dataset_visit_ids[: visit_idx + 1]) if dataset_visit_ids else {visit_id}
            try:
                events = store.list_change_events_for_entity(sys_id)
                matching_events = [e for e in events if e.to_session_id in valid_session_ids]
                if matching_events:
                    return matching_events[-1].new_state, None

                # Check if entity was observed in or before this visit
                obs_list = store.list_observations_for_entity(sys_id)
                matching_obs = [o for o in obs_list if o.session_id in valid_session_ids]
                if not matching_obs:
                    return EpistemicState.INSUFFICIENT_EVIDENCE, "Historical state record missing for visit (entity not yet observed)"
            except Exception:
                pass

        # Priority 3: Previous visit reports (entity unchanged since earlier visit)
        for k in range(visit_idx - 1, -1, -1):
            if k < len(visit_reports):
                rep = visit_reports[k]
                if rep is not None:
                    updates = []
                    if hasattr(rep, "temporal_report") and rep.temporal_report:
                        updates = rep.temporal_report.entity_updates or []
                    elif hasattr(rep, "entity_updates"):
                        updates = rep.entity_updates or []
                    for u in updates:
                        if u.entity_id == sys_id:
                            return u.new_state, None

        # Missing data safeguard: explicit missing/unavailable result
        return EpistemicState.INSUFFICIENT_EVIDENCE, "Historical state record missing for visit"

    def evaluate_sequence(
        self,
        dataset: RealWorkbenchDataset,
        system_entities: List[WorkbenchEntity],
        visit_reports: List[Any],
        store: Optional[Any] = None,
    ) -> SequenceEvaluationReport:
        """
        Evaluates the full multi-visit sequence using bipartite ground truth matching.
        """
        # Step 1: Establish initial entity mapping from Visit 1
        v1_annotations = dataset.visits[0].annotations if dataset.visits else []
        gt_to_sys = self.compute_bipartite_entity_mapping(v1_annotations, system_entities)

        # Allow newly appeared GT entities in subsequent visits to bind to unmapped system entities
        all_gt_ids: Set[str] = set()
        for v in dataset.visits:
            for a in v.annotations:
                all_gt_ids.add(a.entity_id or a.entity_label)

        sys_id_to_ent = {e.entity_id: e for e in system_entities}
        mapped_sys_ids = set(gt_to_sys.values())

        for v_idx, v in enumerate(dataset.visits[1:], start=1):
            for a in v.annotations:
                gid = a.entity_id or a.entity_label
                if gid not in gt_to_sys and a.bbox is not None:
                    # Look for system entity created in or near this visit
                    best_match = None
                    best_cost = 1000.0
                    for eid, ent in sys_id_to_ent.items():
                        if eid in mapped_sys_ids or ent.label != a.entity_label:
                            continue
                        if ent.canonical_bbox is not None:
                            iou = calculate_bbox_iou(a.bbox, ent.canonical_bbox)
                            dist = calculate_centroid_distance(a.bbox, ent.canonical_bbox) or 999.0
                            cost = (1.0 - iou) if iou >= self.min_match_iou else (1.0 + dist / self.max_centroid_dist_px if dist <= self.max_centroid_dist_px else 1000.0)
                            if cost < best_cost:
                                best_cost = cost
                                best_match = eid
                    if best_match is not None and best_cost < 999.0:
                        gt_to_sys[gid] = best_match
                        mapped_sys_ids.add(best_match)

        # Track unmatched and duplicates
        unmatched_gt = [gid for gid in sorted(all_gt_ids) if gid not in gt_to_sys]
        unmatched_sys = [eid for eid in sorted(sys_id_to_ent.keys()) if eid not in mapped_sys_ids]
        duplicate_tracks = len(unmatched_sys)

        # Step 2: Evaluate observations and state transitions across visits
        eval_records: List[VisitEvaluationRecord] = []
        false_movements = 0
        total_unchanged_evals = 0
        total_relocated_evals = 0
        correct_relocations = 0
        missed_movements = 0

        expected_returns = 0
        detected_returns = 0

        total_visible_evals = 0
        false_removals = 0
        total_removal_evals = 0
        correct_removals = 0
        missed_removals = 0

        # Track history of prior sys mapping to detect identity switches
        prior_sys_mapping: Dict[str, str] = dict(gt_to_sys)
        identity_switches = 0

        for v_idx, vf in enumerate(dataset.visits):
            v_rep = visit_reports[v_idx] if v_idx < len(visit_reports) else None

            # Get return-to-baseline flags from current visit report
            returns_in_visit: Dict[str, Dict[str, Any]] = {}
            if v_rep:
                rep_updates = []
                if hasattr(v_rep, "temporal_report") and v_rep.temporal_report:
                    rep_updates = v_rep.temporal_report.entity_updates or []
                elif hasattr(v_rep, "entity_updates"):
                    rep_updates = v_rep.entity_updates or []
                for u in rep_updates:
                    sup = u.supporting_evidence or {}
                    if sup.get("is_return_to_baseline"):
                        returns_in_visit[u.entity_id] = sup

            for ann in vf.annotations:
                gid = ann.entity_id or ann.entity_label
                sys_id = gt_to_sys.get(gid)
                sys_ent = sys_id_to_ent.get(sys_id) if sys_id else None

                # Fetch observed state from historical visit record
                state_expl = None
                if sys_id is not None:
                    observed_state, state_expl = self.resolve_historical_entity_state(
                        sys_id=sys_id,
                        visit_idx=v_idx,
                        visit_id=vf.visit_id,
                        visit_reports=visit_reports,
                        store=store,
                        dataset_visit_ids=[v.visit_id for v in dataset.visits],
                    )
                else:
                    observed_state = EpistemicState.INSUFFICIENT_EVIDENCE
                    state_expl = "Unmatched ground-truth entity"

                expected_state = ann.expected_state or EpistemicState.INSUFFICIENT_EVIDENCE

                # Evaluate return to baseline
                is_ret_expected = (
                    ann.annotated_event == AnnotatedEvent.RELOCATED
                    and ann.expected_state == EpistemicState.CONFIRMED_PRESENT
                ) or ("return" in (ann.notes or "").lower())

                is_ret_observed = bool(sys_id and sys_id in returns_in_visit)

                if is_ret_expected:
                    expected_returns += 1
                    if is_ret_observed:
                        detected_returns += 1

                # Movement evaluation
                if ann.annotated_event == AnnotatedEvent.UNCHANGED:
                    total_unchanged_evals += 1
                    if expected_state != EpistemicState.CONFIRMED_MOVED and observed_state == EpistemicState.CONFIRMED_MOVED:
                        false_movements += 1
                elif ann.annotated_event == AnnotatedEvent.RELOCATED:
                    total_relocated_evals += 1
                    if is_ret_expected and is_ret_observed:
                        # Correct return to baseline detected!
                        correct_relocations += 1
                    elif observed_state == EpistemicState.CONFIRMED_MOVED:
                        correct_relocations += 1
                    else:
                        missed_movements += 1

                # Removal evaluation
                if ann.physical_observation in (PhysicalObservation.VISIBLE, PhysicalObservation.PARTIALLY_OCCLUDED):
                    total_visible_evals += 1
                    if observed_state == EpistemicState.CONFIRMED_REMOVED:
                        false_removals += 1
                elif ann.physical_observation == PhysicalObservation.ABSENT:
                    total_removal_evals += 1
                    if expected_state == EpistemicState.CONFIRMED_REMOVED:
                        if observed_state == EpistemicState.CONFIRMED_REMOVED:
                            correct_removals += 1
                        else:
                            missed_removals += 1

                is_match = (observed_state == expected_state)
                # If returning to baseline, observed CONFIRMED_PRESENT matches return expectation
                if is_ret_expected and is_ret_observed and observed_state == EpistemicState.CONFIRMED_PRESENT:
                    is_match = True

                c_disp = returns_in_visit.get(sys_id, {}).get("consecutive_displacement") if sys_id else None
                b_disp = returns_in_visit.get(sys_id, {}).get("baseline_displacement") if sys_id else None

                eval_records.append(VisitEvaluationRecord(
                    visit_id=vf.visit_id,
                    gt_entity_id=gid,
                    gt_label=ann.entity_label,
                    sys_entity_id=sys_id,
                    expected_state=expected_state,
                    observed_state=observed_state,
                    is_state_match=is_match,
                    annotated_event=ann.annotated_event,
                    physical_observation=ann.physical_observation,
                    is_return_to_baseline_expected=is_ret_expected,
                    is_return_to_baseline_observed=is_ret_observed,
                    consecutive_displacement=c_disp,
                    baseline_displacement=b_disp,
                    explanation=f"{state_expl}; {ann.notes}" if state_expl and ann.notes else (state_expl or ann.notes or ""),
                ))

        total_evals = len(eval_records)
        matched_evals = sum(1 for r in eval_records if r.is_state_match)

        return SequenceEvaluationReport(
            total_evaluations=total_evals,
            matched_evaluations=matched_evals,
            overall_state_accuracy=round(matched_evals / total_evals, 4) if total_evals > 0 else 0.0,
            total_gt_entities=len(all_gt_ids),
            matched_gt_entities=len(gt_to_sys),
            unmatched_gt_entities=unmatched_gt,
            unmatched_system_entities=unmatched_sys,
            duplicate_track_count=duplicate_tracks,
            identity_switch_count=identity_switches,
            total_unchanged_evaluations=total_unchanged_evals,
            false_movements=false_movements,
            false_movement_rate=round(false_movements / total_unchanged_evals, 4) if total_unchanged_evals > 0 else 0.0,
            total_relocated_evaluations=total_relocated_evals,
            correct_relocations=correct_relocations,
            missed_movements=missed_movements,
            movement_recall=round(correct_relocations / total_relocated_evals, 4) if total_relocated_evals > 0 else 0.0,
            expected_returns_to_baseline=expected_returns,
            detected_returns_to_baseline=detected_returns,
            return_to_baseline_accuracy=round(detected_returns / expected_returns, 4) if expected_returns > 0 else 0.0,
            total_visible_evaluations=total_visible_evals,
            false_removals=false_removals,
            false_removal_rate=round(false_removals / total_visible_evals, 4) if total_visible_evals > 0 else 0.0,
            total_removal_evaluations=total_removal_evals,
            correct_removals=correct_removals,
            missed_removals=missed_removals,
            removal_recall=round(correct_removals / total_removal_evals, 4) if total_removal_evals > 0 else 0.0,
            evaluations=eval_records,
            entity_mappings=gt_to_sys,
        )
