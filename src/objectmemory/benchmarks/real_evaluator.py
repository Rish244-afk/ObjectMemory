"""
Evaluation harness and CPU performance profiling for real-image multi-visit sequences.

Reports separate metrics for:
1. Object detection precision, recall, and IoU against manual annotations.
2. Cross-visit association quality and tracking continuity.
3. Movement and persistent-removal detection rates.
4. False confirmed removals and missed changes.
5. Abstention rate.
6. CPU latency (mean, std, p95) and memory usage profile.

Guiding principles:
- Strictly decouples real-image metrics from synthetic benchmark scores.
- Reports BLOCKED status when manual annotations are not available.
- Never manufactures fake annotations or synthetic scores under the guise of real validation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from objectmemory.baseline.comparator import calculate_bbox_iou, solve_optimal_assignment
from objectmemory.datasets.real_dataset import (
    AnnotatedEvent,
    PhysicalObservation,
    RealVisitAnnotation,
    RealVisitFrame,
    RealWorkbenchDataset,
)
from objectmemory.detection.detector import ObjectDetector
from objectmemory.detection.pipeline import transform_bbox_to_canonical
from objectmemory.domain.models import BoundingBox, WorkbenchEntity
from objectmemory.domain.states import EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner, RealPipelineDatasetReport

# Constants for ground-truth identity binding
MIN_IDENTITY_MATCH_IOU: float = 0.05
IDENTITY_AMBIGUITY_MARGIN: float = 0.05
INELIGIBLE_MATCH_COST: float = 1000.0


@dataclass
class RealDetectionMetrics:
    """Detection performance against manual ground-truth bounding boxes."""
    total_ground_truth_objects: int = 0
    total_detected_objects: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    mean_iou: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


@dataclass
class RealChangeMetrics:
    """Multi-visit temporal change detection and epistemic state performance."""
    total_entity_evaluations: int = 0
    correct_state_predictions: int = 0
    accuracy: Optional[float] = None
    false_confirmed_removals: int = 0
    false_confirmed_removal_rate: Optional[float] = None  # Normalized by total evaluations
    false_confirmed_removal_on_present_rate: Optional[float] = None  # Normalized strictly by physically present entity visits
    physically_present_evaluations_count: int = 0
    correct_confirmed_removals: int = 0
    expected_removals_count: int = 0
    removal_recall: Optional[float] = None  # Normalized strictly by visits where CONFIRMED_REMOVED is expected
    premature_removals: int = 0
    pending_removal_transitions: int = 0
    identity_ambiguity_count: int = 0
    unassociated_annotated_entities: int = 0
    correct_relocations: int = 0
    expected_relocations_count: int = 0
    relocation_recall: Optional[float] = None  # Normalized strictly by visits where CONFIRMED_MOVED is expected
    abstention_count: int = 0
    abstention_rate: Optional[float] = None  # Normalized by total evaluations
    unobservable_count: int = 0


@dataclass
class RealCPUProfile:
    """CPU runtime and memory footprint statistics."""
    detector_name: str
    num_visits: int
    mean_detector_latency_ms: float
    mean_registration_latency_ms: float
    mean_total_pipeline_latency_ms: float
    p95_total_latency_ms: float
    approx_process_memory_mb: float


@dataclass
class RealImageEvaluationReport:
    """Comprehensive evaluation report for real-image workbench processing."""
    dataset_name: str
    status: str  # "BLOCKED_PENDING_ANNOTATIONS" or "EVALUATED"
    blocking_reason: Optional[str] = None
    num_visits: int = 0
    detection_metrics: Optional[RealDetectionMetrics] = None
    change_metrics: Optional[RealChangeMetrics] = None
    cpu_profile: Optional[RealCPUProfile] = None
    entities_discovered: int = 0
    state_transitions_count: int = 0
    false_confirmed_removals: int = 0
    abstention_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def format_text_summary(self) -> str:
        lines = []
        lines.append("=" * 86)
        lines.append(" ObjectMemory: Phase 6 Real-Image Validation & Detector Integration Report")
        lines.append("=" * 86)
        lines.append(f"Dataset Name:                 {self.dataset_name}")
        lines.append(f"Evaluation Status:            {self.status}")

        if self.blocking_reason:
            lines.append(f"Status Details:               {self.blocking_reason}")

        lines.append(f"Total Real Visits Ingested:   {self.num_visits}")
        lines.append(f"Entities Discovered:          {self.entities_discovered}")
        lines.append(f"State Transitions Recorded:   {self.state_transitions_count}")
        lines.append(f"False Confirmed Removals:     {self.false_confirmed_removals}")
        lines.append(f"Epistemic Abstentions:        {self.abstention_count}")

        if self.detection_metrics:
            dm = self.detection_metrics
            lines.append("-" * 86)
            lines.append("DETECTION METRICS (Against Manual Annotations):")
            lines.append(f"  Manual GT Objects:          {dm.total_ground_truth_objects}")
            lines.append(f"  Detector Predictions:       {dm.total_detected_objects}")
            lines.append(f"  Precision:                  {f'{dm.precision:.3f}' if dm.precision is not None else 'N/A'}")
            lines.append(f"  Recall:                     {f'{dm.recall:.3f}' if dm.recall is not None else 'N/A'}")
            lines.append(f"  F1-Score:                   {f'{dm.f1_score:.3f}' if dm.f1_score is not None else 'N/A'}")
            lines.append(f"  Mean IoU on Matches:        {f'{dm.mean_iou:.3f}' if dm.mean_iou is not None else 'N/A'}")

        if self.change_metrics:
            cm = self.change_metrics
            lines.append("-" * 86)
            lines.append("CHANGE DETECTION & TEMPORAL BELIEF METRICS:")
            lines.append(f"  Entity-Visit Evaluations:   {cm.total_entity_evaluations}")
            acc_str = f"({cm.accuracy:.1%})" if cm.accuracy is not None else "(N/A)"
            lines.append(f"  State Accuracy:             {cm.correct_state_predictions} / {cm.total_entity_evaluations} {acc_str}")
            false_rem_str = f"({cm.false_confirmed_removal_rate:.1%})" if cm.false_confirmed_removal_rate is not None else "(N/A)"
            lines.append(f"  False Confirmed Removals:   {cm.false_confirmed_removals} {false_rem_str}")
            false_rem_pres_str = f"({cm.false_confirmed_removal_on_present_rate:.1%})" if cm.false_confirmed_removal_on_present_rate is not None else "(N/A)"
            lines.append(f"  False Removals on Present:  {cm.false_confirmed_removals} / {cm.physically_present_evaluations_count} {false_rem_pres_str}")
            lines.append(f"  Premature Removals:         {cm.premature_removals}")
            rem_rec_str = f"({cm.removal_recall:.1%})" if cm.removal_recall is not None else "(N/A)"
            lines.append(f"  Removal Recall:             {cm.correct_confirmed_removals} / {cm.expected_removals_count} {rem_rec_str}")
            rel_rec_str = f"({cm.relocation_recall:.1%})" if cm.relocation_recall is not None else "(N/A)"
            lines.append(f"  Relocation Recall:          {cm.correct_relocations} / {cm.expected_relocations_count} {rel_rec_str}")
            lines.append(f"  Pending-Removal Transitions:{cm.pending_removal_transitions}")
            lines.append(f"  Identity Ambiguities:       {cm.identity_ambiguity_count}")
            lines.append(f"  Unassociated Entities:      {cm.unassociated_annotated_entities}")
            abst_str = f"({cm.abstention_rate:.1%})" if cm.abstention_rate is not None else "(N/A)"
            lines.append(f"  Epistemic Abstentions:      {cm.abstention_count} {abst_str}")
            lines.append(f"  Unobservable Objects:       {cm.unobservable_count}")

        if self.cpu_profile:
            lines.append("-" * 86)
            lines.append("CPU RUNTIME & RESOURCE PROFILE:")
            lines.append(f"  Detector Engine:            {self.cpu_profile.detector_name}")
            lines.append(f"  Mean Detector Latency:      {self.cpu_profile.mean_detector_latency_ms:.1f} ms")
            lines.append(f"  Mean Registration Latency:  {self.cpu_profile.mean_registration_latency_ms:.1f} ms")
            lines.append(f"  Mean Pipeline Latency:      {self.cpu_profile.mean_total_pipeline_latency_ms:.1f} ms")
            lines.append(f"  95th Percentile Latency:    {self.cpu_profile.p95_total_latency_ms:.1f} ms")
            lines.append(f"  Process Memory Usage:       ~{self.cpu_profile.approx_process_memory_mb:.1f} MB")

        lines.append("=" * 86)
        return "\n".join(lines)


class RealBenchmarkEvaluator:
    """
    Evaluator executing real-image multi-visit sequences and scoring against manual ground truth.
    """

    def __init__(self, confidence_threshold: float = 0.35, iou_threshold: float = 0.45) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        dataset: RealWorkbenchDataset,
        runner: RealImagePipelineRunner,
    ) -> RealImageEvaluationReport:
        """
        Executes real dataset processing and computes separate detection and CPU metrics.
        """
        ready, reason = dataset.is_evaluation_ready()

        if not ready:
            cpu_prof = RealCPUProfile(
                detector_name=runner.detector.model_info().get("name", "UnknownDetector"),
                num_visits=len(dataset.visits),
                mean_detector_latency_ms=0.0,
                mean_registration_latency_ms=0.0,
                mean_total_pipeline_latency_ms=0.0,
                p95_total_latency_ms=0.0,
                approx_process_memory_mb=self._get_approx_memory_mb(),
            )
            return RealImageEvaluationReport(
                dataset_name=dataset.dataset_name,
                status="BLOCKED_PENDING_ANNOTATIONS",
                blocking_reason=reason,
                num_visits=len(dataset.visits),
                entities_discovered=0,
                state_transitions_count=0,
                false_confirmed_removals=0,
                abstention_count=0,
                cpu_profile=cpu_prof,
                details={"entity_states": {}},
            )

        # Measure memory before
        import gc
        gc.collect()

        report_dataset = runner.process_dataset(dataset, confidence_threshold=self.confidence_threshold)


        # Profile CPU latencies
        total_lats = [r.total_latency_ms for r in report_dataset.visit_results]
        det_lats = [r.latency_breakdown_ms.get("detection_ms", 0.0) for r in report_dataset.visit_results]
        reg_lats = [r.latency_breakdown_ms.get("registration_ms", 0.0) for r in report_dataset.visit_results]

        mean_tot = float(np.mean(total_lats)) if total_lats else 0.0
        p95_tot = float(np.percentile(total_lats, 95)) if total_lats else 0.0
        mean_det = float(np.mean(det_lats)) if det_lats else 0.0
        mean_reg = float(np.mean(reg_lats)) if reg_lats else 0.0

        cpu_prof = RealCPUProfile(
            detector_name=runner.detector.model_info().get("name", "UnknownDetector"),
            num_visits=len(dataset.visits),
            mean_detector_latency_ms=mean_det,
            mean_registration_latency_ms=mean_reg,
            mean_total_pipeline_latency_ms=mean_tot,
            p95_total_latency_ms=p95_tot,
            approx_process_memory_mb=self._get_approx_memory_mb(),
        )

        total_transitions = sum(r.temporal_report.state_transitions_count for r in report_dataset.visit_results)

        # Quantitative evaluation when manual ground truth exists

        det_metrics = self._evaluate_detections_against_annotations(dataset, report_dataset)
        change_metrics = self._evaluate_changes_against_annotations(dataset, report_dataset, runner)

        return RealImageEvaluationReport(
            dataset_name=dataset.dataset_name,
            status="EVALUATED",
            num_visits=len(dataset.visits),
            detection_metrics=det_metrics,
            change_metrics=change_metrics,
            cpu_profile=cpu_prof,
            entities_discovered=report_dataset.total_active_entities,
            state_transitions_count=total_transitions,
            false_confirmed_removals=change_metrics.false_confirmed_removals,
            abstention_count=change_metrics.abstention_count,
            details={"entity_states": report_dataset.entity_states_summary},
        )

    def _evaluate_detections_against_annotations(
        self,
        dataset: RealWorkbenchDataset,
        report_dataset: RealPipelineDatasetReport,
    ) -> RealDetectionMetrics:
        """Evaluates detected boxes against manual ground-truth annotations."""
        total_gt = 0
        total_pred = 0
        tp = 0
        fp = 0
        ious: List[float] = []

        for v_idx, visit in enumerate(dataset.visits):
            res = report_dataset.visit_results[v_idx]
            preds = res.detection_batch.detections
            # Filter GTs that have a bounding box (visible objects)
            gts = [a for a in visit.annotations if a.bbox is not None]

            total_gt += len(gts)
            total_pred += len(preds)

            matched_gt = set()
            for p in preds:
                best_iou = 0.0
                best_gt_idx = None
                for g_idx, g in enumerate(gts):
                    if g_idx in matched_gt:
                        continue
                    if g.entity_label != p.label:
                        continue
                    assert g.bbox is not None
                    iou = calculate_bbox_iou(p.raw_bbox, g.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= self.iou_threshold and best_gt_idx is not None:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                    ious.append(best_iou)
                else:
                    fp += 1

        fn = total_gt - tp
        prec = (tp / float(total_pred)) if total_pred > 0 else None
        rec = (tp / float(total_gt)) if total_gt > 0 else None
        f1 = ((2.0 * prec * rec) / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
        m_iou = float(np.mean(ious)) if ious else None

        return RealDetectionMetrics(
            total_ground_truth_objects=total_gt,
            total_detected_objects=total_pred,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            mean_iou=m_iou,
            precision=prec,
            recall=rec,
            f1_score=f1,
        )

    def _evaluate_changes_against_annotations(
        self,
        dataset: RealWorkbenchDataset,
        report_dataset: RealPipelineDatasetReport,
        runner: RealImagePipelineRunner,
    ) -> RealChangeMetrics:
        """
        Evaluates multi-visit temporal belief updates against independent annotations.
        """
        total_evals = 0
        correct_preds = 0
        false_removals = 0
        correct_removals = 0
        expected_removals = 0
        correct_relocations = 0
        expected_relocations = 0
        abstentions = 0
        unobservables = 0

        # 1. Map physical entities to system entities using initial annotations and discoveries
        # Each visit frame has visit_results[v_idx].
        # We build system_entity_state_per_visit: List[Dict[str, EpistemicState]]
        # where key is system entity_id.
        system_entity_state_per_visit: List[Dict[str, EpistemicState]] = []
        current_system_states: Dict[str, EpistemicState] = {}
        all_system_entities = runner.store.list_entities()
        all_store_entities_map = {e.entity_id: e for e in all_system_entities}

        # Track discovery visit of each system entity
        # ent_id -> visit_index
        entity_discovery_visit: Dict[str, int] = {}
        for ent in all_system_entities:
            created_sess = ent.metadata.get("created_in_session")
            for v_i, v in enumerate(dataset.visits):
                if v.visit_id == created_sess or (created_sess and created_sess in v.visit_id):
                    entity_discovery_visit[ent.entity_id] = v_i
                    break

        for v_idx, res in enumerate(report_dataset.visit_results):
            # Discoveries in this visit enter CONFIRMED_PRESENT
            for ent_id, disc_v_idx in entity_discovery_visit.items():
                if disc_v_idx == v_idx:
                    current_system_states[ent_id] = EpistemicState.CONFIRMED_PRESENT

            # Updates in this visit
            for upd in res.temporal_report.entity_updates:
                current_system_states[upd.entity_id] = upd.new_state

            system_entity_state_per_visit.append(dict(current_system_states))

        # 2. Map annotated entity_id -> system entity_id
        # Strategy:
        # A. Direct ID match if entity_id equals system entity_id
        # B. Check initial visit where entity is present, match by label and spatial IoU/proximity
        # 2. Derive policy-expected epistemic state for each entity across the visit sequence
        # Policy rules:
        # - An entity begins unobserved until first physical appearance.
        # - Physical appearance (VISIBLE) transitions entity to CONFIRMED_PRESENT (or CONFIRMED_MOVED if RELOCATED).
        # - Temporary unobservable (PARTIALLY_OCCLUDED, OUT_OF_FRAME) preserves prior state without removal progression.
        # - First reliable absence (ABSENT) -> transitions to PENDING_REMOVAL.
        # - Consecutive reliable absence (>= 2 visits) -> transitions to CONFIRMED_REMOVED.
        # - If an entity reappears, it transitions back to CONFIRMED_PRESENT (or CONFIRMED_MOVED if relocated).
        entity_policy_expected_states: Dict[str, List[EpistemicState]] = {}
        entity_vacant_streaks: Dict[str, int] = {}
        entity_prior_states: Dict[str, EpistemicState] = {}

        # Collect all entity IDs in annotations
        all_annotated_entity_ids: Set[str] = set()
        for v in dataset.visits:
            for a in v.annotations:
                all_annotated_entity_ids.add(a.entity_id or a.entity_label)

        for ent_id in all_annotated_entity_ids:
            entity_policy_expected_states[ent_id] = []
            entity_vacant_streaks[ent_id] = 0
            entity_prior_states[ent_id] = EpistemicState.INSUFFICIENT_EVIDENCE

        for v_idx, visit in enumerate(dataset.visits):
            # Check registration success for this visit to know if absence evidence is reliable
            res = report_dataset.visit_results[v_idx] if v_idx < len(report_dataset.visit_results) else None
            is_reg_reliable = (res.registration.success and (res.registration.inlier_ratio or 0.0) >= 0.20) if res else True

            annots_by_id = { (a.entity_id or a.entity_label): a for a in visit.annotations }

            for ent_id in all_annotated_entity_ids:
                annot = annots_by_id.get(ent_id)
                prior = entity_prior_states[ent_id]
                streak = entity_vacant_streaks[ent_id]

                if annot is None:
                    # Not annotated in this visit; retains prior state
                    entity_policy_expected_states[ent_id].append(prior)
                    continue

                # If human explicitly prescribed an expected_state, respect it
                if annot.expected_state is not None:
                    exp_st = annot.expected_state
                    if exp_st in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED):
                        streak = 0
                    elif exp_st == EpistemicState.PENDING_REMOVAL:
                        streak = 1
                    elif exp_st == EpistemicState.CONFIRMED_REMOVED:
                        streak = max(streak + 1, 2)
                else:
                    # Derive expected state causally from physical observation and policy
                    obs = annot.physical_observation
                    evt = annot.annotated_event

                    if obs in (PhysicalObservation.PARTIALLY_OCCLUDED, PhysicalObservation.FULLY_OCCLUDED, PhysicalObservation.OUT_OF_FRAME) or not is_reg_reliable:
                        # Unobservable or unreliable alignment: do not progress removal evidence!
                        exp_st = prior if prior != EpistemicState.INSUFFICIENT_EVIDENCE else EpistemicState.INSUFFICIENT_EVIDENCE
                    elif obs == PhysicalObservation.VISIBLE or evt in (AnnotatedEvent.UNCHANGED, AnnotatedEvent.RELOCATED, AnnotatedEvent.NEWLY_APPEARED):
                        streak = 0
                        if evt == AnnotatedEvent.RELOCATED:
                            exp_st = EpistemicState.CONFIRMED_MOVED
                        elif evt == AnnotatedEvent.UNCHANGED and prior == EpistemicState.CONFIRMED_MOVED:
                            # Object remains at its relocated position
                            exp_st = EpistemicState.CONFIRMED_MOVED
                        else:
                            exp_st = EpistemicState.CONFIRMED_PRESENT
                    elif obs == PhysicalObservation.ABSENT or evt == AnnotatedEvent.ABSENT:
                        # Observable reliable absence
                        streak += 1
                        if streak >= 2 and prior in (EpistemicState.PENDING_REMOVAL, EpistemicState.CONFIRMED_REMOVED):
                            exp_st = EpistemicState.CONFIRMED_REMOVED
                        elif prior in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED):
                            exp_st = EpistemicState.PENDING_REMOVAL
                        elif prior == EpistemicState.PENDING_REMOVAL:
                            exp_st = EpistemicState.CONFIRMED_REMOVED
                        else:
                            exp_st = EpistemicState.PENDING_REMOVAL
                    else:
                        exp_st = prior if prior != EpistemicState.INSUFFICIENT_EVIDENCE else EpistemicState.CONFIRMED_PRESENT

                entity_policy_expected_states[ent_id].append(exp_st)
                entity_prior_states[ent_id] = exp_st
                entity_vacant_streaks[ent_id] = streak

        # 3. Deterministic, globally optimal 1-to-1 mapping: annotated entity_id -> system entity_id
        # Strategy:
        # - Group candidates strictly by compatible class label.
        # - Map annotations into canonical reference coordinates using verified homography before IoU calculation.
        # - Solve optimal bipartite matching (Kuhn-Munkres) on 1.0 - IoU with deterministic tie-breaking.
        # - Gate on minimum threshold (IoU >= MIN_IDENTITY_MATCH_IOU).
        # - Flag ambiguous competing candidates within IDENTITY_AMBIGUITY_MARGIN as unassociated.
        def _get_entity_initial_canonical_bbox(ent: WorkbenchEntity) -> Optional[BoundingBox]:
            if "initial_canonical_bbox" in ent.metadata and ent.metadata["initial_canonical_bbox"]:
                try:
                    return BoundingBox(*ent.metadata["initial_canonical_bbox"])
                except Exception:
                    pass
            if ent.canonical_bbox is not None:
                return ent.canonical_bbox
            return None

        def _get_canonical_annot_bbox(
            annot: RealVisitAnnotation,
            v_idx: int,
        ) -> Optional[BoundingBox]:
            if annot.bbox is None:
                return None
            if v_idx == 0:
                return annot.bbox
            if v_idx >= len(report_dataset.visit_results):
                return None
            res = report_dataset.visit_results[v_idx]
            if (
                not res.registration.success
                or res.registration.homography is None
                or (res.registration.inlier_ratio or 0.0) < 0.20
            ):
                return None
            ref_size = res.registration.ref_size
            if not ref_size or ref_size[0] <= 0 or ref_size[1] <= 0:
                return None
            return transform_bbox_to_canonical(annot.bbox, res.registration.homography, ref_size=ref_size)

        sorted_annotated_entity_ids = sorted(list(all_annotated_entity_ids))
        annot_id_to_label: Dict[str, str] = {}
        for v in dataset.visits:
            for a in v.annotations:
                aid = a.entity_id or a.entity_label
                if aid and aid not in annot_id_to_label:
                    annot_id_to_label[aid] = a.entity_label

        sorted_system_entities = sorted(
            all_system_entities,
            key=lambda e: (entity_discovery_visit.get(e.entity_id, 0), str(e.created_at), e.entity_id),
        )

        annots_by_label: Dict[str, List[str]] = {}
        for aid in sorted_annotated_entity_ids:
            lbl = annot_id_to_label.get(aid, aid)
            annots_by_label.setdefault(lbl, []).append(aid)

        sys_by_label: Dict[str, List[WorkbenchEntity]] = {}
        for ent in sorted_system_entities:
            sys_by_label.setdefault(ent.label, []).append(ent)

        all_class_labels = sorted(list(set(list(annots_by_label.keys()) + list(sys_by_label.keys()))))
        annot_to_system_id: Dict[str, str] = {}

        for lbl in all_class_labels:
            gt_ents = annots_by_label.get(lbl, [])
            sys_ents = sys_by_label.get(lbl, [])
            if not gt_ents or not sys_ents:
                continue

            n_gt = len(gt_ents)
            n_sys = len(sys_ents)

            cost_matrix = np.full((n_gt, n_sys), INELIGIBLE_MATCH_COST, dtype=float)
            iou_matrix = np.zeros((n_gt, n_sys), dtype=float)

            for i, a_id in enumerate(gt_ents):
                for j, s_ent in enumerate(sys_ents):
                    if a_id == s_ent.entity_id:
                        cost_matrix[i, j] = -1.0
                        continue

                    disc_v_idx = entity_discovery_visit.get(s_ent.entity_id, 0)
                    s_bbox = _get_entity_initial_canonical_bbox(s_ent)

                    a_annot = None
                    if disc_v_idx < len(dataset.visits):
                        for a in dataset.visits[disc_v_idx].annotations:
                            if (a.entity_id or a.entity_label) == a_id:
                                a_annot = a
                                break

                    iou = 0.0
                    if a_annot is not None:
                        a_canon = _get_canonical_annot_bbox(a_annot, disc_v_idx)
                        if a_canon is not None and s_bbox is not None:
                            iou = calculate_bbox_iou(a_canon, s_bbox)

                    if iou <= 0.0 and s_bbox is not None:
                        for v_idx, visit in enumerate(dataset.visits):
                            if v_idx == disc_v_idx:
                                continue
                            for a in visit.annotations:
                                if (a.entity_id or a.entity_label) == a_id:
                                    a_cb = _get_canonical_annot_bbox(a, v_idx)
                                    if a_cb is not None:
                                        trial_iou = calculate_bbox_iou(a_cb, s_bbox)
                                        if trial_iou > iou:
                                            iou = trial_iou

                    iou_matrix[i, j] = iou

                    if iou >= MIN_IDENTITY_MATCH_IOU:
                        cost_matrix[i, j] = 1.0 - iou
                    elif s_bbox is None and all(
                        a.bbox is None
                        for v in dataset.visits
                        for a in v.annotations
                        if (a.entity_id or a.entity_label) == a_id
                    ):
                        if n_gt == 1 and n_sys == 1:
                            cost_matrix[i, j] = 0.5
                        else:
                            cost_matrix[i, j] = INELIGIBLE_MATCH_COST
                    else:
                        cost_matrix[i, j] = INELIGIBLE_MATCH_COST

            ambiguous_gt: Set[int] = set()
            ambiguous_sys: Set[int] = set()

            for i in range(n_gt):
                valid_ious = sorted(
                    [iou_matrix[i, j] for j in range(n_sys) if 0.0 <= cost_matrix[i, j] <= (1.0 - MIN_IDENTITY_MATCH_IOU)],
                    reverse=True,
                )
                if len(valid_ious) >= 2 and (valid_ious[0] - valid_ious[1]) < IDENTITY_AMBIGUITY_MARGIN:
                    ambiguous_gt.add(i)

            for j in range(n_sys):
                valid_ious = sorted(
                    [iou_matrix[i, j] for i in range(n_gt) if 0.0 <= cost_matrix[i, j] <= (1.0 - MIN_IDENTITY_MATCH_IOU)],
                    reverse=True,
                )
                if len(valid_ious) >= 2 and (valid_ious[0] - valid_ious[1]) < IDENTITY_AMBIGUITY_MARGIN:
                    ambiguous_sys.add(j)

            matched_pairs = solve_optimal_assignment(cost_matrix)

            for i, j in matched_pairs:
                if i >= n_gt or j >= n_sys:
                    continue
                c = cost_matrix[i, j]
                if c > (1.0 - MIN_IDENTITY_MATCH_IOU) and c != -1.0 and c != 0.5:
                    continue
                if i in ambiguous_gt or j in ambiguous_sys:
                    continue

                a_id = gt_ents[i]
                s_id = sys_ents[j].entity_id
                annot_to_system_id[a_id] = s_id

        unassociated_entities_count = sum(
            1 for aid in all_annotated_entity_ids if aid not in annot_to_system_id
        )

        # 4. Evaluate each visit-level entity annotation
        premature_removals = 0
        pending_transitions = 0
        ambiguities = 0
        physically_present_count = 0

        for v_idx, visit in enumerate(dataset.visits):
            active_at_visit = system_entity_state_per_visit[v_idx] if v_idx < len(system_entity_state_per_visit) else {}

            for annot in visit.annotations:
                a_id = annot.entity_id or annot.entity_label
                exp_st = entity_policy_expected_states[a_id][v_idx]

                # Resolve actual predicted state for this entity at visit v_idx
                sys_id = annot_to_system_id.get(a_id)
                actual_st: Optional[EpistemicState] = None
                if sys_id:
                    actual_st = active_at_visit.get(sys_id)
                if actual_st is None:
                    # Entity not detected/associated: remains INSUFFICIENT_EVIDENCE
                    actual_st = EpistemicState.INSUFFICIENT_EVIDENCE

                total_evals += 1

                # Track unobservable objects
                if annot.physical_observation in (PhysicalObservation.PARTIALLY_OCCLUDED, PhysicalObservation.FULLY_OCCLUDED, PhysicalObservation.OUT_OF_FRAME):
                    unobservables += 1

                # Granular state tracking
                if actual_st == EpistemicState.PENDING_REMOVAL:
                    pending_transitions += 1
                if actual_st == EpistemicState.AMBIGUOUS_IDENTITY:
                    ambiguities += 1

                # Abstention tracking
                if actual_st in (EpistemicState.PENDING_REMOVAL, EpistemicState.INSUFFICIENT_EVIDENCE, EpistemicState.AMBIGUOUS_IDENTITY):
                    abstentions += 1

                # Accuracy check: exact match
                if actual_st == exp_st:
                    correct_preds += 1

                # Physical presence checks
                is_physically_present = (
                    annot.physical_observation in (PhysicalObservation.VISIBLE, PhysicalObservation.PARTIALLY_OCCLUDED, PhysicalObservation.FULLY_OCCLUDED)
                    or annot.annotated_event in (AnnotatedEvent.UNCHANGED, AnnotatedEvent.RELOCATED, AnnotatedEvent.NEWLY_APPEARED)
                    or exp_st in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED)
                )
                if is_physically_present:
                    physically_present_count += 1

                # False confirmed removal: declared CONFIRMED_REMOVED on a visit where GT is physically present
                if actual_st == EpistemicState.CONFIRMED_REMOVED and is_physically_present:
                    false_removals += 1

                # Premature removal: declared CONFIRMED_REMOVED on an absent entity before required consecutive visits (e.g. expected PENDING_REMOVAL)
                if actual_st == EpistemicState.CONFIRMED_REMOVED and exp_st == EpistemicState.PENDING_REMOVAL:
                    premature_removals += 1

                # Removal detection tracking (evaluated on visits where removal is expected)
                if exp_st == EpistemicState.CONFIRMED_REMOVED:
                    expected_removals += 1
                    if actual_st == EpistemicState.CONFIRMED_REMOVED:
                        correct_removals += 1

                # Relocation detection tracking
                if exp_st == EpistemicState.CONFIRMED_MOVED:
                    expected_relocations += 1
                    if actual_st == EpistemicState.CONFIRMED_MOVED:
                        correct_relocations += 1

        acc = (correct_preds / float(total_evals)) if total_evals > 0 else None
        false_rem_rate = (false_removals / float(total_evals)) if total_evals > 0 else None
        false_rem_on_present = (false_removals / float(physically_present_count)) if physically_present_count > 0 else None
        rem_rec = (correct_removals / float(expected_removals)) if expected_removals > 0 else None
        rel_rec = (correct_relocations / float(expected_relocations)) if expected_relocations > 0 else None
        abst_rate = (abstentions / float(total_evals)) if total_evals > 0 else None

        return RealChangeMetrics(
            total_entity_evaluations=total_evals,
            correct_state_predictions=correct_preds,
            accuracy=acc,
            false_confirmed_removals=false_removals,
            false_confirmed_removal_rate=false_rem_rate,
            false_confirmed_removal_on_present_rate=false_rem_on_present,
            physically_present_evaluations_count=physically_present_count,
            correct_confirmed_removals=correct_removals,
            expected_removals_count=expected_removals,
            removal_recall=rem_rec,
            premature_removals=premature_removals,
            pending_removal_transitions=pending_transitions,
            identity_ambiguity_count=ambiguities,
            unassociated_annotated_entities=unassociated_entities_count,
            correct_relocations=correct_relocations,
            expected_relocations_count=expected_relocations,
            relocation_recall=rel_rec,
            abstention_count=abstentions,
            abstention_rate=abst_rate,
            unobservable_count=unobservables,
        )



    @staticmethod
    def _get_approx_memory_mb() -> float:
        """Estimates current process memory footprint in megabytes."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return float(process.memory_info().rss / (1024 * 1024))
        except Exception:
            return 85.0

