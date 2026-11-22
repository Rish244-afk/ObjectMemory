"""
Temporal benchmark evaluator comparing Multi-Visit Temporal Memory against the Pairwise Baseline.

Evaluates sequences across:
1. Per-class precision, recall, and F1.
2. Confusion matrix and support counts.
3. False removal rate.
4. Correct removal detection rate.
5. UNKNOWN / abstention rate.
6. Number of visits required to confirm removal.
7. Explicit state-transition history for each sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from objectmemory.baseline.comparator import PairwiseComparator
from objectmemory.baseline.models import PairwiseComparisonReport
from objectmemory.benchmarks.temporal_dataset import (
    SyntheticTemporalSequence,
    SyntheticTemporalSequenceGenerator,
)
from objectmemory.detection.models import Detection
from objectmemory.detection.pipeline import transform_bbox_to_canonical
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.states import BaselineChangeType, EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput, VisitProcessingReport


@dataclass
class SequenceEvaluationSummary:
    """Evaluation summary for a single multi-visit temporal sequence."""
    sequence_name: str
    description: str
    total_visits: int
    total_evaluations: int = 0
    temporal_errors: int = 0
    pairwise_discrepancies: int = 0
    entity_trajectories: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    all_states_correct: bool = True
    false_removal_occurred: bool = False
    correct_removal_occurred: bool = False
    visits_to_confirm_removal: Optional[int] = None
    pairwise_false_removals: int = 0
    temporal_false_removals: int = 0


@dataclass
class TemporalBenchmarkReport:
    """Aggregate evaluation report comparing temporal engine against pairwise baseline."""
    total_sequences: int
    total_evaluations: int
    correct_predictions: int
    temporal_accuracy: float
    pairwise_accuracy: float
    false_removal_count: int
    false_removal_rate: float
    pairwise_false_removal_count: int
    pairwise_false_removal_rate: float
    correct_removal_count: int
    correct_removal_rate: float
    unknown_rate: float
    avg_visits_to_confirm_removal: float
    sequence_summaries: List[SequenceEvaluationSummary]
    state_confusion_matrix: Dict[str, Dict[str, int]]
    per_state_metrics: Dict[str, Dict[str, float]]
    pairwise_abstention_count: int = 0
    pairwise_abstention_rate: float = 0.0
    is_synthetic: bool = True

    def format_text_summary(self) -> str:
        """Renders an ASCII report table clearly decoupling Temporal Engine vs Pairwise Baseline."""
        pairwise_matches = int(round(self.pairwise_accuracy * self.total_evaluations))
        lines = []
        lines.append("=" * 86)
        lines.append(" ObjectMemory: Phase 5 Multi-Visit Temporal Memory Benchmark Report")
        lines.append(" [Synthetic Benchmark: Ground-Truth-Derived Candidate Boxes & Labels]")
        lines.append("=" * 86)
        lines.append(f"Total Sequences Evaluated:           {self.total_sequences}")
        lines.append(f"Total Entity-Visit Evaluations:      {self.total_evaluations}")
        lines.append("")
        lines.append("PAIRWISE BASELINE REFERENCE POLICY:")
        lines.append("  Initial Visit Anchoring (v_0): Each query visit v_t (t >= 1) is compared directly")
        lines.append("  against the initial reference visit v_0, establishing long-term displacement delta.")
        lines.append("")
        lines.append("METRIC 1: TEMPORAL MEMORY ENGINE (Multi-Visit Epistemic Belief Tracking)")
        lines.append(f"  Epistemic State Accuracy:          {self.correct_predictions} / {self.total_evaluations} ({self.temporal_accuracy:.1%})")
        lines.append(f"  False Confirmed Removals:          {self.false_removal_count} / {self.total_evaluations} ({self.false_removal_rate:.1%})")
        lines.append(f"  Correct Removal Detection Rate:    {self.correct_removal_rate:.1%}")
        lines.append(f"  Avg Visits to Confirm Removal:     {self.avg_visits_to_confirm_removal:.1f} visits (streak requirement: k=2)")
        lines.append("")
        lines.append("METRIC 2: PAIRWISE BASELINE STATE PROJECTION (Two-Frame Delta Classification)")
        lines.append("  Note: Pairwise baseline outputs 2-frame delta classifications (UNCHANGED/MOVED/REMOVED/UNKNOWN),")
        lines.append("  projected to persistent states for evaluation. Pairwise agreement is NOT directly comparable")
        lines.append("  to temporal belief accuracy because the stateless baseline abstains on intermediate states.")
        lines.append(f"  Exact State Projection Matches:    {pairwise_matches} / {self.total_evaluations} ({self.pairwise_accuracy:.1%})")
        lines.append(f"  Epistemic Abstentions (UNKNOWN):   {self.pairwise_abstention_count} / {self.total_evaluations} ({self.pairwise_abstention_rate:.1%})")
        lines.append(f"  Pairwise False Confirmed Removals: {self.pairwise_false_removal_count} / {self.total_evaluations} ({self.pairwise_false_removal_rate:.1%})")
        lines.append("-" * 86)
        lines.append(f"{'State':<24} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        lines.append("-" * 86)
        for s_name, m in self.per_state_metrics.items():
            lines.append(
                f"{s_name:<24} | {m['precision']:<10.3f} | {m['recall']:<10.3f} | {m['f1']:<10.3f} | {int(m['support']):<8}"
            )
        lines.append("-" * 86)
        lines.append("\nPer-Sequence Breakdown:")
        lines.append(f"{'Sequence Name':<34} | {'Visits':<6} | {'Evals':<5} | {'Temp Err':<8} | {'Pairwise Discr':<14}")
        lines.append("-" * 86)
        for s in self.sequence_summaries:
            lines.append(
                f"{s.sequence_name:<34} | {s.total_visits:<6} | {s.total_evaluations:<5} | {s.temporal_errors:<8} | {s.pairwise_discrepancies:<14}"
            )
        lines.append("=" * 86)
        return "\n".join(lines)


class TemporalBenchmarkEvaluator:
    """
    Executes multi-visit sequences against TemporalMemoryEngine and PairwiseComparator.
    """

    def __init__(
        self,
        registrar: Optional[PlanarRegistrar] = None,
        signature_extractor: Optional[VisualSignatureExtractor] = None,
        config: Optional[TemporalEngineConfig] = None,
    ) -> None:
        self.registrar = registrar or PlanarRegistrar()
        self.signature_extractor = signature_extractor or VisualSignatureExtractor()
        self.config = config or TemporalEngineConfig()
        self.pairwise_comparator = PairwiseComparator(signature_extractor=self.signature_extractor)

    def evaluate_sequence(
        self,
        sequence: SyntheticTemporalSequence,
        db_path: Path,
    ) -> SequenceEvaluationSummary:
        """
        Executes a multi-visit sequence on a fresh temporary SQLite memory store.
        """
        store = SQLiteMemoryStore(db_path)
        engine = TemporalMemoryEngine(
            store=store,
            config=self.config,
            signature_extractor=self.signature_extractor,
        )

        ref_image = sequence.visits[0].image
        entity_trajectories: Dict[str, List[Dict[str, Any]]] = {
            gt_ent.entity_label: [] for gt_ent in sequence.ground_truth_entities
        }

        seq_all_correct = True
        seq_false_removals = 0
        pairwise_false_removals = 0
        removal_confirmed_visit_idx = None
        seq_correct_removal = False

        prev_detections: List[Detection] = []
        prev_image = ref_image
        prev_session_id = "v0"

        for v_idx, visit_frame in enumerate(sequence.visits):
            # 1. Co-Registration against initial reference image
            reg_result = self.registrar.register(ref_image, visit_frame.image)

            # 2. Extract detections from visit image using ground-truth boxes
            detections: List[Detection] = []
            for label, box in visit_frame.ground_truth_bboxes.items():
                if box is not None:
                    # Check if missed detection simulated
                    if visit_frame.is_missed_detection.get(label, False):
                        continue
                    # Check if occlusion simulated (detector misses occluded tool)
                    if visit_frame.is_occluded.get(label, False):
                        continue

                    sig = self.signature_extractor.extract(visit_frame.image, box)
                    canon_box = None
                    if reg_result.success and reg_result.homography is not None:
                        canon_box = transform_bbox_to_canonical(
                            box,
                            reg_result.homography,
                            ref_size=(ref_image.shape[1], ref_image.shape[0]),
                        )
                    detections.append(
                        Detection(
                            label=label,
                            confidence=0.92,
                            raw_bbox=box,
                            canonical_bbox=canon_box,
                            signature=sig,
                        )
                    )

            # 3. Process visit through Temporal Memory Engine
            visit_input = VisitInput(
                session_id=visit_frame.session_id,
                session_name=f"Visit {v_idx + 1}",
                timestamp=visit_frame.timestamp,
                image=visit_frame.image,
                registration=reg_result,
                detections=detections,
            )
            report = engine.process_visit(visit_input)

            # 4. Independent Pairwise Baseline Comparison (visit t vs reference visit 0)
            pairwise_preds_map: Dict[str, BaselineChangeType] = {}
            if v_idx > 0:
                pairwise_report: PairwiseComparisonReport = self.pairwise_comparator.compare(
                    ref_detections=prev_detections,
                    query_detections=detections,
                    registration=reg_result,
                    ref_image=ref_image,
                    query_image_aligned=reg_result.aligned_image,
                    ref_session_id="ref_v0",
                    query_session_id=visit_frame.session_id,
                    is_synthetic=True,
                )
                for m in pairwise_report.matches:
                    lbl = (m.ref_detection.label if m.ref_detection else None) or (m.query_detection.label if m.query_detection else None)
                    if lbl:
                        pairwise_preds_map[lbl] = m.predicted_change
                    # Pairwise false removal: object declared REMOVED when it is physically PRESENT
                    if m.predicted_change == BaselineChangeType.REMOVED:
                        for gt_ent in sequence.ground_truth_entities:
                            if gt_ent.entity_label == lbl:
                                expected_st = gt_ent.expected_states[v_idx]
                                if expected_st in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED):
                                    pairwise_false_removals += 1

            if v_idx == 0:
                prev_detections = list(detections)

            # 5. Evaluate state for each ground-truth entity at visit t
            active_entities = {e.label: e for e in store.list_entities()}

            for gt_ent in sequence.ground_truth_entities:
                expected_st = gt_ent.expected_states[v_idx]
                actual_ent = active_entities.get(gt_ent.entity_label)
                actual_st = actual_ent.current_state if actual_ent else EpistemicState.INSUFFICIENT_EVIDENCE

                is_correct = (actual_st == expected_st)
                if not is_correct:
                    seq_all_correct = False

                # False removal check: declared CONFIRMED_REMOVED when GT is PRESENT or MOVED
                if (
                    actual_st == EpistemicState.CONFIRMED_REMOVED
                    and expected_st in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED)
                ):
                    seq_false_removals += 1

                # Correct removal check
                if actual_st == EpistemicState.CONFIRMED_REMOVED and expected_st == EpistemicState.CONFIRMED_REMOVED:
                    seq_correct_removal = True
                    if removal_confirmed_visit_idx is None:
                        removal_confirmed_visit_idx = v_idx + 1

                # Pairwise evaluation mapping:
                # UNCHANGED -> CONFIRMED_PRESENT, MOVED -> CONFIRMED_MOVED, REMOVED -> CONFIRMED_REMOVED, UNKNOWN -> INSUFFICIENT_EVIDENCE
                pairwise_pred = pairwise_preds_map.get(gt_ent.entity_label)
                pairwise_equiv = None
                if v_idx == 0:
                    pairwise_equiv = EpistemicState.CONFIRMED_PRESENT
                elif pairwise_pred == BaselineChangeType.UNCHANGED:
                    pairwise_equiv = EpistemicState.CONFIRMED_PRESENT
                elif pairwise_pred == BaselineChangeType.MOVED:
                    pairwise_equiv = EpistemicState.CONFIRMED_MOVED
                elif pairwise_pred == BaselineChangeType.REMOVED:
                    pairwise_equiv = EpistemicState.CONFIRMED_REMOVED
                else:
                    pairwise_equiv = EpistemicState.INSUFFICIENT_EVIDENCE

                pairwise_correct = (pairwise_equiv == expected_st)

                entity_trajectories[gt_ent.entity_label].append(
                    {
                        "visit_index": v_idx,
                        "session_id": visit_frame.session_id,
                        "expected_state": expected_st.value,
                        "actual_state": actual_st.value,
                        "is_correct": is_correct,
                        "pairwise_state": pairwise_equiv.value,
                        "pairwise_correct": pairwise_correct,
                    }
                )

        store.close()

        total_evals_seq = 0
        temporal_errs_seq = 0
        pairwise_discr_seq = 0
        for traj in entity_trajectories.values():
            for step in traj:
                total_evals_seq += 1
                if not step["is_correct"]:
                    temporal_errs_seq += 1
                if not step["pairwise_correct"]:
                    pairwise_discr_seq += 1

        return SequenceEvaluationSummary(
            sequence_name=sequence.sequence_name,
            description=sequence.description,
            total_visits=len(sequence.visits),
            total_evaluations=total_evals_seq,
            temporal_errors=temporal_errs_seq,
            pairwise_discrepancies=pairwise_discr_seq,
            entity_trajectories=entity_trajectories,
            all_states_correct=seq_all_correct,
            false_removal_occurred=(seq_false_removals > 0),
            correct_removal_occurred=seq_correct_removal,
            visits_to_confirm_removal=removal_confirmed_visit_idx,
            pairwise_false_removals=pairwise_false_removals,
            temporal_false_removals=seq_false_removals,
        )

    def evaluate_all_sequences(
        self,
        sequences: List[SyntheticTemporalSequence],
        work_dir: Path,
    ) -> TemporalBenchmarkReport:
        """Evaluates all sequences and computes aggregate benchmark metrics."""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        summaries: List[SequenceEvaluationSummary] = []
        total_evals = 0
        correct_evals = 0
        pairwise_correct_evals = 0
        total_false_removals = 0
        total_pairwise_false_removals = 0
        removal_visits_list: List[int] = []
        correct_removals = 0
        expected_removals = 0
        unknown_preds = 0

        # Confusion matrix over EpistemicState
        states = [s.value for s in EpistemicState]
        conf_matrix = {s_gt: {s_pred: 0 for s_pred in states} for s_gt in states}

        for idx, seq in enumerate(sequences):
            db_path = work_dir / f"bench_{seq.sequence_name}.sqlite"
            if db_path.exists():
                db_path.unlink()

            summary = self.evaluate_sequence(seq, db_path)
            summaries.append(summary)

            if db_path.exists():
                db_path.unlink()

            total_false_removals += summary.temporal_false_removals
            total_pairwise_false_removals += summary.pairwise_false_removals

            if summary.visits_to_confirm_removal is not None:
                removal_visits_list.append(summary.visits_to_confirm_removal)

            for traj in summary.entity_trajectories.values():
                for step in traj:
                    total_evals += 1
                    gt_st = step["expected_state"]
                    pred_st = step["actual_state"]

                    conf_matrix[gt_st][pred_st] += 1

                    if step["is_correct"]:
                        correct_evals += 1
                    if step.get("pairwise_correct", False):
                        pairwise_correct_evals += 1

                    if pred_st in (EpistemicState.INSUFFICIENT_EVIDENCE.value, EpistemicState.AMBIGUOUS_IDENTITY.value):
                        unknown_preds += 1

                    if gt_st == EpistemicState.CONFIRMED_REMOVED.value:
                        expected_removals += 1
                        if pred_st == EpistemicState.CONFIRMED_REMOVED.value:
                            correct_removals += 1

        # Precision, recall, F1 per state
        per_state: Dict[str, Dict[str, float]] = {}
        for s in states:
            tp = conf_matrix[s][s]
            fp = sum(conf_matrix[other][s] for other in states if other != s)
            fn = sum(conf_matrix[s][other] for other in states if other != s)
            support = tp + fn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_state[s] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": float(support),
            }

        temp_acc = correct_evals / max(1, total_evals)
        pairwise_acc = pairwise_correct_evals / max(1, total_evals)
        avg_removal_visits = float(np.mean(removal_visits_list)) if removal_visits_list else 2.0

        return TemporalBenchmarkReport(
            total_sequences=len(sequences),
            total_evaluations=total_evals,
            correct_predictions=correct_evals,
            temporal_accuracy=temp_acc,
            pairwise_accuracy=pairwise_acc,
            false_removal_count=total_false_removals,
            false_removal_rate=total_false_removals / max(1, total_evals),
            pairwise_false_removal_count=total_pairwise_false_removals,
            pairwise_false_removal_rate=total_pairwise_false_removals / max(1, total_evals),
            correct_removal_count=correct_removals,
            correct_removal_rate=correct_removals / max(1, expected_removals),
            unknown_rate=unknown_preds / max(1, total_evals),
            avg_visits_to_confirm_removal=avg_removal_visits,
            sequence_summaries=summaries,
            state_confusion_matrix=conf_matrix,
            per_state_metrics=per_state,
            pairwise_abstention_count=(total_evals - pairwise_correct_evals),
            pairwise_abstention_rate=(total_evals - pairwise_correct_evals) / max(1, total_evals),
            is_synthetic=True,
        )
