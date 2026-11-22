"""
Benchmark evaluation engine for the independent pairwise baseline.

Evaluates PairwiseComparator predictions against synthetic ground truth.
Reports:
- Precision, recall, and F1 per change type.
- Full confusion matrix counts.
- Rate of UNKNOWN predictions (epistemic refusal rate).
- Object-level vs image-level scenario results.
- Per-scenario breakdowns exposing lighting and occlusion failure modes.

Safeguards:
- No manufactured scores.
- Zero temporal memory or multi-visit confirmation is used.
- Distinguishes refusal-to-guess (UNKNOWN) from incorrect predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from objectmemory.baseline.comparator import PairwiseComparator, calculate_bbox_iou
from objectmemory.baseline.models import PairwiseComparisonReport, PairwiseMatch
from objectmemory.benchmarks.synthetic_dataset import (
    GroundTruthObject,
    SyntheticScenarioPair,
)
from objectmemory.detection.models import Detection
from objectmemory.detection.pipeline import transform_bbox_to_canonical
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.states import BaselineChangeType
from objectmemory.registration.models import AlignmentStatus
from objectmemory.registration.registrar import PlanarRegistrar


@dataclass
class ObjectEvaluationRecord:
    """Individual evaluation record for one ground truth physical object."""
    scenario_name: str
    object_id: str
    label: str
    ground_truth: BaselineChangeType
    predicted: BaselineChangeType
    match_confidence: Optional[float]
    is_correct: bool
    is_unknown: bool
    explanation: str = ""
    notes: str = ""


@dataclass
class ScenarioEvaluationSummary:
    """Summary of evaluation for a single scenario."""
    scenario_name: str
    description: str
    total_objects: int
    num_correct: int
    num_incorrect: int
    num_unknown: int
    accuracy: float
    alignment_status: str
    object_records: List[ObjectEvaluationRecord] = field(default_factory=list)


@dataclass
class BenchmarkEvaluationReport:
    """Complete aggregated evaluation report across all benchmark scenarios."""
    total_objects: int
    total_correct: int
    total_incorrect: int
    total_refusals: int
    total_unknown: int
    overall_accuracy: float
    refusal_rate: float
    unknown_rate: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[str, Dict[str, int]]
    scenario_summaries: List[ScenarioEvaluationSummary]
    unassociated_predictions_count: int = 0
    is_synthetic: bool = True

    def format_text_summary(self) -> str:
        """Renders an ASCII report table of measured metrics."""
        lines = []
        lines.append("=" * 78)
        lines.append(" ObjectMemory: Pairwise Baseline Benchmark Report ")
        lines.append("=" * 78)
        lines.append(f"Total Objects Evaluated: {self.total_objects}")
        lines.append(f"Correct Predictions:     {self.total_correct} ({self.overall_accuracy:.1%})")
        lines.append(f"Incorrect Predictions:   {self.total_incorrect} ({self.total_incorrect / max(1, self.total_objects):.1%})")
        lines.append(f"Abstentions / Refusals:  {self.total_refusals} ({self.refusal_rate:.1%})")
        lines.append(f"Total UNKNOWN Outputs:   {self.total_unknown} (Rate: {self.unknown_rate:.1%})")
        if self.unassociated_predictions_count > 0:
            lines.append(f"Unassociated Match Detections: {self.unassociated_predictions_count}")
        lines.append("-" * 78)
        lines.append(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        lines.append("-" * 78)
        for cls_name, m in self.per_class_metrics.items():
            lines.append(
                f"{cls_name:<12} | {m['precision']:<10.3f} | {m['recall']:<10.3f} | {m['f1']:<10.3f} | {int(m['support']):<8}"
            )
        lines.append("-" * 78)
        lines.append("\nPer-Scenario Results:")
        lines.append(f"{'Scenario':<22} | {'Status':<12} | {'Objs':<5} | {'Corr':<5} | {'Inc':<5} | {'Ref':<5} | {'Acc':<6}")
        lines.append("-" * 78)
        for s in self.scenario_summaries:
            lines.append(
                f"{s.scenario_name:<22} | {s.alignment_status:<12} | {s.total_objects:<5} | {s.num_correct:<5} | {s.num_incorrect:<5} | {s.num_unknown:<5} | {s.accuracy:<6.1%}"
            )
        lines.append("=" * 78)
        return "\n".join(lines)


class PairwiseBenchmarkEvaluator:
    """
    Executes registration, detection extraction, and pairwise comparison
    for synthetic benchmark pairs and calculates rigorous classification metrics.
    """

    def __init__(
        self,
        comparator: Optional[PairwiseComparator] = None,
        registrar: Optional[PlanarRegistrar] = None,
        signature_extractor: Optional[VisualSignatureExtractor] = None,
    ) -> None:
        self.signature_extractor = signature_extractor or VisualSignatureExtractor()
        self.comparator = comparator or PairwiseComparator(signature_extractor=self.signature_extractor)
        self.registrar = registrar or PlanarRegistrar()

    def evaluate_pair(self, scenario: SyntheticScenarioPair) -> Tuple[ScenarioEvaluationSummary, int]:
        """
        Evaluates a single scenario pair.
        Returns the scenario summary and the count of unassociated pairwise predictions.
        """
        # 1. Planar Co-Registration
        reg_result = self.registrar.register(scenario.ref_image, scenario.query_image)

        # 2. Extract detections from synthetic images using ground-truth regions
        ref_detections: List[Detection] = []
        for obj in scenario.objects:
            if obj.ref_bbox is not None:
                sig = self.signature_extractor.extract(scenario.ref_image, obj.ref_bbox)
                ref_detections.append(
                    Detection(
                        label=obj.label,
                        confidence=0.92,
                        raw_bbox=obj.ref_bbox,
                        canonical_bbox=obj.ref_bbox,
                        signature=sig,
                    )
                )

        query_detections: List[Detection] = []
        for obj in scenario.objects:
            # If the object is present in query frame
            if obj.query_bbox is not None and obj.ground_truth_change != BaselineChangeType.REMOVED:
                # In occlusion scenario, if object is occluded, simulate detector degradation or omission
                if obj.is_occluded:
                    # In partial occlusion, object is 55% occluded; detector misses the object or has low confidence
                    # We omit it to test whether the baseline handles the detector miss without falsely claiming removal
                    continue

                sig = self.signature_extractor.extract(scenario.query_image, obj.query_bbox)

                # Map to canonical reference frame using estimated homography
                canonical = None
                if reg_result.success and reg_result.homography is not None:
                    canonical = transform_bbox_to_canonical(
                        obj.query_bbox,
                        reg_result.homography,
                        ref_size=(scenario.ref_image.shape[1], scenario.ref_image.shape[0]),
                    )

                query_detections.append(
                    Detection(
                        label=obj.label,
                        confidence=0.88,
                        raw_bbox=obj.query_bbox,
                        canonical_bbox=canonical,
                        signature=sig,
                    )
                )

        # 3. Execute Pairwise Comparison
        report: PairwiseComparisonReport = self.comparator.compare(
            ref_detections=ref_detections,
            query_detections=query_detections,
            registration=reg_result,
            ref_image=scenario.ref_image,
            query_image_aligned=reg_result.aligned_image,
            ref_session_id=f"{scenario.scenario_name}_ref",
            query_session_id=f"{scenario.scenario_name}_query",
            is_synthetic=True,
        )

        # 4. Map pairwise matches to ground truth objects strictly by provenance
        object_records: List[ObjectEvaluationRecord] = []
        associated_match_indices = set()

        for gt_obj in scenario.objects:
            pred_match, match_idx = self._find_matching_prediction(gt_obj, report.matches)
            if pred_match is not None and match_idx is not None:
                associated_match_indices.add(match_idx)
                predicted_change = pred_match.predicted_change
                match_conf = pred_match.match_confidence
                explanation = pred_match.explanation
            else:
                predicted_change = BaselineChangeType.UNKNOWN
                match_conf = None
                explanation = "No matching prediction associated with object."

            is_correct = (predicted_change == gt_obj.ground_truth_change)
            is_unk = (predicted_change == BaselineChangeType.UNKNOWN)

            object_records.append(
                ObjectEvaluationRecord(
                    scenario_name=scenario.scenario_name,
                    object_id=gt_obj.object_id,
                    label=gt_obj.label,
                    ground_truth=gt_obj.ground_truth_change,
                    predicted=predicted_change,
                    match_confidence=match_conf,
                    is_correct=is_correct,
                    is_unknown=is_unk,
                    explanation=explanation,
                    notes=gt_obj.notes,
                )
            )

        unassociated_count = len(report.matches) - len(associated_match_indices)

        num_corr = sum(1 for r in object_records if r.is_correct)
        # Refusal = predicted UNKNOWN when Ground Truth is NOT UNKNOWN (abstention)
        num_ref = sum(1 for r in object_records if r.predicted == BaselineChangeType.UNKNOWN and r.ground_truth != BaselineChangeType.UNKNOWN)
        num_inc = len(object_records) - num_corr - num_ref
        acc = num_corr / max(1, len(object_records))

        summary = ScenarioEvaluationSummary(
            scenario_name=scenario.scenario_name,
            description=scenario.description,
            total_objects=len(object_records),
            num_correct=num_corr,
            num_incorrect=num_inc,
            num_unknown=num_ref,
            accuracy=acc,
            alignment_status=reg_result.status.value,
            object_records=object_records,
        )
        return summary, unassociated_count

    def evaluate_all(self, scenarios: List[SyntheticScenarioPair]) -> BenchmarkEvaluationReport:
        """Evaluates all scenarios and computes aggregate metrics."""
        summaries: List[ScenarioEvaluationSummary] = []
        all_records: List[ObjectEvaluationRecord] = []
        total_unassociated = 0

        for scenario in scenarios:
            summary, unassoc = self.evaluate_pair(scenario)
            summaries.append(summary)
            all_records.extend(summary.object_records)
            total_unassociated += unassoc

        # Build confusion matrix across all 5 BaselineChangeType values
        classes = [c.value for c in BaselineChangeType]
        conf_matrix = {c_gt: {c_pred: 0 for c_pred in classes} for c_gt in classes}

        for rec in all_records:
            gt_val = rec.ground_truth.value
            pred_val = rec.predicted.value
            conf_matrix[gt_val][pred_val] += 1

        # Compute precision, recall, F1 per class
        per_class: Dict[str, Dict[str, float]] = {}
        for c in classes:
            tp = conf_matrix[c][c]
            fp = sum(conf_matrix[other][c] for other in classes if other != c)
            fn = sum(conf_matrix[c][other] for other in classes if other != c)
            support = tp + fn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[c] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": float(support),
            }

        total_obs = len(all_records)
        total_corr = sum(1 for r in all_records if r.is_correct)
        # Explicit distinction:
        # total_refusals = count of objects where baseline abstained (pred=UNKNOWN) when GT was definite (gt != UNKNOWN)
        total_ref = sum(1 for r in all_records if r.predicted == BaselineChangeType.UNKNOWN and r.ground_truth != BaselineChangeType.UNKNOWN)
        total_inc = total_obs - total_corr - total_ref
        overall_acc = total_corr / max(1, total_obs)
        refusal_rate = total_ref / max(1, total_obs)
        # total_unknown = all predictions of UNKNOWN (including correct UNKNOWN)
        total_all_unk = sum(1 for r in all_records if r.predicted == BaselineChangeType.UNKNOWN)
        unk_rate = total_all_unk / max(1, total_obs)

        return BenchmarkEvaluationReport(
            total_objects=total_obs,
            total_correct=total_corr,
            total_incorrect=total_inc,
            total_refusals=total_ref,
            total_unknown=total_all_unk,
            overall_accuracy=overall_acc,
            refusal_rate=refusal_rate,
            unknown_rate=unk_rate,
            per_class_metrics=per_class,
            confusion_matrix=conf_matrix,
            scenario_summaries=summaries,
            unassociated_predictions_count=total_unassociated,
            is_synthetic=True,
        )

    def _find_matching_prediction(
        self,
        gt_obj: GroundTruthObject,
        matches: List[PairwiseMatch],
    ) -> Tuple[Optional[PairwiseMatch], Optional[int]]:
        """
        Associates a ground truth object with the corresponding PairwiseMatch
        strictly respecting reference vs query provenance.
        Returns (best_match, match_index) or (None, None).
        """
        best_match = None
        best_idx = None
        best_overlap = 0.0

        for idx, m in enumerate(matches):
            if gt_obj.ref_bbox is not None:
                # Ground truth object has a reference bbox: must associate via ref_detection
                if m.ref_detection is not None:
                    r_box = m.ref_detection.raw_bbox or m.ref_detection.canonical_bbox
                    if r_box is not None:
                        iou = calculate_bbox_iou(gt_obj.ref_bbox, r_box)
                        if iou > 0.40 and iou > best_overlap:
                            best_overlap = iou
                            best_match = m
                            best_idx = idx
            else:
                # Ground truth object was ADDED (no reference bbox):
                # Must associate strictly with an unmatched query prediction (ref_detection is None)
                if m.ref_detection is None and m.query_detection is not None:
                    q_box = m.query_detection.raw_bbox or m.query_detection.canonical_bbox
                    if q_box is not None and gt_obj.query_bbox is not None:
                        iou = calculate_bbox_iou(gt_obj.query_bbox, q_box)
                        if iou > 0.40 and iou > best_overlap:
                            best_overlap = iou
                            best_match = m
                            best_idx = idx

        return best_match, best_idx
