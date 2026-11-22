"""
Benchmark Evaluation Runner for ObjectMemory (Remediated Phase 3C.2).

Coordinates:
1. Versioned manifest ingestion and validation (Phase 3A.1).
2. Deterministic baseline and ablation configuration resolution (Phase 3B).
3. Complete database and state isolation per configuration run.
4. Immutable per-visit prediction snapshots (zero historical state leakage).
5. Canonical coordinate-frame transformations via 4-corner perspective mapping.
6. Anchored 1-to-1 bipartite ground-truth matching (stable across object relocation).
7. Independent execution of supported configurations:
   - B0: Memoryless Detector Baseline (Precision, Recall, F1; N/A temporal scores)
   - B3: Complete ObjectMemory (Full Proposed System)
   - A5: No Ambiguity Handling Ablation
   - A2 / A3: Partially Supported Ablations (Opt-in only with explicit limitation labels)
8. Scientifically defensible metric computation:
   - False removal rate (False Positive Rate over non-removed visits)
   - False discovery proportion (FP / (TP + FP) for removal predictions)
   - Removal recall (visit-level and event-level)
   - Confirmation rate and conditional confirmation delay
   - Identity consistency (penalizing missed observations)
   - Duplicate identities and per-entity track fragmentation
   - Unsupported movement rate
   - Historical retrieval accuracy
9. Machine-readable JSON/CSV output and human-readable provenance-separated reporting.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

from objectmemory.baseline.comparator import (
    calculate_bbox_iou,
    calculate_centroid_distance,
    solve_optimal_assignment,
)
from objectmemory.benchmarks.config_factory import (
    BenchmarkConfig,
    BenchmarkConfigFactory,
    ConfigurationSupportStatus,
    UnsupportedConfigurationError,
)
from objectmemory.benchmarks.manifest import (
    CANONICAL_STATE_VOCABULARY,
    DatasetManifest,
    DatasetProvenance,
    ManifestLoader,
    ManifestValidationError,
    ObjectAnnotationEntry,
    ObservationEntry,
    SequenceEntry,
)
from objectmemory.detection.detector import MockObjectDetector, ObjectDetector
from objectmemory.detection.models import BoundingBox, Detection, DetectionBatch
from objectmemory.detection.pipeline import is_bbox_boundary_clipped, transform_bbox_to_canonical
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import WorkbenchEntity
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import (
    TemporalEngineConfig,
    VisitInput,
    VisitProcessingReport,
)

logger = logging.getLogger(__name__)


def transform_query_bbox_to_reference(
    bbox: Optional[BoundingBox],
    homography: Optional[np.ndarray],
    ref_size: Tuple[int, int],
) -> Optional[BoundingBox]:
    """
    Transforms a 2D bounding box from query image coordinates to reference keyframe
    canonical coordinates via homography H using 4-corner perspective transformation.

    Safeguards:
    1. Returns None if bbox is None, homography is None, or homography contains non-finite values.
    2. Transforms all 4 corners using homogeneous coordinates.
    3. Verifies projective transformation produces finite coordinates.
    4. Constructs enclosing axis-aligned bounding box and clamps to ref_size canvas.
    5. Discards degenerate or out-of-canvas boxes (width <= 0 or height <= 0).
    """
    if bbox is None or homography is None:
        return None
    if not np.all(np.isfinite(homography)):
        return None

    return transform_bbox_to_canonical(bbox, homography, ref_size)


@dataclass(frozen=True)
class EntityVisitSnapshot:
    """
    Immutable snapshot of a system entity's exact belief state immediately after a visit.
    Guarantees strict temporal isolation and eliminates historical state leakage.
    """
    entity_id: str
    label: str
    epistemic_state: EpistemicState
    visibility_status: VisibilityStatus
    canonical_bbox: Optional[BoundingBox]
    raw_bbox: Optional[BoundingBox]
    is_associated: bool
    is_newly_created: bool
    registration_reliable: bool
    confidence: float
    vacant_streak: int
    unobserved_streak: int
    explanation: str
    visit_id: str
    visit_index: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisitPredictionSnapshot:
    """
    Complete state snapshot for a single visit immediately following processing.
    """
    visit_index: int
    visit_id: str
    timestamp: str
    registration: RegistrationResult
    registration_reliable: bool
    entities: Dict[str, EntityVisitSnapshot]  # entity_id -> snapshot
    raw_detections: List[Detection] = field(default_factory=list)
    unassociated_detections: List[Detection] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    """
    Formally defined, scientifically audited benchmark metrics for ObjectMemory.
    Every metric tracks exact numerators, denominators, and explicit zero-denominator behavior.
    """
    total_observations_processed: int = 0
    total_entity_evaluations: int = 0

    # 1. False Removal Rate (FRR)
    # Formulated as False Positive Rate: FP / (FP + TN) over eligible non-removed visits
    false_removal_rate: Optional[float] = None
    false_removal_count: int = 0
    false_removal_denominator: int = 0

    # 2. False Discovery Proportion (FDP)
    # Formulated as False Discovery Rate: FP / (TP + FP) over confirmed removal predictions
    false_discovery_proportion: Optional[float] = None
    false_discovery_count: int = 0
    false_discovery_denominator: int = 0

    # 3. Removal Recall (RR)
    # Formulated as True Positive Rate: TP / (TP + FN) over ground-truth removal visits
    removal_recall: Optional[float] = None
    removal_recall_count: int = 0
    removal_recall_denominator: int = 0

    # Event-Level Removal Recall
    event_removal_recall: Optional[float] = None
    event_removal_recall_count: int = 0
    event_removal_recall_denominator: int = 0

    # 4. Confirmation Rate (CR)
    # Confirmed eligible change events / Total eligible ground-truth change events
    confirmation_rate: Optional[float] = None
    confirmed_change_events_count: int = 0
    total_eligible_change_events: int = 0
    unconfirmed_change_events_count: int = 0

    # 5. Confirmation Delay (CD)
    # Conditional mean delay in visits among successfully confirmed change events
    mean_confirmation_delay_visits: Optional[float] = None
    confirmation_delay_events_count: int = 0

    # 6. Unsupported Movement Rate (UMR)
    # Fraction of stationary entity visits falsely predicted as CONFIRMED_MOVED
    unsupported_movement_rate: Optional[float] = None
    unsupported_movement_count: int = 0
    unsupported_movement_denominator: int = 0

    # 7. Identity Consistency (IC)
    # Dominant-track matched observations divided by total present ground-truth visits
    # Strictly penalizes missed observations / false negatives
    identity_consistency: Optional[float] = None
    identity_consistency_numerator: int = 0
    identity_consistency_denominator: int = 0

    # 8. ID Switches (IDS)
    # Count of times assigned system ID changes across consecutive observed visits
    id_switches: int = 0
    id_switch_rate: Optional[float] = None
    id_switch_transitions: int = 0

    # 9. Duplicate Identities & Track Fragmentation
    # Fragmentation: excess distinct system IDs assigned to the same GT entity
    fragmented_identities: int = 0
    spurious_system_tracks: int = 0
    duplicate_identities: int = 0

    # 10. Historical Retrieval Accuracy (HRA)
    # Fraction of reappearing objects correctly re-associated with historical entity ID
    historical_retrieval_accuracy: Optional[float] = None
    historical_retrieval_count: int = 0
    historical_retrieval_denominator: int = 0

    # 11. Raw Object Detection Metrics (B0 Detector Baseline)
    detection_true_positives: Optional[int] = None
    detection_false_positives: Optional[int] = None
    detection_false_negatives: Optional[int] = None
    detection_precision: Optional[float] = None
    detection_recall: Optional[float] = None
    detection_f1_score: Optional[float] = None

    # Overall Epistemic State Accuracy
    overall_state_accuracy: Optional[float] = None
    correct_state_evaluations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_observations_processed": self.total_observations_processed,
            "total_entity_evaluations": self.total_entity_evaluations,
            "false_removal_rate": self.false_removal_rate,
            "false_removal_count": self.false_removal_count,
            "false_removal_denominator": self.false_removal_denominator,
            "false_discovery_proportion": self.false_discovery_proportion,
            "false_discovery_count": self.false_discovery_count,
            "false_discovery_denominator": self.false_discovery_denominator,
            "removal_recall": self.removal_recall,
            "removal_recall_count": self.removal_recall_count,
            "removal_recall_denominator": self.removal_recall_denominator,
            "event_removal_recall": self.event_removal_recall,
            "event_removal_recall_count": self.event_removal_recall_count,
            "event_removal_recall_denominator": self.event_removal_recall_denominator,
            "confirmation_rate": self.confirmation_rate,
            "confirmed_change_events_count": self.confirmed_change_events_count,
            "total_eligible_change_events": self.total_eligible_change_events,
            "unconfirmed_change_events_count": self.unconfirmed_change_events_count,
            "mean_confirmation_delay_visits": self.mean_confirmation_delay_visits,
            "confirmation_delay_events_count": self.confirmation_delay_events_count,
            "unsupported_movement_rate": self.unsupported_movement_rate,
            "unsupported_movement_count": self.unsupported_movement_count,
            "unsupported_movement_denominator": self.unsupported_movement_denominator,
            "identity_consistency": self.identity_consistency,
            "identity_consistency_numerator": self.identity_consistency_numerator,
            "identity_consistency_denominator": self.identity_consistency_denominator,
            "id_switches": self.id_switches,
            "id_switch_rate": self.id_switch_rate,
            "id_switch_transitions": self.id_switch_transitions,
            "fragmented_identities": self.fragmented_identities,
            "spurious_system_tracks": self.spurious_system_tracks,
            "duplicate_identities": self.duplicate_identities,
            "historical_retrieval_accuracy": self.historical_retrieval_accuracy,
            "historical_retrieval_count": self.historical_retrieval_count,
            "historical_retrieval_denominator": self.historical_retrieval_denominator,
            "detection_true_positives": self.detection_true_positives,
            "detection_false_positives": self.detection_false_positives,
            "detection_false_negatives": self.detection_false_negatives,
            "detection_precision": self.detection_precision,
            "detection_recall": self.detection_recall,
            "detection_f1_score": self.detection_f1_score,
            "overall_state_accuracy": self.overall_state_accuracy,
            "correct_state_evaluations": self.correct_state_evaluations,
        }


@dataclass
class BenchmarkRunResult:
    """
    Complete, reproducible outcome of evaluating a configuration on a dataset.
    """
    run_id: str
    timestamp: str
    dataset_name: str
    dataset_provenance: str
    manifest_version: str
    is_pilot_dataset: bool
    is_genuine_physical: bool
    sequence_ids: List[str]
    config_id: str
    config_name: str
    config_fingerprint: str
    pipeline_type: str
    support_status: str
    is_partial_ablation: bool
    partial_ablation_limitations: List[str]
    effective_parameters: Dict[str, Any]
    metrics: BenchmarkMetrics
    unscorable_cases: Dict[str, str] = field(default_factory=dict)
    uninstantiated_gt_entities: List[str] = field(default_factory=list)
    execution_errors: Dict[str, str] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    sequence_reports: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "dataset_provenance": self.dataset_provenance,
            "manifest_version": self.manifest_version,
            "is_pilot_dataset": self.is_pilot_dataset,
            "is_genuine_physical": self.is_genuine_physical,
            "sequence_ids": self.sequence_ids,
            "config_id": self.config_id,
            "config_name": self.config_name,
            "config_fingerprint": self.config_fingerprint,
            "pipeline_type": self.pipeline_type,
            "support_status": self.support_status,
            "is_partial_ablation": self.is_partial_ablation,
            "partial_ablation_limitations": self.partial_ablation_limitations,
            "effective_parameters": self.effective_parameters,
            "metrics": self.metrics.to_dict(),
            "unscorable_cases": self.unscorable_cases,
            "uninstantiated_gt_entities": self.uninstantiated_gt_entities,
            "execution_errors": self.execution_errors,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "sequence_reports": self.sequence_reports,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def format_report(self) -> str:
        """
        Renders a human-readable benchmark evaluation report with strict provenance banner.
        """
        lines = []
        lines.append("=" * 86)
        lines.append(f" ObjectMemory Benchmark Evaluation Report: {self.config_id} ({self.config_name})")
        lines.append("=" * 86)

        # 1. Provenance Banner
        lines.append("DATASET PROVENANCE & SCIENTIFIC INTEGRITY:")
        lines.append(f"  Dataset Name:        {self.dataset_name}")
        lines.append(f"  Manifest Version:    {self.manifest_version}")
        lines.append(f"  Provenance Tier:     {self.dataset_provenance}")
        lines.append(f"  Genuine Physical:    {self.is_genuine_physical}")
        lines.append(f"  Exploratory Pilot:   {self.is_pilot_dataset}")
        if self.is_pilot_dataset:
            lines.append("  > [!WARNING] EXPLORATORY PILOT DATA: This dataset represents an exploratory pilot.")
            lines.append("  > Pilot datasets must NEVER be represented as a final physical-world validation dataset.")
        lines.append("")

        # 2. Configuration Fingerprint
        lines.append("CONFIGURATION METADATA:")
        lines.append(f"  Configuration ID:    {self.config_id}")
        lines.append(f"  Support Status:      {self.support_status}")
        lines.append(f"  Fingerprint (SHA):   {self.config_fingerprint}")
        lines.append(f"  Pipeline Type:       {self.pipeline_type}")
        if self.is_partial_ablation:
            lines.append("  Partial Ablation:    TRUE (Engine retains unconfigurable safeguards)")
            for lim in self.partial_ablation_limitations:
                lines.append(f"    - Note: {lim}")
        lines.append("")

        # 3. Core Evaluated Metrics
        m = self.metrics
        lines.append("SCIENTIFIC EVALUATION METRICS:")
        lines.append(f"  Total Observations:  {m.total_observations_processed}")
        lines.append(f"  Entity Evaluations:  {m.total_entity_evaluations}")

        if self.pipeline_type == "MEMORYLESS_DETECTION":
            lines.append("")
            lines.append("  OBJECT DETECTION BASELINE METRICS (Frame-level 1-to-1 matching, IoU >= 0.50):")
            p_str = f"{m.detection_precision:.1%}" if m.detection_precision is not None else "N/A"
            r_str = f"{m.detection_recall:.1%}" if m.detection_recall is not None else "N/A"
            f1_str = f"{m.detection_f1_score:.3f}" if m.detection_f1_score is not None else "N/A"
            lines.append(f"    - Precision:       {p_str} ({m.detection_true_positives} TP / {m.detection_true_positives + (m.detection_false_positives or 0)} Pred)")
            lines.append(f"    - Recall:          {r_str} ({m.detection_true_positives} TP / {m.detection_true_positives + (m.detection_false_negatives or 0)} GT)")
            lines.append(f"    - F1 Score:        {f1_str}")
            lines.append("    - Note:            mAP is not implemented in this evaluation pass.")
            lines.append("    - Temporal States: N/A (Stateless baseline does not maintain persistent tracks).")
        else:
            frr_str = f"{m.false_removal_rate:.1%}" if m.false_removal_rate is not None else "N/A"
            lines.append(f"  False Removal Rate:  {frr_str} ({m.false_removal_count} FP / {m.false_removal_denominator} Eligible Non-Removed Visits)")

            fdp_str = f"{m.false_discovery_proportion:.1%}" if m.false_discovery_proportion is not None else "N/A"
            lines.append(f"  False Discovery Prop:{fdp_str} ({m.false_discovery_count} FP / {m.false_discovery_denominator} Removal Predictions)")

            rr_str = f"{m.removal_recall:.1%}" if m.removal_recall is not None else "N/A"
            lines.append(f"  Removal Recall (vis):{rr_str} ({m.removal_recall_count} TP / {m.removal_recall_denominator} Removal Visits)")

            err_str = f"{m.event_removal_recall:.1%}" if m.event_removal_recall is not None else "N/A"
            lines.append(f"  Removal Recall (ev): {err_str} ({m.event_removal_recall_count} Confirmed / {m.event_removal_recall_denominator} Removal Events)")

            cr_str = f"{m.confirmation_rate:.1%}" if m.confirmation_rate is not None else "N/A"
            lines.append(f"  Confirmation Rate:   {cr_str} ({m.confirmed_change_events_count} Confirmed / {m.total_eligible_change_events} Change Events; {m.unconfirmed_change_events_count} Unconfirmed)")

            cd_str = f"{m.mean_confirmation_delay_visits:.1f} visits" if m.mean_confirmation_delay_visits is not None else "N/A"
            lines.append(f"  Confirmation Delay:  {cd_str} (Conditional mean among confirmed events; {m.confirmation_delay_events_count} events)")

            umr_str = f"{m.unsupported_movement_rate:.1%}" if m.unsupported_movement_rate is not None else "N/A"
            lines.append(f"  Unsupported Moves:   {umr_str} ({m.unsupported_movement_count} FP / {m.unsupported_movement_denominator} Stationary Visits)")

            ic_str = f"{m.identity_consistency:.1%}" if m.identity_consistency is not None else "N/A"
            lines.append(f"  Identity Continuity: {ic_str} ({m.identity_consistency_numerator} Dominant Matched / {m.identity_consistency_denominator} Total Present Visits)")

            lines.append(f"  ID Switches:         {m.id_switches} (across {m.id_switch_transitions} consecutive observed transitions)")
            lines.append(f"  Track Fragmentation: {m.fragmented_identities} excess IDs across GT entities (spurious tracks: {m.spurious_system_tracks})")

            hra_str = f"{m.historical_retrieval_accuracy:.1%}" if m.historical_retrieval_accuracy is not None else "N/A"
            lines.append(f"  Historical Re-ID:    {hra_str} ({m.historical_retrieval_count} / {m.historical_retrieval_denominator})")

            acc_str = f"{m.overall_state_accuracy:.1%}" if m.overall_state_accuracy is not None else "N/A"
            lines.append(f"  Overall State Acc:   {acc_str} ({m.correct_state_evaluations} / {m.total_entity_evaluations})")

        lines.append("")

        # 4. Unscorable Cases & Caveats
        if self.unscorable_cases:
            lines.append("UNSCORABLE CASES & EXCLUSIONS:")
            for k, reason in self.unscorable_cases.items():
                lines.append(f"  - {k}: {reason}")
            lines.append("")

        if self.uninstantiated_gt_entities:
            lines.append("UNINSTANTIATED GROUND-TRUTH ENTITIES:")
            for gid in self.uninstantiated_gt_entities:
                lines.append(f"  - {gid} (never detected or matched)")
            lines.append("")

        lines.append(f"Runtime: {self.runtime_seconds:.3f}s | Run ID: {self.run_id}")
        lines.append("=" * 86)
        return "\n".join(lines)


class BenchmarkRunner:
    """
    Standardized, isolated evaluation runner for ObjectMemory benchmarks.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        registrar: Optional[PlanarRegistrar] = None,
        signature_extractor: Optional[VisualSignatureExtractor] = None,
        allow_partial: bool = False,
        allow_blocked: bool = False,
    ) -> None:
        self.detector = detector
        self.registrar = registrar or PlanarRegistrar()
        self.signature_extractor = signature_extractor or VisualSignatureExtractor()
        self.allow_partial = allow_partial
        self.allow_blocked = allow_blocked

    def _get_detector(self) -> ObjectDetector:
        """Lazily obtains default detector if none provided."""
        if self.detector is not None:
            return self.detector
        try:
            return DetectorRegistry.get_default()
        except Exception:
            return MockObjectDetector()

    def run_manifest(
        self,
        manifest_or_path: Union[str, Path, DatasetManifest],
        config_ids: Optional[Sequence[str]] = None,
        sequence_ids: Optional[Sequence[str]] = None,
        dataset_root: Optional[Union[str, Path]] = None,
        validate_files: bool = False,
    ) -> Dict[str, BenchmarkRunResult]:
        """
        Loads and executes a benchmark manifest across requested configurations.
        """
        if isinstance(manifest_or_path, DatasetManifest):
            manifest = manifest_or_path
        else:
            manifest = ManifestLoader.load(
                manifest_path=manifest_or_path,
                dataset_root=dataset_root,
                check_files=validate_files,
            )

        target_configs = list(config_ids) if config_ids else ["B0", "B3", "A5"]

        results: Dict[str, BenchmarkRunResult] = {}
        for cid in target_configs:
            config = BenchmarkConfigFactory.create_config(
                config_id=cid,
                allow_blocked=self.allow_blocked,
                allow_partial=self.allow_partial,
            )
            res = self.run_config(manifest=manifest, config=config, sequence_ids=sequence_ids)
            results[cid] = res

        return results

    def run_config(
        self,
        manifest: DatasetManifest,
        config: BenchmarkConfig,
        sequence_ids: Optional[Sequence[str]] = None,
    ) -> BenchmarkRunResult:
        """
        Executes a single benchmark configuration across manifest sequences in complete isolation.
        """
        t0 = time.perf_counter()
        run_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # Reject BLOCKED configurations before any execution
        if config.support_status == ConfigurationSupportStatus.BLOCKED and not self.allow_blocked:
            raise UnsupportedConfigurationError(
                f"Configuration '{config.config_id}' is BLOCKED by engine architecture and cannot be executed."
            )

        seq_targets = manifest.sequences
        if sequence_ids:
            target_set = set(sequence_ids)
            seq_targets = [s for s in manifest.sequences if s.sequence_id in target_set]

        partial_limitations: List[str] = []
        is_partial = (config.support_status == ConfigurationSupportStatus.PARTIALLY_SUPPORTED)
        if is_partial:
            partial_limitations.append(config.notes)

        all_seq_reports: Dict[str, Any] = {}
        all_uninstantiated: Set[str] = set()
        exec_errors: Dict[str, str] = {}

        # Aggregate metric accumulators
        total_obs = 0
        total_evals = 0
        correct_state_evals = 0

        num_frr = 0
        den_frr = 0
        num_fdp = 0
        den_fdp = 0
        num_rr = 0
        den_rr = 0
        num_event_rr = 0
        den_event_rr = 0
        num_cr = 0
        den_cr = 0
        unconfirmed_events = 0
        num_umr = 0
        den_umr = 0

        ic_num_total = 0
        ic_den_total = 0
        id_switches_total = 0
        id_transitions_total = 0
        fragmented_identities_total = 0
        spurious_tracks_total = 0

        delay_visits_sum = 0
        delay_events_count = 0

        hra_num_total = 0
        hra_den_total = 0

        # B0 detector metric accumulators
        det_tp_total = 0
        det_fp_total = 0
        det_fn_total = 0

        unscorable_cases: Dict[str, str] = {}

        for seq in seq_targets:
            try:
                if config.pipeline_type == "MEMORYLESS_DETECTION":
                    seq_res = self._execute_b0_sequence(seq)
                    det_tp_total += seq_res.get("det_tp", 0)
                    det_fp_total += seq_res.get("det_fp", 0)
                    det_fn_total += seq_res.get("det_fn", 0)
                else:
                    seq_res = self._execute_temporal_sequence(seq, config)
                    num_frr += seq_res["frr_num"]
                    den_frr += seq_res["frr_den"]
                    num_fdp += seq_res["fdp_num"]
                    den_fdp += seq_res["fdp_den"]
                    num_rr += seq_res["rr_num"]
                    den_rr += seq_res["rr_den"]
                    num_event_rr += seq_res["event_rr_num"]
                    den_event_rr += seq_res["event_rr_den"]
                    num_cr += seq_res["cr_num"]
                    den_cr += seq_res["cr_den"]
                    unconfirmed_events += seq_res["unconfirmed_events"]
                    num_umr += seq_res["umr_num"]
                    den_umr += seq_res["umr_den"]

                    ic_num_total += seq_res["ic_num"]
                    ic_den_total += seq_res["ic_den"]
                    id_switches_total += seq_res["id_switches"]
                    id_transitions_total += seq_res["id_transitions"]
                    fragmented_identities_total += seq_res["fragmented_identities"]
                    spurious_tracks_total += seq_res["spurious_tracks"]

                    delay_visits_sum += seq_res["delay_visits_sum"]
                    delay_events_count += seq_res["delay_events_count"]

                    hra_num_total += seq_res["hra_num"]
                    hra_den_total += seq_res["hra_den"]

                all_seq_reports[seq.sequence_id] = seq_res
                all_uninstantiated.update(seq_res["uninstantiated_gt_entities"])

                total_obs += seq_res["observations_processed"]
                total_evals += seq_res["entity_evaluations"]
                correct_state_evals += seq_res["correct_state_evaluations"]

            except Exception as exc:
                logger.exception("Error executing sequence %s under config %s", seq.sequence_id, config.config_id)
                exec_errors[seq.sequence_id] = str(exc)

        # Build Metrics Object
        if config.pipeline_type == "MEMORYLESS_DETECTION":
            # B0 detection metrics
            p_den = det_tp_total + det_fp_total
            r_den = det_tp_total + det_fn_total
            det_p = (det_tp_total / p_den) if p_den > 0 else None
            det_r = (det_tp_total / r_den) if r_den > 0 else None
            det_f1 = (2.0 * det_p * det_r / (det_p + det_r)) if (det_p and det_r and (det_p + det_r) > 0) else None

            unscorable_cases["temporal_change_metrics"] = (
                "Stateless memoryless detection baseline: False Removal Rate, False Discovery Proportion, "
                "Removal Recall, Confirmation Rate/Delay, and Track Fragmentation are not defined."
            )

            metrics = BenchmarkMetrics(
                total_observations_processed=total_obs,
                total_entity_evaluations=total_evals,
                detection_true_positives=det_tp_total,
                detection_false_positives=det_fp_total,
                detection_false_negatives=det_fn_total,
                detection_precision=det_p,
                detection_recall=det_r,
                detection_f1_score=det_f1,
                overall_state_accuracy=(correct_state_evals / total_evals) if total_evals > 0 else None,
                correct_state_evaluations=correct_state_evals,
            )
        else:
            # Temporal metrics (B3, A5, etc.)
            frr = (num_frr / den_frr) if den_frr > 0 else None
            if den_frr == 0:
                unscorable_cases["false_removal_rate"] = "No eligible non-removed ground-truth entity visits evaluated."

            fdp = (num_fdp / den_fdp) if den_fdp > 0 else None
            if den_fdp == 0:
                unscorable_cases["false_discovery_proportion"] = "Zero confirmed removal predictions made by system."

            rr = (num_rr / den_rr) if den_rr > 0 else None
            if den_rr == 0:
                unscorable_cases["removal_recall"] = "No confirmed removal ground-truth visits present in sequence."

            event_rr = (num_event_rr / den_event_rr) if den_event_rr > 0 else None
            if den_event_rr == 0:
                unscorable_cases["event_removal_recall"] = "No confirmed removal ground-truth events present in sequence."

            cr = (num_cr / den_cr) if den_cr > 0 else None
            if den_cr == 0:
                unscorable_cases["confirmation_rate"] = "No eligible change events (removals or relocations) in ground truth."

            umr = (num_umr / den_umr) if den_umr > 0 else None
            if den_umr == 0:
                unscorable_cases["unsupported_movement_rate"] = "No stationary ground-truth entity visits evaluated."

            ic = (ic_num_total / ic_den_total) if ic_den_total > 0 else None
            if ic_den_total == 0:
                unscorable_cases["identity_consistency"] = "No observable present entity visits in ground truth."

            id_switch_rate = (id_switches_total / id_transitions_total) if id_transitions_total > 0 else None

            mean_delay = (delay_visits_sum / delay_events_count) if delay_events_count > 0 else None
            if delay_events_count == 0:
                unscorable_cases["mean_confirmation_delay_visits"] = "Zero persistent change events confirmed by system."

            hra = (hra_num_total / hra_den_total) if hra_den_total > 0 else None
            if hra_den_total == 0:
                unscorable_cases["historical_retrieval_accuracy"] = "No return-to-baseline / reappearance events present in ground truth."

            overall_acc = (correct_state_evals / total_evals) if total_evals > 0 else None

            metrics = BenchmarkMetrics(
                total_observations_processed=total_obs,
                total_entity_evaluations=total_evals,
                false_removal_rate=frr,
                false_removal_count=num_frr,
                false_removal_denominator=den_frr,
                false_discovery_proportion=fdp,
                false_discovery_count=num_fdp,
                false_discovery_denominator=den_fdp,
                removal_recall=rr,
                removal_recall_count=num_rr,
                removal_recall_denominator=den_rr,
                event_removal_recall=event_rr,
                event_removal_recall_count=num_event_rr,
                event_removal_recall_denominator=den_event_rr,
                confirmation_rate=cr,
                confirmed_change_events_count=num_cr,
                total_eligible_change_events=den_cr,
                unconfirmed_change_events_count=unconfirmed_events,
                mean_confirmation_delay_visits=mean_delay,
                confirmation_delay_events_count=delay_events_count,
                unsupported_movement_rate=umr,
                unsupported_movement_count=num_umr,
                unsupported_movement_denominator=den_umr,
                identity_consistency=ic,
                identity_consistency_numerator=ic_num_total,
                identity_consistency_denominator=ic_den_total,
                id_switches=id_switches_total,
                id_switch_rate=id_switch_rate,
                id_switch_transitions=id_transitions_total,
                fragmented_identities=fragmented_identities_total,
                spurious_system_tracks=spurious_tracks_total,
                duplicate_identities=fragmented_identities_total + spurious_tracks_total,
                historical_retrieval_accuracy=hra,
                historical_retrieval_count=hra_num_total,
                historical_retrieval_denominator=hra_den_total,
                overall_state_accuracy=overall_acc,
                correct_state_evaluations=correct_state_evals,
            )

        runtime = time.perf_counter() - t0

        # Construct comprehensive fingerprint capturing all effective settings
        detector = self._get_detector()
        full_effective_parameters = dict(config.effective_parameters)
        full_effective_parameters["detector_name"] = detector.__class__.__name__
        full_effective_parameters["manifest_version"] = manifest.manifest_version
        full_effective_parameters["dataset_name"] = manifest.dataset_name
        full_effective_parameters["benchmark_match_iou_threshold"] = 0.50

        return BenchmarkRunResult(
            run_id=run_id,
            timestamp=now_iso,
            dataset_name=manifest.dataset_name,
            dataset_provenance=manifest.provenance.value,
            manifest_version=manifest.manifest_version,
            is_pilot_dataset=manifest.is_pilot_dataset,
            is_genuine_physical=manifest.is_genuine_physical,
            sequence_ids=[s.sequence_id for s in seq_targets],
            config_id=config.config_id,
            config_name=config.name,
            config_fingerprint=config.deterministic_hash,
            pipeline_type=config.pipeline_type,
            support_status=config.support_status.value,
            is_partial_ablation=is_partial,
            partial_ablation_limitations=partial_limitations,
            effective_parameters=full_effective_parameters,
            metrics=metrics,
            unscorable_cases=unscorable_cases,
            uninstantiated_gt_entities=sorted(list(all_uninstantiated)),
            execution_errors=exec_errors,
            runtime_seconds=runtime,
            sequence_reports=all_seq_reports,
        )

    # -------------------------------------------------------------------------
    # Pipeline Execution: B0 Memoryless Detector Baseline
    # -------------------------------------------------------------------------
    def _execute_b0_sequence(self, seq: SequenceEntry) -> Dict[str, Any]:
        """
        Executes B0: Memoryless per-frame detector baseline directly against ground truth.
        Computes detection precision, recall, and F1. Sets temporal metrics to None.
        """
        detector = self._get_detector()
        observations = seq.observations

        total_evals = 0
        correct_state_evals = 0
        det_tp = 0
        det_fp = 0
        det_fn = 0

        for obs in observations:
            img = self._load_observation_image(obs)
            batch = detector.detect(img)
            dets = batch.detections

            gt_anns = obs.objects
            active_gt = [a for a in gt_anns if a.bounding_box is not None]

            # Greedy 1-to-1 match by class and IoU >= 0.50
            matched_gt_indices: Set[int] = set()
            used_det_indices: Set[int] = set()

            candidates = []
            for d_idx, d in enumerate(dets):
                for a_idx, a in enumerate(active_gt):
                    if d.label.lower() == a.class_name.lower():
                        iou = calculate_bbox_iou(d.raw_bbox, a.bounding_box)
                        if iou >= 0.50:
                            candidates.append((iou, d_idx, a_idx))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for iou, d_idx, a_idx in candidates:
                if d_idx not in used_det_indices and a_idx not in matched_gt_indices:
                    used_det_indices.add(d_idx)
                    matched_gt_indices.add(a_idx)

            frame_tp = len(matched_gt_indices)
            frame_fp = len(dets) - frame_tp
            frame_fn = len(active_gt) - frame_tp

            det_tp += frame_tp
            det_fp += frame_fp
            det_fn += frame_fn

            for a_idx, a in enumerate(active_gt):
                total_evals += 1
                is_detected = (a_idx in matched_gt_indices)
                predicted_state = (
                    EpistemicState.CONFIRMED_PRESENT.value
                    if is_detected
                    else EpistemicState.INSUFFICIENT_EVIDENCE.value
                )

                expected_state = a.object_state
                if expected_state == predicted_state or (
                    expected_state in (EpistemicState.CONFIRMED_PRESENT.value, "STATIONARY")
                    and is_detected
                ):
                    correct_state_evals += 1

            for a in [a for a in gt_anns if a.bounding_box is None]:
                total_evals += 1
                if a.object_state == EpistemicState.CONFIRMED_REMOVED.value:
                    pass  # B0 does not confirm removal

        return {
            "observations_processed": len(observations),
            "entity_evaluations": total_evals,
            "correct_state_evaluations": correct_state_evals,
            "det_tp": det_tp,
            "det_fp": det_fp,
            "det_fn": det_fn,
            "uninstantiated_gt_entities": [],
        }

    # -------------------------------------------------------------------------
    # Pipeline Execution: B3, A5, and Temporal Configurations
    # -------------------------------------------------------------------------
    def _execute_temporal_sequence(
        self,
        seq: SequenceEntry,
        config: BenchmarkConfig,
    ) -> Dict[str, Any]:
        """
        Executes a temporal memory configuration on an ordered sequence.
        Captures immutable per-visit prediction snapshots to eliminate historical leakage.
        Anchors 1-to-1 bipartite matching to initial observation to preserve relocated entities.
        """
        store = SQLiteMemoryStore(":memory:")
        temp_cfg = config.get_isolated_temporal_config() or TemporalEngineConfig()
        engine = TemporalMemoryEngine(
            store=store,
            config=temp_cfg,
            signature_extractor=self.signature_extractor,
        )

        detector = self._get_detector()
        observations = seq.observations

        canonical_ref_img: Optional[np.ndarray] = None
        visit_snapshots: Dict[int, VisitPredictionSnapshot] = {}
        obs_to_frame_matches: Dict[int, Dict[str, str]] = {}  # v_idx -> {gt_id: sys_id}

        # Step A: Chronological pipeline execution with immutable snapshot capture
        for v_idx, obs in enumerate(observations, start=1):
            img = self._load_observation_image(obs)
            h_img, w_img = img.shape[:2]

            # Co-registration against canonical reference image
            if canonical_ref_img is None:
                canonical_ref_img = img
                reg = self.registrar.register(img, img)
            else:
                reg = self.registrar.register(canonical_ref_img, img)

            is_reg_reliable = bool(
                reg.success
                and reg.homography is not None
                and (reg.inlier_ratio is None or reg.inlier_ratio >= temp_cfg.min_registration_inlier_ratio)
            )

            # Object detection
            batch = detector.detect(img)
            raw_dets = batch.detections

            # Canonical coordinate projection & visual signature extraction
            ref_w, ref_h = canonical_ref_img.shape[1], canonical_ref_img.shape[0]
            processed_dets: List[Detection] = []
            for det in raw_dets:
                canon_box: Optional[BoundingBox] = None
                if is_reg_reliable and reg.homography is not None:
                    canon_box = transform_query_bbox_to_reference(
                        det.raw_bbox,
                        reg.homography,
                        ref_size=(ref_w, ref_h),
                    )

                sig = self.signature_extractor.extract(img, det.raw_bbox)
                is_clipped = is_bbox_boundary_clipped(det.raw_bbox, (w_img, h_img), margin=temp_cfg.border_margin_px)

                processed_dets.append(
                    Detection(
                        label=det.label,
                        confidence=det.confidence,
                        raw_bbox=det.raw_bbox,
                        canonical_bbox=canon_box,
                        signature=sig,
                        metadata={"is_clipped": is_clipped},
                    )
                )

            # Temporal memory update
            v_in = VisitInput(
                session_id=obs.observation_id,
                session_name=f"{seq.sequence_id}_{obs.observation_id}",
                timestamp=obs.timestamp,
                image=img,
                registration=reg,
                detections=processed_dets,
                metadata={"sequence_id": seq.sequence_id},
            )
            rep = engine.process_visit(v_in)

            # Frame-level query-frame matching between GT annotations and processed detections
            # Both are measured in the exact same physical image canvas (query pixels)
            current_frame_matches: Dict[str, str] = {}
            active_anns = [a for a in obs.objects if a.bounding_box is not None]

            # Build query-frame match pairs (IoU >= 0.50, matching class)
            for ann in active_anns:
                best_det_idx = None
                best_det_iou = 0.0
                for d_idx, d in enumerate(processed_dets):
                    if d.label.lower() == ann.class_name.lower():
                        iou = calculate_bbox_iou(ann.bounding_box, d.raw_bbox)
                        if iou > best_det_iou:
                            best_det_iou = iou
                            best_det_idx = d_idx

                if best_det_idx is not None and best_det_iou >= 0.50:
                    matched_det = processed_dets[best_det_idx]
                    # Link to assigned entity
                    for u in rep.entity_updates:
                        if u.observation_id == obs.observation_id or u.new_state == EpistemicState.CONFIRMED_PRESENT:
                            pass
                    # Find entity by spatial overlap in store
                    for ent in store.list_entities():
                        if ent.label.lower() == ann.class_name.lower() and ent.canonical_bbox is not None:
                            if matched_det.canonical_bbox is not None:
                                c_iou = calculate_bbox_iou(matched_det.canonical_bbox, ent.canonical_bbox)
                                if c_iou >= 0.30:
                                    current_frame_matches[ann.object_id] = ent.entity_id
                                    break
                            elif ent.last_seen_session_id == obs.observation_id:
                                current_frame_matches[ann.object_id] = ent.entity_id
                                break

            obs_to_frame_matches[v_idx] = current_frame_matches

            # CAPTURE IMMUTABLE PER-VISIT PREDICTION SNAPSHOT
            # Every active entity state is frozen immediately after this visit
            active_entities = store.list_entities()
            rep_updates = {u.entity_id: u for u in (rep.entity_updates or [])}

            snapshots_for_visit: Dict[str, EntityVisitSnapshot] = {}
            for ent in active_entities:
                up = rep_updates.get(ent.entity_id)
                is_new = (ent.metadata.get("created_in_session") == obs.observation_id)

                if up is not None:
                    st = up.new_state
                    vis = up.visibility_status
                    expl = up.explanation
                    vac_streak = up.consecutive_vacant_visits
                    unobs_streak = ent.metadata.get("unobserved_streak", 0)
                    is_assoc = True
                elif is_new:
                    st = ent.current_state
                    vis = VisibilityStatus.CLEARLY_VISIBLE
                    expl = f"Newly instantiated entity in visit {v_idx}"
                    vac_streak = 0
                    unobs_streak = 0
                    is_assoc = True
                else:
                    # Inherit state from previous visit snapshot
                    prev_snap = visit_snapshots.get(v_idx - 1)
                    prev_ent = prev_snap.entities.get(ent.entity_id) if prev_snap else None
                    if prev_ent is not None:
                        st = prev_ent.epistemic_state
                        vis = prev_ent.visibility_status
                        expl = f"Carried from visit {prev_ent.visit_index}: {prev_ent.explanation}"
                        vac_streak = prev_ent.vacant_streak
                        unobs_streak = prev_ent.unobserved_streak
                        is_assoc = False
                    else:
                        st = ent.current_state
                        vis = VisibilityStatus.UNCERTAIN
                        expl = "State from store"
                        vac_streak = 0
                        unobs_streak = 0
                        is_assoc = False

                snapshots_for_visit[ent.entity_id] = EntityVisitSnapshot(
                    entity_id=ent.entity_id,
                    label=ent.label,
                    epistemic_state=st,
                    visibility_status=vis,
                    canonical_bbox=copy.deepcopy(ent.canonical_bbox),
                    raw_bbox=None,
                    is_associated=is_assoc,
                    is_newly_created=is_new,
                    registration_reliable=is_reg_reliable,
                    confidence=float(ent.metadata.get("initial_confidence", 1.0)),
                    vacant_streak=vac_streak,
                    unobserved_streak=unobs_streak,
                    explanation=expl,
                    visit_id=obs.observation_id,
                    visit_index=v_idx,
                    timestamp=obs.timestamp,
                    metadata=dict(ent.metadata),
                )

            visit_snapshots[v_idx] = VisitPredictionSnapshot(
                visit_index=v_idx,
                visit_id=obs.observation_id,
                timestamp=obs.timestamp,
                registration=reg,
                registration_reliable=is_reg_reliable,
                entities=snapshots_for_visit,
                raw_detections=raw_dets,
                unassociated_detections=rep.unassociated_detections,
            )

        # Step B: Stable 1-to-1 Bipartite Ground-Truth to System Entity Mapping
        # Anchored to initial observation visit to guarantee immunity against relocation
        v1_snapshot = visit_snapshots.get(1)
        v1_sys_entities = list(v1_snapshot.entities.values()) if v1_snapshot else []

        gt_to_sys, uninstantiated_gt = self._compute_stable_bipartite_mapping(
            seq=seq,
            v1_entities=v1_sys_entities,
            all_system_entities=store.list_entities(),
        )

        # Step C: Audited Metric Accounting Using Immutable Snapshots
        total_evals = 0
        correct_state_evals = 0
        num_frr = 0
        den_frr = 0
        num_fdp = 0
        den_fdp = 0
        num_rr = 0
        den_rr = 0
        num_umr = 0
        den_umr = 0

        # Change event tracking (Removal and Relocation)
        # ev_key -> (onset_v_idx, confirmed_v_idx)
        first_change_onset: Dict[str, int] = {}
        first_change_confirmed: Dict[str, int] = {}
        removal_events_onset: Set[str] = set()
        removal_events_confirmed: Set[str] = set()

        # Reappearance / return tracking
        reappearance_events: Dict[str, Tuple[str, bool]] = {}

        for v_idx, obs in enumerate(observations, start=1):
            snap = visit_snapshots[v_idx]

            for ann in obs.objects:
                total_evals += 1
                gid = ann.object_id
                sys_id = gt_to_sys.get(gid)
                expected_state_str = ann.object_state

                # Resolve observed state strictly from this visit's snapshot
                ent_snap = snap.entities.get(sys_id) if sys_id else None
                observed_state = ent_snap.epistemic_state if ent_snap else None
                obs_str = observed_state.value if observed_state is not None else "UNKNOWN"

                # Check state match
                is_match = False
                if sys_id is not None and observed_state is not None:
                    if obs_str == expected_state_str:
                        is_match = True
                    elif expected_state_str in ("STATIONARY", "MOVED_BACK_TO_BASELINE") and obs_str == EpistemicState.CONFIRMED_PRESENT.value:
                        is_match = True

                if is_match:
                    correct_state_evals += 1

                # 1. False Removal Rate (False Positive Rate: FP / (FP + TN) over non-removed visits)
                if expected_state_str != EpistemicState.CONFIRMED_REMOVED.value:
                    den_frr += 1
                    if obs_str == EpistemicState.CONFIRMED_REMOVED.value:
                        num_frr += 1

                # 2. False Discovery Proportion (FP / (TP + FP) over removal predictions)
                if obs_str == EpistemicState.CONFIRMED_REMOVED.value:
                    den_fdp += 1
                    if expected_state_str != EpistemicState.CONFIRMED_REMOVED.value:
                        num_fdp += 1

                # 3. Removal Recall (TP / (TP + FN) over removal visits)
                if expected_state_str == EpistemicState.CONFIRMED_REMOVED.value:
                    den_rr += 1
                    if obs_str == EpistemicState.CONFIRMED_REMOVED.value:
                        num_rr += 1

                    # Event-level removal tracking
                    rem_ev_key = f"{gid}_removal"
                    removal_events_onset.add(rem_ev_key)
                    if obs_str == EpistemicState.CONFIRMED_REMOVED.value:
                        removal_events_confirmed.add(rem_ev_key)

                # 4. Unsupported Movement Rate (FP / (FP + TN) over stationary visits)
                if expected_state_str in (EpistemicState.CONFIRMED_PRESENT.value, "STATIONARY"):
                    den_umr += 1
                    if obs_str == EpistemicState.CONFIRMED_MOVED.value:
                        num_umr += 1

                # Change event confirmation tracking
                if expected_state_str in (EpistemicState.CONFIRMED_REMOVED.value, EpistemicState.CONFIRMED_MOVED.value):
                    ev_key = f"{gid}_{expected_state_str}"
                    if ev_key not in first_change_onset:
                        first_change_onset[ev_key] = v_idx
                    if obs_str == expected_state_str and ev_key not in first_change_confirmed:
                        first_change_confirmed[ev_key] = v_idx

                # Historical re-identification tracking
                is_return_ev = (
                    "return" in (ann.evidence_notes or "").lower()
                    or ann.event_id == "return_to_baseline"
                    or (ann.metadata and ann.metadata.get("expected_event") == "return_to_baseline")
                )
                if is_return_ev and sys_id is not None:
                    matched_in_v = obs_to_frame_matches.get(v_idx, {}).get(gid)
                    reassociated = (matched_in_v == sys_id)
                    reappearance_events[f"{gid}_v{v_idx}"] = (sys_id, reassociated)

        # 5. Identity Consistency (Penalizes missed observations!)
        # Denominator: total eligible ground-truth present visits
        # Numerator: dominant system entity matches
        ic_numerator = 0
        ic_denominator = 0
        id_switches = 0
        id_transitions = 0
        assigned_sys_ids_per_gt: Dict[str, Set[str]] = {}

        for gid in {ann.object_id for obs in observations for ann in obs.objects}:
            # Find all visits where this entity was physically present and observable
            present_visits_count = 0
            matched_seq = []
            assigned_sys_ids_per_gt[gid] = set()

            for v_idx, obs in enumerate(observations, start=1):
                ann = next((a for a in obs.objects if a.object_id == gid), None)
                if ann is not None and ann.bounding_box is not None and ann.visibility_status != "OUT_OF_FRAME":
                    present_visits_count += 1
                    s_id = obs_to_frame_matches.get(v_idx, {}).get(gid)
                    if s_id is not None:
                        matched_seq.append(s_id)
                        assigned_sys_ids_per_gt[gid].add(s_id)

            ic_denominator += present_visits_count
            if matched_seq:
                counts: Dict[str, int] = {}
                for s in matched_seq:
                    counts[s] = counts.get(s, 0) + 1
                dom_count = max(counts.values())
                ic_numerator += dom_count

                for k in range(1, len(matched_seq)):
                    id_transitions += 1
                    if matched_seq[k] != matched_seq[k - 1]:
                        id_switches += 1

        # 6. Duplicate Identities & Track Fragmentation
        # Fragmentation: sum of excess system IDs assigned to the same GT entity
        fragmented_identities = sum(
            max(0, len(s_set) - 1) for s_set in assigned_sys_ids_per_gt.values()
        )
        all_sys_entities = store.list_entities()
        mapped_sys_ids = set(gt_to_sys.values())
        spurious_tracks = len([ent for ent in all_sys_entities if ent.entity_id not in mapped_sys_ids])

        # 7. Confirmation Rate & Conditional Delay
        total_change_events = len(first_change_onset)
        confirmed_change_events = len(first_change_confirmed)
        unconfirmed_change_events = total_change_events - confirmed_change_events

        delay_visits_sum = 0
        delay_events_count = 0
        for ev_key, onset_v in first_change_onset.items():
            if ev_key in first_change_confirmed:
                confirm_v = first_change_confirmed[ev_key]
                delay = confirm_v - onset_v + 1
                delay_visits_sum += delay
                delay_events_count += 1

        # 8. Event-Level Removal Recall
        event_rr_num = len(removal_events_confirmed)
        event_rr_den = len(removal_events_onset)

        # 9. Historical Retrieval Accuracy
        hra_num = sum(1 for _, reassociated in reappearance_events.values() if reassociated)
        hra_den = len(reappearance_events)

        return {
            "observations_processed": len(observations),
            "entity_evaluations": total_evals,
            "correct_state_evaluations": correct_state_evals,
            "frr_num": num_frr,
            "frr_den": den_frr,
            "fdp_num": num_fdp,
            "fdp_den": den_fdp,
            "rr_num": num_rr,
            "rr_den": den_rr,
            "event_rr_num": event_rr_num,
            "event_rr_den": event_rr_den,
            "cr_num": confirmed_change_events,
            "cr_den": total_change_events,
            "unconfirmed_events": unconfirmed_change_events,
            "umr_num": num_umr,
            "umr_den": den_umr,
            "ic_num": ic_numerator,
            "ic_den": ic_denominator,
            "id_switches": id_switches,
            "id_transitions": id_transitions,
            "fragmented_identities": fragmented_identities,
            "spurious_tracks": spurious_tracks,
            "delay_visits_sum": delay_visits_sum,
            "delay_events_count": delay_events_count,
            "hra_num": hra_num,
            "hra_den": hra_den,
            "uninstantiated_gt_entities": uninstantiated_gt,
        }

    # -------------------------------------------------------------------------
    # Helper: Anchored 1-to-1 Bipartite Ground Truth Mapping
    # -------------------------------------------------------------------------
    def _compute_stable_bipartite_mapping(
        self,
        seq: SequenceEntry,
        v1_entities: List[EntityVisitSnapshot],
        all_system_entities: List[WorkbenchEntity],
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Computes 1-to-1 bipartite assignment between ground truth and system tracks.
        Anchors matching to the initial observation visit so that relocated or removed
        entities retain their stable identity throughout the sequence.
        """
        all_gt_ids = sorted(list({ann.object_id for obs in seq.observations for ann in obs.objects}))
        if not all_gt_ids:
            return {}, []

        # Find initial bounding box and class for each GT entity
        gt_initial_info: Dict[str, Tuple[str, Optional[BoundingBox]]] = {}
        for gid in all_gt_ids:
            for obs in seq.observations:
                ann = next((a for a in obs.objects if a.object_id == gid), None)
                if ann is not None and ann.bounding_box is not None:
                    gt_initial_info[gid] = (ann.class_name.lower(), ann.bounding_box)
                    break
            if gid not in gt_initial_info:
                sample_ann = next(a for obs in seq.observations for a in obs.objects if a.object_id == gid)
                gt_initial_info[gid] = (sample_ann.class_name.lower(), None)

        gt_ids = list(gt_initial_info.keys())

        # Candidate system entities (prefer initial canonical boxes)
        sys_candidate_info: List[Tuple[str, str, Optional[BoundingBox]]] = []
        for s in all_system_entities:
            # Check if initial_canonical_bbox is preserved in metadata
            init_canon = None
            if s.metadata and "initial_canonical_bbox" in s.metadata:
                raw_init = s.metadata["initial_canonical_bbox"]
                if raw_init and len(raw_init) == 4:
                    init_canon = BoundingBox(raw_init[0], raw_init[1], raw_init[2], raw_init[3])
            if init_canon is None:
                init_canon = s.canonical_bbox
            sys_candidate_info.append((s.entity_id, s.label.lower(), init_canon))

        cost_matrix = np.full((len(gt_ids), len(sys_candidate_info)), 1000.0, dtype=np.float64)

        for g_idx, gid in enumerate(gt_ids):
            g_class, g_box = gt_initial_info[gid]
            for s_idx, (s_id, s_class, s_box) in enumerate(sys_candidate_info):
                if s_class != g_class:
                    continue
                if g_box is not None and s_box is not None:
                    iou = calculate_bbox_iou(g_box, s_box)
                    dist = calculate_centroid_distance(g_box, s_box) or 999.0
                    cost = (1.0 - iou) if iou >= 0.05 else (1.0 + dist / 150.0 if dist <= 150.0 else 1000.0)
                    cost_matrix[g_idx, s_idx] = cost
                elif g_box is None and s_box is None:
                    cost_matrix[g_idx, s_idx] = 10.0

        assignments = solve_optimal_assignment(cost_matrix)

        gt_to_sys: Dict[str, str] = {}
        mapped_sys_ids: Set[str] = set()

        for r, c in assignments:
            if cost_matrix[r, c] < 999.0:
                gid = gt_ids[r]
                sid = sys_candidate_info[c][0]
                gt_to_sys[gid] = sid
                mapped_sys_ids.add(sid)

        uninstantiated = [gid for gid in all_gt_ids if gid not in gt_to_sys]
        return gt_to_sys, uninstantiated

    def _load_observation_image(self, obs: ObservationEntry) -> np.ndarray:
        """Loads observation image safely, falling back to a deterministic canvas if missing."""
        if obs.image_path and obs.image_path.exists() and obs.image_path.is_file():
            img = cv2.imread(str(obs.image_path))
            if img is not None:
                return img
        return np.zeros((480, 640, 3), dtype=np.uint8)
