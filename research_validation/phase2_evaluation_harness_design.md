# ObjectMemory: Phase 2 Evaluation Harness Architecture
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Status:** Evaluation Harness Architectural Specification

---

## 1. Audit of Existing Evaluation Codebase

Before designing new harness modules, the existing evaluation and benchmarking assets were inspected:

```
src/objectmemory/
├── baseline/
│   ├── comparator.py          # PairwiseComparator, IoU, centroid distance, Kuhn-Munkres solver [ALREADY IMPLEMENTED]
│   └── models.py              # PairwiseMatchConfig, PairwiseComparisonReport [ALREADY IMPLEMENTED]
├── benchmarks/
│   ├── ground_truth_matcher.py# GroundTruthMatcher: bipartite matching between GT and system tracks [ALREADY IMPLEMENTED]
│   ├── real_evaluator.py      # RealBenchmarkEvaluator: P/R/F1, Latency & CPU profiling [ALREADY IMPLEMENTED]
│   ├── temporal_evaluator.py  # TemporalBenchmarkEvaluator: Synthetic sequence evaluation [ALREADY IMPLEMENTED]
│   ├── synthetic_dataset.py   # Synthetic affine warp generator [ALREADY IMPLEMENTED]
│   └── temporal_dataset.py    # Synthetic temporal scenario generator [ALREADY IMPLEMENTED]
└── datasets/
    ├── physical_workbench.py  # PhysicalWorkbenchDataset data models and validators [ALREADY IMPLEMENTED]
    ├── controlled_dataset.py  # Controlled dataset models [ALREADY IMPLEMENTED]
    └── real_dataset.py        # Real-image dataset loaders [ALREADY IMPLEMENTED]
```

### Assessment of Existing Assets:
1. **`GroundTruthMatcher` (`ground_truth_matcher.py`):** Fully implements bipartite assignment between ground-truth entities and system tracks via Kuhn-Munkres, tracking identity switches, duplicate tracks, and movement recall. **(Reuse directly)**.
2. **`PairwiseComparator` (`comparator.py`):** Fully implements the memoryless pairwise comparison logic needed for Baselines B0 and B1. **(Reuse directly)**.
3. **`PhysicalWorkbenchDataset` (`physical_workbench.py`):** Fully defines physical annotations, visibility states, and absence reasons. **(Reuse directly)**.
4. **Gaps to Address:**
   - Existing evaluators lack a unified CLI runner that takes a configuration flag (`--baseline B0|B1|B2|B3`, `--ablation A1..A6`) and outputs standardized JSON/CSV reports.
   - Absence of an explicit Dataset Manifest parser that binds physical sequence image paths, calibration files, and ground-truth annotation files into an immutable replay queue.

---

## 2. Evaluation Harness Architecture & Modules

The Phase 2 Evaluation Harness is designed around clean separation of concerns:

```
[Dataset Manifest (YAML/JSON)]
             |
             v
    [ManifestLoader] --------> [GroundTruthAnnotationLoader]
             |                                |
             v                                v
     [SequenceReplayer] ------------> [GroundTruthMatcher]
             |                                |
    +--------+--------+                       v
    | Baseline Injector| ------------> [MetricCalculator]
    | (B0, B1, B2, B3) |                       |
    | Ablation Factory |                       v
    | (A1 .. A6)       |             [ReportGenerator]
    +-----------------+              /               \
                                    v                 v
                          [Per-Sequence JSON]   [Aggregate CSV + MD]
```

---

## 3. Detailed Component Specifications

### A. Dataset Manifest Schema (`manifest.json`)
Binds physical directories, visit order, timestamps, and calibration files into a reproducible dataset specification:

```json
{
  "manifest_version": "2.0",
  "dataset_name": "PhysicalWorkbench-Benchmark",
  "dataset_root": "data/physical_workbench",
  "sequences": [
    {
      "sequence_id": "PHYS-01-STAT",
      "challenge_mode": "STATIONARY_LIGHTING_DRIFT",
      "reference_image": "PHYS-01-STAT/visit_01.png",
      "calibration_file": "PHYS-01-STAT/calibration.json",
      "ground_truth_file": "PHYS-01-STAT/ground_truth.json",
      "visits": [
        {
          "visit_id": "visit_01",
          "image_path": "PHYS-01-STAT/visit_01.png",
          "timestamp": "2026-11-14T09:00:00Z"
        },
        {
          "visit_id": "visit_02",
          "image_path": "PHYS-01-STAT/visit_02.png",
          "timestamp": "2026-11-14T09:30:00Z"
        }
      ]
    }
  ]
}
```

### B. Configuration Factory (`config_factory.py`)
Dynamically constructs isolated `TemporalEngineConfig` and pipeline wrappers for baselines and ablations without touching application code:

```python
class BenchmarkConfigFactory:
    """Produces frozen configuration objects for Phase 2 baselines and ablations."""

    @staticmethod
    def get_baseline_config(baseline_id: str) -> Tuple[TemporalEngineConfig, Dict[str, Any]]:
        if baseline_id == "B3":
            # Proposed complete system (audited runtime defaults)
            cfg = TemporalEngineConfig(
                signature_similarity_threshold=0.65,
                ambiguity_margin=0.12,
                vacancy_similarity_max=0.40,
                vacancy_visits_for_removal=2,
                unobserved_visits_for_removal=3,
                min_registration_inlier_ratio=0.20,
                min_overlap_fraction=0.70,
                movement_displacement_threshold=35.0,
            )
            return cfg, {"mode": "FULL_TEMPORAL"}
        elif baseline_id == "B2":
            # Ungated temporal reasoning (disable reliability safeguards)
            cfg = TemporalEngineConfig(
                min_registration_inlier_ratio=0.0, # Accept all alignments
                min_overlap_fraction=0.0,          # Disable FOV check
                vacancy_similarity_max=1.0,        # Any non-detection considered vacant
            )
            return cfg, {"mode": "UNGATED_TEMPORAL"}
        elif baseline_id == "B1":
            # Tracking association only
            return TemporalEngineConfig(), {"mode": "PAIRWISE_TRACKING"}
        elif baseline_id == "B0":
            # Memoryless detection
            return TemporalEngineConfig(), {"mode": "MEMORYLESS"}
        raise ValueError(f"Unknown baseline: {baseline_id}")

    @staticmethod
    def get_ablation_config(ablation_id: str) -> Tuple[TemporalEngineConfig, Dict[str, Any]]:
        # Inherit B3 defaults, override single audited mechanism
        cfg, meta = BenchmarkConfigFactory.get_baseline_config("B3")
        meta["ablation_id"] = ablation_id
        if ablation_id == "A1":
            cfg.min_registration_inlier_ratio = 0.0 # No registration freeze
        elif ablation_id == "A2":
            cfg.vacancy_similarity_max = 1.0        # No visual vacancy verification
        elif ablation_id == "A3":
            cfg.min_overlap_fraction = 0.0          # No FOV safeguard
        elif ablation_id == "A4":
            cfg.vacancy_visits_for_removal = 1      # Single-visit confirmation
        elif ablation_id == "A5":
            cfg.ambiguity_margin = 0.0              # No ambiguity margin
        return cfg, meta
```

### C. Metric Computation Engine (`metrics.py`)
Computes unbiased, statistically defensible metrics using the equations defined in `phase2_benchmark_protocol.md`:
- Detection $P, R, F1$ (IoU $\ge 0.45$).
- Identity Switches ($ID_{\text{sw}}$) and Duplicate Track Rate ($DTR$).
- Movement $P, R, F1$ and Unsupported Movement Rate ($UMR$).
- Removal Recall ($RR$) and False Removal on Present Rate ($FRPR$).
- Removal Confirmation Latency ($RCL$).
- Bootstrap 95% Confidence Intervals:
  ```python
  def bootstrap_metric_ci(
      values: List[float], n_bootstrap: int = 1000, ci: float = 0.95
  ) -> Tuple[float, float, float]:
      """Computes median and [lower, upper] empirical bootstrap confidence interval."""
      boot_medians = []
      n = len(values)
      for _ in range(n_bootstrap):
          sample = np.random.choice(values, size=n, replace=True)
          boot_medians.append(float(np.mean(sample)))
      alpha = (1.0 - ci) / 2.0
      lower = float(np.percentile(boot_medians, alpha * 100))
      upper = float(np.percentile(boot_medians, (1.0 - alpha) * 100))
      return float(np.mean(values)), lower, upper
  ```

### D. Prediction & Report Output Schemas

#### 1. Per-Sequence Evaluation JSON (`{seq_id}_{config_id}.json`)
```json
{
  "sequence_id": "PHYS-03-REM",
  "config_id": "B3",
  "git_commit_sha": "7a8b9c0d",
  "execution_timestamp": "2026-11-14T11:00:00Z",
  "summary_metrics": {
    "total_evaluations": 48,
    "state_accuracy": 0.958,
    "detection_f1": 0.925,
    "identity_switches": 0,
    "duplicate_tracks": 0,
    "false_removals_count": 0,
    "false_removal_on_present_rate": 0.000,
    "correct_removals_count": 6,
    "removal_recall": 1.000,
    "avg_removal_confirmation_latency": 2.0,
    "unsupported_movement_rate": 0.000
  },
  "evaluations_log": [
    {
      "visit_id": "visit_03",
      "gt_entity_id": "tool_multimeter_01",
      "sys_entity_id": "ent_01",
      "expected_state": "CONFIRMED_PRESENT",
      "observed_state": "CONFIRMED_PRESENT",
      "is_correct": true,
      "explanation": "Object remains present (displacement 2.3px <= 35.0px)."
    }
  ]
}
```

#### 2. Master Comparison CSV (`master_benchmark_results.csv`)
Columns:
`experiment_id`, `dataset`, `sequence_id`, `configuration_id`, `state_accuracy`, `detection_f1`, `id_switches`, `duplicate_track_rate`, `false_removal_rate`, `false_removal_on_present_rate`, `removal_recall`, `removal_confirmation_latency`, `unsupported_movement_rate`, `mean_reproj_error_px`.

---

## 4. Reproducible Command-Line Workflow

The harness is executed via a standardized CLI runner:

```powershell
# Run Baseline B3 (Full Proposed System) on the Physical Benchmark
python -m objectmemory.benchmarks.runner `
  --manifest data/manifests/physical_benchmark_manifest.json `
  --baseline B3 `
  --out-dir artifacts/results/physical/B3

# Run Ablation A2 (No Vacancy Check)
python -m objectmemory.benchmarks.runner `
  --manifest data/manifests/physical_benchmark_manifest.json `
  --ablation A2 `
  --out-dir artifacts/results/physical/A2

# Generate Cross-Baseline Comparative Report Table
python -m objectmemory.benchmarks.report_generator `
  --results-dir artifacts/results/physical `
  --output-format markdown `
  --out-file research_validation/physical_benchmark_comparison.md
```

---

## 5. Architectural Implementation Status Audit

| Harness Subsystem | Sub-Component | Classification | Source Code Location |
|---|---|---|---|
| **Data Ingestion** | Physical Annotation Models | **ALREADY IMPLEMENTED** | `src/objectmemory/datasets/physical_workbench.py:58-95` |
| **Data Ingestion** | Sequence Manifest Parser | **REQUIRES IMPLEMENTATION** | Proposed in `src/objectmemory/benchmarks/manifest.py` |
| **Ground-Truth Matching** | Kuhn-Munkres Entity Bipartite Matcher | **ALREADY IMPLEMENTED** | `src/objectmemory/benchmarks/ground_truth_matcher.py:107-160` |
| **Ground-Truth Matching** | Track Switching & Duplicate Detection | **ALREADY IMPLEMENTED** | `src/objectmemory/benchmarks/ground_truth_matcher.py:71-78` |
| **Baseline Architecture** | B0 Memoryless Pairwise Matcher | **ALREADY IMPLEMENTED** | `src/objectmemory/baseline/comparator.py:150-250` |
| **Baseline Architecture** | B1 Standard Hungarian Association | **ALREADY IMPLEMENTED** | `src/objectmemory/temporal/associator.py:168-187` |
| **Baseline Architecture** | B2 Ungated Temporal Reasoning | **REQUIRES IMPLEMENTATION** | Eval-mode harness config factory needed |
| **Baseline Architecture** | B3 Complete ObjectMemory | **ALREADY IMPLEMENTED** | `src/objectmemory/temporal/engine.py:230-464` |
| **Ablation Engine** | A1–A6 Configuration Factory | **REQUIRES IMPLEMENTATION** | Proposed in `src/objectmemory/benchmarks/config_factory.py` |
| **Metrics Engine** | Precision, Recall, F1, Latency | **PARTIALLY IMPLEMENTED** | In `real_evaluator.py`; needs bootstrap CI calculation |
| **Benchmarking CLI** | Unified Batch Benchmark Runner | **REQUIRES IMPLEMENTATION** | Proposed in `src/objectmemory/benchmarks/runner.py` |
| **Physical Dataset** | Physical Benchmark Capture Data | **REQUIRES DATA** | Physical images to be captured under Protocol 2.0 |
| **OpenLORIS Dataset** | Office Traversal Annotations | **REQUIRES DATA** | Phase 1E annotations to be established |
| **Hypothesis Verification**| Empirical Performance Superiority | **REQUIRES EXPERIMENTAL VALIDATION** | Awaiting benchmark execution runs |

---

## 6. Prioritized Implementation Plan (Only Genuinely Missing Code)

To make the evaluation harness executable without adding unnecessary bloat, only three lean modules need to be implemented in `src/objectmemory/benchmarks/`:

1. **Step 1: `src/objectmemory/benchmarks/manifest.py` (~120 lines):**
   - Simple JSON/YAML parser that reads `manifest.json`, validates that images and annotations exist on disk, and yields `(visit_input, ground_truth_annotations)` tuples in chronological order.
2. **Step 2: `src/objectmemory/benchmarks/config_factory.py` (~90 lines):**
   - Implements `BenchmarkConfigFactory` to return isolated `TemporalEngineConfig` instances for B0–B3 and A1–A6.
3. **Step 3: `src/objectmemory/benchmarks/runner.py` (~200 lines):**
   - Standard CLI runner leveraging the existing `GroundTruthMatcher` and `TemporalMemoryEngine`.
   - Iterates through the manifest sequences, writes isolated SQLite databases, computes metrics, and outputs JSON summaries.

This completely prepares the test infrastructure. A researcher can implement these three harness files in less than an hour and execute the first physical or OpenLORIS benchmark immediately.
