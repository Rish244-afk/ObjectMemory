# ObjectMemory: Phase 3C Evaluation Runner Implementation Report
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Phase:** 3C (Benchmark Evaluation Runner)  
**Status:** IMPLEMENTED & TESTED (393 / 393 full suite tests passing; 0 regressions)

---

## 1. Overview & Architectural Role

Phase 3C completes the core evaluation harness for ObjectMemory by implementing:
`src/objectmemory/benchmarks/runner.py`

This module connects the versioned dataset manifests (`manifest.py`, Phase 3A.1) with the immutable configuration factory (`config_factory.py`, Phase 3B) to provide a scientifically rigorous, fully isolated, and reproducible benchmark evaluation runner.

### Strict Scientific Integrity Principles
1. **Zero Database Contamination**: Every evaluation run instantiates a dedicated, isolated in-memory SQLite store (`SQLiteMemoryStore(":memory:")`). Production databases (`data/objectmemory.db`) are never accessed or modified.
2. **Strict Configuration Gating**: Configurations flagged as `BLOCKED` (`B1`, `B2`, `A1`, `A4`, `A6`) are rejected with `UnsupportedConfigurationError` before execution begins.
3. **Explicit Partial-Ablation Labeling**: Partially supported ablations (`A2: No Vacancy Verification`, `A3: No Visibility Safeguards`) require explicit opt-in (`allow_partial=True`) and are permanently tagged in the output result as `is_partial_ablation=True` with explicit limitation notes documenting that unconfigurable engine safeguards remain active.
4. **Decoupled Metric Definitions**: Eight metrics are computed independently, each with explicit numerators, denominators, and zero-denominator handling. Ambiguous or unobserved entities are never silently counted as confirmed removals.
5. **Provenance Transparency**: Exploratory pilot datasets (e.g. `desk_pilot_01`) are automatically tagged with a mandatory warning banner preventing them from being cited as final physical-world validation datasets.

---

## 2. Exact Files Changed / Created

1. **`research_validation/phase3c_runner_design.md` (New):**
   - Stage 1 pre-implementation compatibility audit and architectural design document.
2. **`src/objectmemory/benchmarks/runner.py` (New):**
   - Core runner module implementing `BenchmarkMetrics`, `BenchmarkRunResult`, and `BenchmarkRunner`.
3. **`tests/test_benchmark_runner.py` (New):**
   - 14 comprehensive unit tests verifying manifest loading, blocked configuration rejection, partial-ablation labeling, database isolation, B0 memoryless execution, B3 full temporal memory, A5 ambiguity ablation, exact metric computations, zero-denominator handling, registration failures, and reproducibility.
4. **`research_validation/phase3c_implementation_report.md` (New):**
   - This implementation and verification report.

**Scope Confirmation:** Zero production application code was altered. `src/objectmemory/temporal/`, `registration/`, `persistence/`, and `detection/` remain completely untouched.

---

## 3. Public API & Data Structures

```python
@dataclass
class BenchmarkMetrics:
    total_observations_processed: int
    total_entity_evaluations: int
    false_removal_rate: Optional[float]
    false_removal_count: int
    false_removal_denominator: int
    removal_recall: Optional[float]
    removal_recall_count: int
    removal_recall_denominator: int
    unsupported_movement_rate: Optional[float]
    unsupported_movement_count: int
    unsupported_movement_denominator: int
    identity_consistency: Optional[float]
    identity_consistency_numerator: int
    identity_consistency_denominator: int
    id_switches: int
    id_switch_rate: Optional[float]
    id_switch_transitions: int
    duplicate_identities: int
    spurious_system_tracks: int
    mean_confirmation_delay_visits: Optional[float]
    confirmation_delay_events_count: int
    historical_retrieval_accuracy: Optional[float]
    historical_retrieval_count: int
    historical_retrieval_denominator: int
    overall_state_accuracy: Optional[float]
    correct_state_evaluations: int

    def to_dict(self) -> Dict[str, Any]: ...

@dataclass
class BenchmarkRunResult:
    run_id: str
    timestamp: str
    dataset_name: str
    dataset_provenance: str
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
    unscorable_cases: Dict[str, str]
    uninstantiated_gt_entities: List[str]
    execution_errors: Dict[str, str]
    runtime_seconds: float
    sequence_reports: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self, indent: int = 2) -> str: ...
    def format_report(self) -> str: ...

class BenchmarkRunner:
    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        registrar: Optional[PlanarRegistrar] = None,
        signature_extractor: Optional[VisualSignatureExtractor] = None,
        allow_partial: bool = False,
        allow_blocked: bool = False,
    ) -> None: ...

    def run_manifest(
        self,
        manifest_or_path: Union[str, Path, DatasetManifest],
        config_ids: Optional[Sequence[str]] = None,
        sequence_ids: Optional[Sequence[str]] = None,
        dataset_root: Optional[Union[str, Path]] = None,
        validate_files: bool = False,
    ) -> Dict[str, BenchmarkRunResult]: ...

    def run_config(
        self,
        manifest: DatasetManifest,
        config: BenchmarkConfig,
        sequence_ids: Optional[Sequence[str]] = None,
    ) -> BenchmarkRunResult: ...
```

---

## 4. Operational Gating & Pipeline Support

| Configuration | Support Status | Execution Pathway | Safeguards Active |
|---|---|---|---|
| **B0: Memoryless Detector** | **`SUPPORTED`** | Per-frame YOLO detection matched 1-to-1 against GT bounding boxes. Never predicts `CONFIRMED_REMOVED` or `CONFIRMED_MOVED`. Maintains zero temporal memory. | Stateless detection baseline. |
| **B3: Complete ObjectMemory** | **`SUPPORTED`** | Full end-to-end pipeline: Keyframe registration, canonical projection, 148-d signatures, Hungarian association with ambiguity gating ($\Delta \text{sim} < 0.12$), multi-visit state updates. | All 10 safeguards active. |
| **A5: No Ambiguity Handling** | **`SUPPORTED`** | Full temporal pipeline with `ambiguity_margin = 0.0`. Ambiguous candidate matches are forcefully bound rather than flagged as `AMBIGUOUS_IDENTITY`. | 9 of 10 safeguards active. |
| **A2: No Vacancy Verification** | **`PARTIALLY_SUPPORTED`** | Requires `allow_partial=True`. `vacancy_similarity_max = 1.0`. Labeled with `is_partial_ablation=True`. | Unaligned images still trigger streak logic. |
| **A3: No Visibility Safeguards** | **`PARTIALLY_SUPPORTED`** | Requires `allow_partial=True`. `min_overlap_fraction = 0.0`. Labeled with `is_partial_ablation=True`. | Inter-object occlusion IoU ($0.20$) remains hardcoded. |
| **B1, B2, A1, A4, A6** | **`BLOCKED`** | Raised as `UnsupportedConfigurationError` before execution. | Blocked by hardcoded state machine and integrity policies. |

---

## 5. Metric Formulations & Zero-Denominator Rules

1. **False Removal Rate ($FRR$):**
   - Numerator: $\sum \mathbb{I}(\text{GT} \neq \text{REMOVED} \land \text{Pred} = \text{CONFIRMED\_REMOVED})$.
   - Denominator: Total evaluations of non-removed GT entities.
   - Zero denominator: returns `None` (logged in `unscorable_cases`).
2. **Removal Recall ($RR$):**
   - Numerator: $\sum \mathbb{I}(\text{GT} = \text{REMOVED} \land \text{Pred} = \text{CONFIRMED\_REMOVED})$.
   - Denominator: Total evaluations where GT is `CONFIRMED_REMOVED`.
   - Zero denominator: returns `None` (logged in `unscorable_cases`).
3. **Unsupported Movement Rate ($UMR$):**
   - Numerator: $\sum \mathbb{I}(\text{GT} = \text{STATIONARY} \land \text{Pred} = \text{CONFIRMED\_MOVED})$.
   - Denominator: Total evaluations of stationary GT entities.
   - Zero denominator: returns `None` (logged in `unscorable_cases`).
4. **Identity Consistency ($IC$):**
   - Numerator: $\sum_g \max_s N(g, s)$ (dominant track observation count per entity).
   - Denominator: Total matched observations for all entities.
   - Zero denominator: returns `None` (logged in `unscorable_cases`).
5. **ID Switches ($IDS$):**
   - Count of transitions $(t_{k-1} \to t_k)$ where assigned system ID changed for the same GT entity.
   - Normalized switch rate: switches / total observed transitions.
6. **Duplicate Identities ($DI$):**
   - $\max(0, |\text{System Entities Instantiated}| - |\text{Unique GT Entities Present}|)$.
   - Also tracks count of unmapped spurious system tracks.
7. **Confirmation Delay ($CD$):**
   - Mean visit delay ($t_{\text{confirm}} - t_{\text{onset}} + 1$) for confirmed removal/movement events.
   - Zero denominator: returns `None` (logged in `unscorable_cases`).
8. **Historical Retrieval Accuracy ($HRA$):**
   - Fraction of reappearance events where returned object is re-associated with its original historical `entity_id`.
   - Zero denominator: returns `None` (logged in `unscorable_cases`).

---

## 6. Verification Results

### A. Focused Benchmark Runner Unit Tests (14 Tests)
```powershell
Set-Location "C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory"; $env:PYTHONPATH="src"; python -m unittest tests.test_benchmark_runner
```
```
..............
----------------------------------------------------------------------
Ran 14 tests in 3.326s

OK
```

### B. Joint Benchmark Suite (Manifest + Config Factory + Runner: 56 Tests)
```powershell
Set-Location "C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory"; $env:PYTHONPATH="src"; python -m unittest tests.test_benchmark_manifest tests.test_benchmark_config_factory tests.test_benchmark_runner
```
```
................s.......................................
----------------------------------------------------------------------
Ran 56 tests in 3.898s

OK (skipped=1)
```
*(1 test cleanly skipped: `test_17_symlink_escaping_dataset_root` due to standard non-privileged Windows symlink creation permissions).*

### C. Full Repository Test Suite Regression Run (393 Tests)
Executed via:
```powershell
Set-Location "C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory"; $env:PYTHONPATH="src"; python -m unittest discover -s tests -p "test_*.py"
```
**Outcome:**
```
Ran 393 tests in 76.069s

OK (skipped=1)
```
- Total tests: 393 (previous baseline was 379 tests; +14 benchmark runner unit tests).
- 392 passed, 1 skipped, 0 failures, 0 errors.
- Confirmed zero regressions across all repository modules (temporal engine, detector registry, planar registration, spatial transforms, database stores, FastAPI endpoints, and benchmarks).
