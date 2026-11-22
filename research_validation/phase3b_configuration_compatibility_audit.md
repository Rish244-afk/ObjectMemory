# ObjectMemory: Phase 3B Configuration Compatibility Audit
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Audit Date:** November 14, 2026  
**Auditor:** Automated Benchmark Integrity Suite (Antigravity)  
**Status:** Pre-Implementation Compatibility Audit (Stage 1 Complete)

---

## Executive Summary

Before implementing the benchmark configuration factory (`config_factory.py`), a comprehensive read-only audit of the entire codebase was conducted to determine:
1. Whether all call sites handle the newly hardened manifest parser return types.
2. Whether manifest vocabulary normalizations strictly match domain enums.
3. Which baselines (B0–B3) and ablations (A1–A6) can be realized natively using existing configuration dataclasses versus which are **BLOCKED** by hardcoded engine logic.
4. What hidden architectural couplings exist between registration, visibility, vacancy verification, and identity association.

### Key Audit Finding
- **B3 (Proposed System) and A5 (No Ambiguity Gating)** are **100% natively supported** through existing dataclass settings.
- **B0 (Memoryless Detection)** cannot be executed through `TemporalMemoryEngine`, but is natively executable as a standalone detector evaluation pipeline.
- **A1, A2, A3, A4, A6, B1, and B2** cannot be fully or cleanly realized solely by adjusting `TemporalEngineConfig` values because key safeguards (e.g., direct removal state transition prohibition, `registration.success` checks, 0.20 occlusion IoU threshold, inclusion of `CONFIRMED_REMOVED` entities in association) are **hardcoded in `src/objectmemory/temporal/engine.py` and `persistence/states.py`**.

In accordance with strict research integrity instructions:
> *"Do not describe a baseline as implemented if it cannot be reproduced using the current code... If a baseline or ablation cannot be supported by the current engine, report it as BLOCKED. Do not fake it by changing a configuration label."*

---

## 1. Manifest API & Enum Compatibility Verification

### A. Caller Compatibility with `ObjectAnnotationEntry.from_dict()`
- **Audit Target:** `ObjectAnnotationEntry.from_dict(data, scope) -> Tuple[Optional[ObjectAnnotationEntry], List[str]]`.
- **Finding:** A repository-wide code search across all `.py` files confirmed that `ObjectAnnotationEntry.from_dict` is called **only** in:
  1. `src/objectmemory/benchmarks/manifest.py:648` (within `ManifestLoader.parse_dict`).
  2. `tests/test_benchmark_manifest.py` (across 10 dedicated unit tests).
- **Other `from_dict` implementations:**
  - `PhysicalWorkbenchAnnotation.from_dict()` in `physical_workbench.py` (unaltered, returns `PhysicalWorkbenchAnnotation`).
  - `RealVisitAnnotation.from_dict()` in `real_dataset.py` (unaltered, returns `RealVisitAnnotation`).
  - `GroundTruthObject.from_dict()` in `synthetic_dataset.py` (unaltered, returns `GroundTruthObject`).
- **Verdict:** **NO CALLER REGRESSIONS.** All invocations correctly unpack `(entry, errors)`.

### B. Enum Vocabulary Alignment
- **`VisibilityStatus`:**
  - Canonical Enum (`src/objectmemory/domain/states.py:30-41`): `CLEARLY_VISIBLE`, `PARTIALLY_OCCLUDED`, `HEAVILY_OCCLUDED`, `OUT_OF_FRAME`, `UNOBSERVED`, `UNCERTAIN`.
  - Manifest Vocabulary (`manifest.py:37`): Exactly identical set. Legacy aliases (`VISIBLE`, `FULLY_OCCLUDED`, `ABSENT`) map deterministically into the canonical set.
- **`EpistemicState`:**
  - Canonical Enum (`src/objectmemory/domain/states.py:13-28`): `CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `CONFIRMED_REMOVED`, `CONFIRMED_ADDED`, `PENDING_REMOVAL`, `INSUFFICIENT_EVIDENCE`, `AMBIGUOUS_IDENTITY`.
  - Manifest Vocabulary (`manifest.py:46`): Exactly identical set. Legacy aliases (`MOVED`, `STATIONARY`, `REMOVED`, `REAPPEARED`, `ADDED`) map deterministically into the canonical set.
- **Verdict:** **100% CANONICAL ENUM ALIGNMENT.**

---

## 2. Baseline & Ablation Feasibility Audit

### A. Baseline Feasibility Analysis

| Baseline | Proposed Definition | Implementation Mechanism in Current Code | Feasibility Status | Detailed Technical Reason |
|---|---|---|---|---|
| **B0** | Raw per-frame YOLO detections, memoryless | Standalone detection evaluation against ground-truth bboxes per frame. | **SUPPORTED (as Pipeline Mode)** | Cannot run through `TemporalMemoryEngine` (which inherently tracks entities). Must be evaluated as a detector-only pipeline mode (`pipeline_type="MEMORYLESS"`). |
| **B1** | Conventional Hungarian tracking, immediate removal | Hungarian association with instant drop upon non-detection. | **BLOCKED in Engine** | `VALID_STATE_TRANSITIONS` (`domain/states.py:77`) and `engine.py:1070` strictly forbid direct transitions from `CONFIRMED_PRESENT` to `CONFIRMED_REMOVED`. Even with `vacancy_visits_for_removal = 1`, the first missed visit transitions to `PENDING_REMOVAL`. |
| **B2** | Temporal memory without registration/vacancy safeguards | Multi-visit state updates without registration gating or patch verification. | **BLOCKED in Engine** | `engine.py:983` unconditionally checks `not self._is_registration_reliable(visit.registration) or visit.registration.overlap_mask is None`. If registration fails, absence reasoning is completely halted regardless of config. Occlusion threshold (0.20 IoU) is hardcoded. |
| **B3** | Complete proposed ObjectMemory | Full uncertainty-aware, registration-gated, vacancy-verified memory. | **FULLY SUPPORTED** | Exactly matches default `TemporalEngineConfig()` and production engine behavior. |

---

### B. Ablation Feasibility Analysis

| Ablation | Intended Modification | Config Parameter | Engine Support Status | Technical Reason & Engine Code Reference |
|---|---|---|---|---|
| **A1** | Disable registration-aware absence freezing | `min_registration_inlier_ratio = 0.0` | **BLOCKED for Failed Alignments** | In `engine.py:83`, `_is_registration_reliable()` returns `False` if `registration.success == False`, regardless of inlier ratio. Furthermore, `engine.py:985` halts absence if `overlap_mask is None`. There is no configuration flag to bypass these checks. |
| **A2** | Disable appearance vacancy verification | `vacancy_similarity_max = 1.0` | **PARTIALLY SUPPORTED** | Setting `vacancy_similarity_max = 1.0` forces `patch_sim < 1.0` to evaluate to `is_vacant = True` when patches exist. However, if `aligned_image is None` or patch extraction fails, `is_vacant` remains `False`, taking the 3-visit fallback. |
| **A3** | Disable visibility & FOV safeguards | `min_overlap_fraction = 0.0` | **PARTIALLY SUPPORTED** | Setting `min_overlap_fraction = 0.0` disables the bounding-box field-of-view check (`engine.py:1013`). However, the inter-object occlusion check (`IoU > 0.20` in `engine.py:1031`) is hardcoded and cannot be disabled via config. |
| **A4** | Single-visit removal confirmation | `vacancy_visits_for_removal = 1` | **BLOCKED for First Miss** | In `engine.py:1070`, if an object is in `CONFIRMED_PRESENT`, the first unobserved visit is hardcoded to transition to `PENDING_REMOVAL`. Setting `vacancy_visits_for_removal = 1` allows removal on visit 2 instead of visit 3, but cannot confirm removal on visit 1. |
| **A5** | Disable ambiguity handling | `ambiguity_margin = 0.0` | **FULLY SUPPORTED** | Setting `ambiguity_margin = 0.0` natively disables ambiguity gating in `associator.py:152`. Because candidate affinities are sorted descending, `affs[0] - affs[1] < 0.0` is impossible (except exact mathematical ties). |
| **A6** | Disable historical identity recovery | None | **BLOCKED in Engine** | `engine.py:346` retrieves all entities (`self.store.list_entities()`), including `CONFIRMED_REMOVED` objects, and passes them to the associator. If re-observed, `engine.py:614` automatically restores them to `CONFIRMED_PRESENT`. There is no config flag to filter out removed entities. |

---

## 3. Hidden Dependencies & Architectural Couplings

The audit identified three critical architectural couplings in `src/objectmemory/temporal/engine.py`:

```
               [visit.registration]
                        |
        +---------------+---------------+
        |                               |
[registration.success]        [overlap_mask]
        |                               |
        v                               v
[det.canonical_bbox]          [BoundingBox FOV Check]
        |                               |
        +---------------+               v
                        |      [engine.py:983 Gating]
                        v               |
            [can_use_spatial = False]   v
                        |      [Freezes as UNOBSERVED]
                        v
            [Associator falls back
             to appearance signatures only]
```

1. **Registration Success Couples Spatial Projection and Entity Creation:**
   In `engine.py:313` and `383`, if registration fails, query detections cannot receive canonical bounding boxes (`det.canonical_bbox = None`). Unmatched candidate detections cannot be converted into new persistent entities after visit 0.
2. **Overlap Mask Couples Absence Reasoning to Registration Success:**
   In `engine.py:985`, absence evaluation requires `visit.registration.overlap_mask is not None`. An unaligned query frame has no overlap mask, which halts absence reasoning before FOV, occlusion, or patch vacancy logic can execute.
3. **Vacancy Verification Couples to Aligned Image Generation:**
   In `engine.py:1046`, patch visual signatures are extracted from `visit.registration.aligned_image`. If registration fails, no aligned query image exists, meaning vacancy verification cannot be computed.
4. **State Transition Policy Couplings in Database Layer:**
   In `src/objectmemory/persistence/repository.py:272` and `domain/states.py:77`, `record_state_transition()` asserts `is_valid_transition(current_state, new_state)`. Even if the temporal engine attempted an instant removal (`CONFIRMED_PRESENT -> CONFIRMED_REMOVED`), the database layer would raise a `ValueError`, cleanly rolling back the transaction.

---

## 4. Recommendations for Phase 3B Configuration Factory Design

To maintain strict scientific integrity and avoid creating broken or illusory configuration switches:

1. **Implement `BenchmarkConfigFactory` with Explicit Capability Flags:**
   Every configuration must explicitly declare its operational status:
   - `B3`: `STATUS_SUPPORTED` (Full proposed system).
   - `B0`: `STATUS_SUPPORTED` (Handled via standalone detection pipeline).
   - `A5`: `STATUS_SUPPORTED` (Full native config support).
   - `A2`, `A3`: `STATUS_PARTIALLY_SUPPORTED` (Explicitly documents active vs. unconfigurable aspects).
   - `B1`, `B2`, `A1`, `A4`, `A6`: `STATUS_BLOCKED_IN_CURRENT_ENGINE` (Clearly documented as requiring engine-level parameterization before execution).
2. **Immutable Dataclass Architecture:**
   Use frozen dataclasses (`BenchmarkConfig`) providing:
   - Stable identifier (`config_id`).
   - Immutable copy of `TemporalEngineConfig`.
   - Complete dictionary of all effective parameters for serialization and auditability.
   - Deterministic SHA-256 configuration hash.
3. **No Fake Ablations:**
   The factory must raise `UnsupportedConfigurationError` if a caller attempts to instantiate an unsupported baseline/ablation in a benchmark run, unless an explicit `--allow-partial-ablations` flag is provided with documented caveats.
