# ObjectMemory: Phase 2 Configuration & Threshold Audit
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Audit Date:** November 13, 2026  
**Auditor:** Automated Research Alignment & Codebase Integrity Suite (Antigravity)  
**Status:** Frozen Runtime Configuration Reference

---

## Executive Summary

During Phase 1, inspection revealed discrepancies between numbers reported in historical narrative audits and the concrete dataclass defaults defined in the active codebase. Specifically:
1. `ambiguity_margin` was reported in Phase 1 summaries as `0.08`, whereas `TemporalEngineConfig.ambiguity_margin` is `0.12`.
2. `vacancy_similarity_max` was reported in Phase 1 summaries as `0.45`, whereas `TemporalEngineConfig.vacancy_similarity_max` is `0.40`.
3. `min_overlap_fraction` was discussed ambiguously between global frame registration (`0.15`) and entity-level field-of-view bounding box coverage (`0.70`).

This audit resolves every discrepancy by tracing the parameters directly to source code line numbers, detailing dataclass defaults, engine overrides, test overrides, and the effective runtime defaults executed in normal application runs.

---

## 1. Systematic Parameter Audit Table

| Parameter | Source File & Line | Dataclass Default | Engine-Level Override | Test-Specific Overrides | Effective Runtime Default | Status & Resolution |
|---|---|---|---|---|---|---|
| **`ambiguity_margin`** | [src/objectmemory/temporal/models.py#L39](src/objectmemory/temporal/models.py#L39) | `0.12` | None | `0.10` in `test_temporal_engine.py:60`<br>`0.20` in `test_phase14c2_evaluator.py:136`<br>`0.05` in `test_temporal_failure_modes.py` | **`0.12`** | **RESOLVED:** `0.08` was an erroneous value in narrative audit notes; `0.12` is the active dataclass default. |
| **`signature_similarity_threshold`** | [src/objectmemory/temporal/models.py#L38](src/objectmemory/temporal/models.py#L38) | `0.65` | None | `0.60` in `test_temporal_engine.py:60`<br>`0.50` in `test_pairwise_baseline.py:80` | **`0.65`** | **FROZEN:** Evaluated via 148-d cosine similarity. Pairs with $\text{sim} < 0.65$ are rejected. |
| **`cross_class_ambiguity_margin`** | [src/objectmemory/temporal/models.py#L83](src/objectmemory/temporal/models.py#L83) | `0.15` | None | None | **`0.15`** | **FROZEN:** Higher threshold required to prevent cross-category identity swaps. |
| **`vacancy_similarity_max`** | [src/objectmemory/temporal/models.py#L51](src/objectmemory/temporal/models.py#L51) | `0.40` | None | `0.40` in `test_phase14c2_evaluator.py:161`<br>`0.35` in `PairwiseMatchConfig` | **`0.40`** | **RESOLVED:** `0.45` was an erroneous conflation with detector `nms_threshold=0.45`; `0.40` is the active default. |
| **`vacancy_visits_for_removal`** | [src/objectmemory/temporal/models.py#L57](src/objectmemory/temporal/models.py#L57) | `2` | None | `2` in all regression suites | **`2`** | **FROZEN:** Requires 2 consecutive verified vacant visits before transitioning `PENDING_REMOVAL` $\to$ `CONFIRMED_REMOVED`. |
| **`vacancy_visits_for_pending`** | [src/objectmemory/temporal/models.py#L56](src/objectmemory/temporal/models.py#L56) | `1` | None | None | **`1`** | **FROZEN:** Immediate transition to `PENDING_REMOVAL` on the first unobserved visit with verified vacant patch. |
| **`unobserved_visits_for_removal`** | [src/objectmemory/temporal/models.py#L59](src/objectmemory/temporal/models.py#L59) | `3` | None | None | **`3`** | **FROZEN:** When patch appearance is unverified/object-like, requires 3 consecutive unobserved visits. |
| **Registration Inlier Ratio** | [src/objectmemory/registration/models.py#L60](src/objectmemory/registration/models.py#L60)<br>[src/objectmemory/temporal/models.py#L45](src/objectmemory/temporal/models.py#L45) | `0.20` | None | None | **`0.20`** ($20\%$) | **FROZEN:** Minimum ratio of RANSAC inliers to tentative matches. Both registration evaluator and temporal engine enforce $0.20$. |
| **Registration Inlier Count** | [src/objectmemory/registration/models.py#L59](src/objectmemory/registration/models.py#L59) | `12` | None | None | **`12`** | **FROZEN:** Absolute minimum inlier keypoint count. |
| **Registration Reprojection Error** | [src/objectmemory/registration/models.py#L61](src/objectmemory/registration/models.py#L61)<br>[src/objectmemory/temporal/models.py#L47](src/objectmemory/temporal/models.py#L47) | `4.0` px | None | None | **`4.0` px** | **FROZEN:** Mean reprojection error across inliers must be $\le 4.0$ px. RANSAC internal threshold is $3.0$ px. |
| **Global Scene Registration Overlap** | [src/objectmemory/registration/models.py#L62](src/objectmemory/registration/models.py#L62) | `0.15` ($15\%$) | None | None | **`0.15`** | **FROZEN:** Minimum fraction of reference image area covered by projected query quadrilateral for valid registration. |
| **Entity Bounding Box Overlap** | [src/objectmemory/temporal/models.py#L46](src/objectmemory/temporal/models.py#L46) | `0.70` ($70\%$) | None | None | **`0.70`** | **FROZEN:** Entity's canonical bbox must have $\ge 70\%$ area inside the query overlap mask; $< 70\%$ flags `OUT_OF_FRAME`. |
| **Movement Displacement Threshold** | [src/objectmemory/temporal/models.py#L62](src/objectmemory/temporal/models.py#L62) | `35.0` px | Dynamic $\Delta U$ added in `engine.py:550` | None | **$35.0\text{ px} + \Delta U$** | **FROZEN:** Base physical movement threshold in canonical coordinates. Dynamic margin $\Delta U \ge 0$ added from registration error. |
| **Clipped Movement Threshold** | [src/objectmemory/temporal/models.py#L67](src/objectmemory/temporal/models.py#L67) | `100.0` px | None | None | **`100.0` px** | **FROZEN:** Displacements $< 100$ px for boundary-truncated boxes are treated as framing artifacts, not physical relocations. |
| **Max Drift Step Displacement** | [src/objectmemory/temporal/models.py#L70](src/objectmemory/temporal/models.py#L70) | `20.0` px | None | None | **`20.0` px** | **FROZEN:** Per-step displacement limit for gradual coordinate adjustment without triggering `CONFIRMED_MOVED`. |
| **Drift IoU Threshold** | [src/objectmemory/temporal/models.py#L73](src/objectmemory/temporal/models.py#L73) | `0.35` | None | None | **`0.35`** | **FROZEN:** Minimum overlap with initial canonical bbox required to consider accumulated shift as coordinate drift. |
| **Border Margin** | [src/objectmemory/temporal/models.py#L48](src/objectmemory/temporal/models.py#L48) | `12.0` px | None | None | **`12.0` px** | **FROZEN:** Proximity to image edge ($< 12$ px) that triggers boundary-clipping branch. |
| **Detector NMS IoU Threshold** | [src/objectmemory/detection/detector.py#L41](src/objectmemory/detection/detector.py#L41)<br>[src/objectmemory/detection/models.py#L32](src/objectmemory/detection/models.py#L32) | `0.45` | None | None | **`0.45`** | **FROZEN:** Non-Maximum Suppression threshold inside OpenCV DNN inference. |
| **Detector Confidence Threshold** | [src/objectmemory/detection/detector.py#L40](src/objectmemory/detection/detector.py#L40) | `0.25` | Set to `0.35` in `benchmarks/real_evaluator.py:186` | `0.25` in unit tests | **`0.25`** (engine)<br>**`0.35`** (evaluator) | **FROZEN:** Engine accepts $\ge 0.25$. Evaluator filters to $\ge 0.35$ for high-precision real benchmarking. |

---

## 2. Root Cause Analysis of Prior Discrepancies

### A. Ambiguity Margin Discrepancy (0.08 vs 0.12)
- **Origin:** In the earliest exploratory proposal (Phase 1 design notes), an ambiguity margin of `0.08` was theorized. When the multi-visit temporal associator was formally implemented and hardened in Phase 5 (`src/objectmemory/temporal/models.py`), the value was set to `0.12` to provide a wider safety cushion against near-identical desktop items (e.g. adjacent screws or duplicate pens).
- **Resolution:** The active runtime codebase uses **`0.12`**. The value `0.08` is deprecated and deleted from documentation.

### B. Vacancy Similarity Discrepancy (0.45 vs 0.40)
- **Origin:** The number `0.45` appears in multiple places in the codebase:
  1. `nms_threshold = 0.45` in `detector.py:41` (OpenCV Non-Maximum Suppression).
  2. `iou_threshold = 0.45` in `benchmarks/real_evaluator.py:186` (P/R evaluation matching).
  3. `confidence = 0.45` in `docs/phase5_evaluation_audit.md` (background table detection).
  In Phase 1 reporting, this number was accidentally cross-transcribed as the vacancy threshold.
- **Resolution:** The active runtime parameter in `TemporalEngineConfig.vacancy_similarity_max` is strictly **`0.40`**.

### C. Overlap Fraction Duality (0.15 vs 0.70)
- **Clarification:** These two thresholds govern completely different physical scopes:
  1. **Global Scene Overlap (`AlignmentConfig.min_overlap_fraction = 0.15`):** Governs whether the homography between the current query image and the reference frame covers at least $15\%$ of the reference image area. If $< 15\%$, registration returns `INSUFFICIENT_OVERLAP`.
  2. **Entity-Specific Field-of-View Overlap (`TemporalEngineConfig.min_overlap_fraction = 0.70`):** Governs whether a specific object's canonical bounding box is at least $70\%$ inside the query image's valid registration mask. If $< 70\%$, the object is classified as `OUT_OF_FRAME`, preventing unsupported absence decisions.
- **Resolution:** Both values are correct and active in their respective modules.

---

## 3. Configuration Management Policy for Phase 2

1. **Production Immutability:** No production source code file in `src/objectmemory/` will be altered to hardcode benchmark overrides.
2. **Explicit Harness Injection:** All experimental baselines (B0–B3) and ablations (A1–A6) must instantiate `TemporalEngineConfig` explicitly and pass it via dependency injection to `TemporalMemoryEngine(config=cfg)`.
3. **No Retuning on Final Evaluation Splits:** Thresholds audited here are frozen. Parameter sensitivity studies must be conducted strictly on designated development sequences (e.g. `Physical-Dev-01`), never on evaluation test splits.
