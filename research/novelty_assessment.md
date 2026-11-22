# ObjectMemory Research — Novelty Assessment & Implementation Evidence (Part D)

**Document Version:** 2.0.0 (Source-Verified & Code-Audited)  
**Date:** November 2026  
**Auditor:** Antigravity Autonomous Research & Verification Suite  
**Project:** ObjectMemory (Long-Term Visual Memory for Persistent Environmental Changes)  
**Target File:** `research/novelty_assessment.md`

---

## 1. Source-Backed Correction Log

| Item / Claim | Initial Description | Source Verification Finding | Verdict / Status |
|---|---|---|---|
| **LT-Mem Mechanism** | Described as 3D surfel submaps with heuristic session-absence counters | **Contradicted by source.** LT-Mem (*"LT-Mem: Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding"*, arXiv:2608.19059, Aug 2026) uses instance-level 3D perception with Tri-Memory (Live, Delta, Meta) and volatility-aware policies (*overwrite*, *hold*, *multi-hypothesis*). | **Contradicted by source** (Corrected) |
| **Remember with Confidence Mechanism** | Described as 3D bounding ellipsoids with LiDAR SLAM and conformal ray tracing | **Contradicted by source.** Remember with Confidence (arXiv:2606.08277, June 2026, MIT SPARK) addresses **semantic uncertainty in Vision-Language Models (VLMs)**, measuring caption scatter across views and fusing captions with an LLM under probabilistic guarantees. | **Contradicted by source** (Corrected) |
| **Two-Gate Policy Novelty** | Claimed as potentially distinctive/novel algorithmic contribution | **Established in prior art.** The two principles (verifying geometric registration before updating occupancy, and testing photometric vacancy against a baseline) are established in 3D mapping (*Ambruș et al., IROS 2014*) and planar change detection (*Taneja et al., 2011/2015*). ObjectMemory's contribution is a **domain-specific systems integration** for lightweight monocular workbench auditing. | **Established** in principle; **Distinctive** in systems architecture |
| **6-Criterion Algebraic Gate** | Six explicit tests claimed in code | **Verified.** Exact source lines identified in [`src/objectmemory/registration/evaluator.py`](src/objectmemory/registration/evaluator.py#L85-L251). | **Verified** |
| **Dynamic Uncertainty Inflation** | $T_{\text{effective}} = T_{\text{base}} + \Delta_U$ claimed in code | **Verified.** Exact source lines identified in [`src/objectmemory/temporal/engine.py`](src/objectmemory/temporal/engine.py#L98-L160). | **Verified** |

---

## 2. Code-Level Implementation Evidence of Vacancy Verification Mechanisms

The table below documents the exact implementation of every component claimed in `ObjectMemory`:

| Mechanism | Source File & Class | Exact Lines | Actual Formula / Logic | Threshold Source | Relevant Tests | Test Validation Status |
|---|---|---|---|---|---|---|
| **`is_spatial_reliable`** | [`temporal/engine.py`](src/objectmemory/temporal/engine.py)<br>`TemporalEngine` | Lines 78–96 (`_is_registration_reliable`), 176–225 (`_is_spatial_comparison_reliable`), 488, 704, 716, 725, 781, 808 | Returns `True` iff `reg.success` and `reg.homography is not None` and `inlier_ratio >= min_inlier_ratio` and `reproj_stats.mean <= max_reproj_error` and spatial overlap with canon bbox is valid. | `TemporalConfig.min_registration_inlier_ratio` (0.20), `max_reprojection_error` (4.0px) | `test_boundary_and_registration.py`<br>`test_temporal_failure_modes.py` | **Verified**: Validates that unreliable registration suppresses spatial updates and preserves `RawDetection`. |
| **Homography Condition Number** | [`registration/evaluator.py`](src/objectmemory/registration/evaluator.py)<br>`AlignmentQualityEvaluator` | Lines 121–139 in `evaluate_homography` | `det = abs(np.linalg.det(H))` (must be $\ge 10^{-8}$); `cond = np.linalg.cond(H)` (must be $\le 10^7$). Rejects if $\text{cond} > 10^7$. | `evaluator.py` line 132 (constant $10^7$) | `test_registration.py` (L231, L248: singular & ill-conditioned homographies) | **Verified**: Rejects ill-conditioned matrices as `SINGULAR_HOMOGRAPHY`. |
| **Minimum Inlier Count** | [`registration/evaluator.py`](src/objectmemory/registration/evaluator.py)<br>`AlignmentQualityEvaluator` | Lines 142–150 in `evaluate_homography` | `num_inliers < self.config.min_inliers` triggers `INSUFFICIENT_INLIERS`. | `AlignmentConfig.min_inliers = 12` ([`models.py:59`](src/objectmemory/registration/models.py#L59)) | `test_registration.py` (L200) | **Verified**: Rejects registration when inlier count $< 12$. |
| **Minimum Inlier Ratio** | [`registration/evaluator.py`](src/objectmemory/registration/evaluator.py)<br>`AlignmentQualityEvaluator` | Lines 152–160 in `evaluate_homography` | `inlier_ratio = num_inliers / max(num_tentative, 1)`; must be $\ge 0.20$. | `AlignmentConfig.min_inlier_ratio = 0.20` ([`models.py:60`](src/objectmemory/registration/models.py#L60)) | `test_registration.py` (L217) | **Verified**: Rejects low-consensus matches. |
| **Maximum Reprojection Error** | [`registration/evaluator.py`](src/objectmemory/registration/evaluator.py)<br>`AlignmentQualityEvaluator` | Lines 162–171 in `evaluate_homography` | $\text{mean\_err} = \frac{1}{N} \sum \|H p_{\text{query}} - p_{\text{ref}}\|$; must be $\le 4.0$px. | `AlignmentConfig.max_mean_reprojection_error = 4.0` ([`models.py:61`](src/objectmemory/registration/models.py#L61)) | `test_registration.py` (L265) | **Verified**: Rejects alignments with mean error $> 4.0$px. |
| **Quadrilateral Convexity & Area** | [`registration/evaluator.py`](src/objectmemory/registration/evaluator.py)<br>`AlignmentQualityEvaluator` | Lines 62–84 (`is_convex_quadrilateral`), 201–225 (`evaluate_homography`) | Checks cross-products $v_i \times v_{i+1} > 10^{-5}$ for 4 warped image corners; computes shoelace area $\text{scale\_ratio} = \frac{\text{area}_{\text{warped}}}{\text{area}_{\text{orig}}} \in [0.15, 6.0]$. | `AlignmentConfig.min_scale_ratio = 0.15`, `max_scale_ratio = 6.0` ([`models.py:63-64`](src/objectmemory/registration/models.py#L63)) | `test_registration.py` (L285: inverted corners) | **Verified**: Rejects bow-tie, folded, or inverted quadrilaterals. |
| **Dynamic Uncertainty Inflation** | [`temporal/engine.py`](src/objectmemory/temporal/engine.py)<br>`TemporalEngine` | Lines 98–160 (`compute_effective_movement_threshold`) | $T_{\text{effective}} = T_{\text{base}} + \Delta_U$, where $\Delta_U = \min(45, u_{\text{reproj}} + u_{\text{inlier}} + u_{\text{cov}} + u_{\text{scale}})$. | `TemporalConfig.movement_displacement_threshold = 35.0`px | `test_keyframe_registration.py` (L400) | **Verified**: Inflates threshold dynamically when inliers drop or reprojection error increases. |
| **Canonical Patch Extraction** | [`temporal/engine.py`](src/objectmemory/temporal/engine.py)<br>`TemporalEngine` | Line 1049 in `_evaluate_observational_evidence` | `patch_sig = self.signature_extractor.extract(aligned_img, canon_box)` | Canonical bounding box coordinates from entity record | `test_temporal_engine.py`<br>`test_pairwise_baseline.py` | **Verified**: Warps query frame to canonical space and crops entity bounding box patch. |
| **Appearance Cosine Similarity** | [`temporal/engine.py`](src/objectmemory/temporal/engine.py)<br>`TemporalEngine` | Lines 1053–1057 in `_evaluate_observational_evidence` | `patch_sim = ent_sig.cosine_similarity(patch_sig)`; `is_vacant = (patch_sim < self.config.vacancy_similarity_max)`. | `TemporalConfig.vacancy_similarity_max = 0.35` | `test_temporal_engine.py` (L181)<br>`test_temporal_reasoning_reliability.py` (L185) | **Verified**: Confirms vacancy when cosine similarity drops below 0.35. |
| **Foreground Occlusion / FoV Test** | [`temporal/engine.py`](src/objectmemory/temporal/engine.py)<br>`TemporalEngine` | Lines 1012–1042 in `_evaluate_observational_evidence` | Checks FoV overlap $< 0.70$ (returns `OUT_OF_FRAME`); checks $\text{IoU}(\text{canon\_box}, \text{det\_box}) > 0.20$ for any active detection (returns `PARTIALLY_OCCLUDED`). Freezes absence streak! | `TemporalConfig.min_overlap_fraction = 0.70`, $\text{IoU} = 0.20$ | `test_temporal_engine.py` (L229, L251)<br>`test_temporal_failure_modes.py` (L156) | **Verified**: Prevents false removal progression when an object is occluded or clipped by camera boundary. |
| **Consecutive Vacancy Confirmation** | [`temporal/engine.py`](src/objectmemory/temporal/engine.py)<br>`TemporalEngine` | Lines 1061–1118 in `_evaluate_observational_evidence` | If vacant: `vacant_streak += 1`. Streak 1 $\rightarrow$ `PENDING_REMOVAL`; Streak 2 $\rightarrow$ `CONFIRMED_REMOVED`. If patch not vacant: streak 1 preserves state (`CONFIRMED_PRESENT`); streak 2 $\rightarrow$ `PENDING_REMOVAL`; streak 3 $\rightarrow$ `CONFIRMED_REMOVED`. | `vacancy_visits_for_removal = 2`, `unobserved_visits_for_removal = 3` | `test_temporal_engine.py` (L201, L203)<br>`test_phase10a_corrections.py` | **Verified**: Validates that removal confirmation strictly requires 2 verified vacant visits. |

---

## 3. Prior Art Investigation for the Two-Gate Policy

### 3.1 Historical Context
The policy of **conditioning absence updates on observation validity** and **verifying vacancy via appearance** is not a novel mathematical primitive invented by this project. Rather, it represents the synthesis of established techniques across three literatures:

1. **3D Volumetric Mapping & Space Carving (Ambruș et al., IROS 2014, *Meta-rooms*):**
   In robotic 3D mapping, an object cannot be removed simply because it was not detected. A mobile robot must cast sensor rays through the object's previous 3D bounding volume to verify that the return hits background surface rather than foreground, confirming vacancy.
2. **Photometric & Geometric Change Detection (Taneja et al., CVPR 2011, TPAMI 2015):**
   In visual change detection, a candidate removal is verified by projecting the prior object patch onto the registered query image and computing radiometric or color consistency. If the warped patch matches the empty background model rather than the object template, removal is confirmed.
3. **Object Permanence in Robotics (Mason et al., 2016):**
   Robotic manipulators maintain persistent object beliefs when objects pass behind occluders by suppressing track deletion when another bounding box or obstacle overlaps the expected region.

### 3.2 What ObjectMemory Does Differently (Systems Distinctiveness)
While the underlying primitives (homography condition numbers, patch cosine similarity, IoU occlusion tests) are **established**, ObjectMemory implements a **distinctive systems architecture for planar visual monitoring**:
- **Asymmetric Decoupling:** When registration fails, the system does not drop detections or enter a "lost tracking" fault state. It persists raw detections into `raw_detections` for visualization and forensic auditing, while strictly isolating temporal state updates and freezing absence counters.
- **Relational Forensic Auditability:** Every transition is written to SQLite with an exact algebraic certificate ($d_{\text{px}}$ vs. $T_{\text{effective}}$, condition number, inlier ratio, patch cosine similarity), allowing complete post-hoc forensic inspection without re-running inference.

---

## 4. Rigorous Novelty Verdict for the 6 Candidate Areas

| # | Candidate Contribution | Literature Precedent | ObjectMemory Implementation | Honest Academic Verdict |
|---|---|---|---|---|
| **1** | **Raw Evidence Preservation** | Factor graphs, sensor logging in robotics | Persists `RawDetection` independent of homography; decouples display from memory updates | **Established** in principle; **Potentially distinctive** as a lightweight systems architecture |
| **2** | **Registration-Conditioned Absence** | View-frustum checks in SLAM; visibility gating | 6-criterion algebraic gate (`is_spatial_reliable`) freezes absence counter | **Established** in principle; **Engineering implementation of fault-tolerant estimation** |
| **3** | **Evidence-Aware State Transitions** | Multi-hypothesis tracking, finite-state track management | FSM with dynamic uncertainty threshold ($T_{\text{base}} + \Delta_U$) and return-to-baseline check | **Established** in concept; **Domain-specific formulation for workbenches** |
| **4** | **Occlusion vs. Removal Vacancy** | 3D space carving (Ambruș 2014); patch differencing (Taneja 2015) | 2-gate policy: IoU/FoV guard + 2-visit patch cosine similarity $< 0.35$ | **Established** in computer vision; **Effective practical integration** |
| **5** | **Historical Identity Recovery** | Person/object Re-ID, appearance matching | 148-d RGB+HSV color histogram matching against removed entities | **Not novel** (Classical color histogram matching) |
| **6** | **Explainable Event Histories** | Data provenance, flight recorders, audit logging | Forensic certificate stored in `ChangeEvent.evidence_summary` | **Established** in software systems; **Distinctive feature for visual memory systems** |

---

## 5. Summary Conclusion

ObjectMemory does **not** introduce a fundamentally new machine learning algorithm or mathematical theorem. Rather, its research value lies in its **robust, fault-tolerant systems design**: bringing together classical geometric safeguards, active photometric vacancy verification, and decoupled raw evidence logging into an open, auditable, CPU-runnable visual memory engine.
