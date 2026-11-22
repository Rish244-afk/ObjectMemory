# ObjectMemory Research — Executive Summary, Research Verdict & Strategic Recommendations (Parts G & H)

**Document Version:** 2.0.0 (Source-Verified & Code-Audited)  
**Date:** November 2026  
**Auditor:** Antigravity Autonomous Research & Verification Suite  
**Project:** ObjectMemory (Long-Term Visual Memory for Persistent Environmental Changes)  
**Target File:** `research/research_summary.md`

---

## 1. Source-Backed Correction Log

| Item / Claim | Initial Status | Verification Finding | Status |
|---|---|---|---|
| **LT-Mem (arXiv:2608.19059)** | Described as 3D Gaussian/surfel SLAM with heuristic absence counter | **Contradicted by source.** LT-Mem uses instance-level 3D perception with Tri-Memory (Live, Delta, Meta) and a volatility-aware memory policy (*overwrite*, *hold*, *multi-hypothesis*) backed by multi-session SLAM. | **Contradicted by source** (Corrected) |
| **Remember with Confidence (arXiv:2606.08277)** | Described as 3D bounding ellipsoids with LiDAR SLAM and conformal ray tracing | **Contradicted by source.** The paper focuses on **semantic uncertainty in Vision-Language Models (VLMs)**, cross-view caption scatter, and LLM-based multi-view caption fusion with probabilistic guarantees. | **Contradicted by source** (Corrected) |
| **Claimed Novelty of Two-Gate Policy** | Claimed as an unprecedented novelty | **Established in prior art.** The principles of verifying registration before updating spatial state and testing photometric vacancy against a baseline patch have existed in 3D mapping (*Ambruș et al., 2014*) and planar change detection (*Taneja et al., 2011/2015*). The value of ObjectMemory is as a cohesive, fault-tolerant **systems architecture**. | **Established** in principle; **Distinctive** as a systems design |
| **Unit Test Exit Code Discrepancy** | Initial log showed exit code 1; subsequent report claimed 337 passed | **Verified.** The command with exit code 1 was run from the root workspace `C:\Users\Rishh\.gemini\antigravity` where no `tests/` directory existed (`ImportError: Start directory is not importable: 'tests'`). Running the command from `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory` succeeds with exit code 0 (`Ran 337 tests in 126.079s, OK`). | **Verified** |

---

## 2. Part G: Comprehensive Executive Summary

### 2.1 System Overview
`ObjectMemory` is a lightweight, edge-runnable long-term visual memory engine designed for monitoring planar environments (e.g., desktops, assembly workbenches, inspection surfaces) across discrete temporal visits separated by minutes, hours, or days.

Rather than treating incoming frames in isolation or assuming continuous video, ObjectMemory maintains persistent entity lifecycles, baseline coordinates, and auditable event histories by integrating:
1. **Raw Sensor Evidence Decoupling:** Ingesting and storing every detection in [`raw_detections`](src/objectmemory/persistence/raw_detections.py) independently of geometric registration success.
2. **Strict Homography Quality Gating:** Evaluating registration quality across six algebraic criteria (finiteness, condition number $< 10^7$, inlier count $\ge 12$, inlier ratio $\ge 0.20$, reprojection error $\le 4.0$px, quadrilateral convexity, and area scale bounds) before granting authority to update spatial memory.
3. **Bipartite Entity Association:** Matching observations using Kuhn-Munkres assignment over 148-dimensional color/spatial visual signatures with identity ambiguity margin flagging ($\Delta \text{aff} < 0.12$).
4. **Deterministic Two-Gate Vacancy Verification:** Freezing absence counters when registration fails, field-of-view is clipped, or another object occludes the region; and confirming removal only when canonical patch appearance cosine similarity drops below 0.35 across consecutive visits.
5. **Auditable Forensic Event Logging:** Persisting structured mathematical certificates with every state transition, detailing exact pixel displacements, condition numbers, inlier ratios, and patch similarities.

---

### 2.2 SOTA Landscape Summary (Late 2026)

| Paradigm / Paper | Representation | Strengths | Limitations Relative to ObjectMemory |
|---|---|---|---|
| **LT-Mem (arXiv:2608.19059, Aug 2026)** | Instance-level 3D perception with Tri-Memory (Live, Delta, Meta) | Multi-session SLAM for mobile robots; volatility-aware policies (*overwrite*, *hold*, *multi-hypothesis*) | Requires multi-session SLAM infrastructure; does not formulate 2D planar homography quality gates or patch vacancy verification for monocular cameras. |
| **VSCD (ICML 2026)** | Non-parametric query-reference video cross-attention volume | Detects 2D pixel-level changes across unaligned video streams without rigid alignment | Heavy video transformer; outputs black-box pixel heatmaps without persistent entity identities, state machines, or forensic audit records. |
| **Remember with Confidence (arXiv:2606.08277, June 2026)** | Spatio-temporal semantic memory with VLM captions & uncertainty scores | Quantifies semantic uncertainty from cross-view caption scatter; probabilistic guarantees for view selection | Focuses on high-level VLM caption ambiguity rather than low-level geometric registration failure and photometric vacancy verification. |
| **ObVi-SLAM (IEEE RA-L 2024)** | Dual-layer map (sparse ORB points + 3D bounding boxes) | Integrated object tracking within SLAM factor graphs | Increments absence counter purely on bounding box non-detection; lacks patch vacancy verification and raw detection decoupling. |

---

## 3. Part H: Final Research Verdict & Strategic Answers

---

### Question 1: What exactly is our two-gate vacancy-verification algorithm?
**Answer:**  
Our two-gate vacancy-verification algorithm governs when an entity can transition from `CONFIRMED_PRESENT` to `PENDING_REMOVAL` and finally `CONFIRMED_REMOVED` when it is not detected in visit $t$:

* **Gate 1: Geometric Reliability & Occlusion Guard ([`temporal/engine.py:981-1043`](src/objectmemory/temporal/engine.py#L981-L1043))**  
  1. *Registration Reliability Check:* Evaluates `_is_registration_reliable(visit.registration)` (requires homography success, inlier ratio $\ge 0.20$, and mean reprojection error $\le 4.0$px). If failed, returns `UNOBSERVED` and freezes removal.
  2. *Field of View Check:* Checks warped overlap fraction of canonical bounding box against `overlap_mask`. If $< 0.70$, returns `OUT_OF_FRAME` and freezes removal.
  3. *Foreground Occlusion Check:* Computes $\text{IoU}$ between the missing entity's canonical box and all active detections in the current frame. If any $\text{IoU} > 0.20$, returns `PARTIALLY_OCCLUDED` and freezes removal.
* **Gate 2: Photometric Patch Vacancy Test ([`temporal/engine.py:1044-1118`](src/objectmemory/temporal/engine.py#L1044-L1118))**  
  1. Warps the current query image to canonical space using homography $H$.
  2. Extracts the image patch at the entity's canonical bounding box and extracts a 148-d visual signature (`patch_sig`).
  3. Computes cosine similarity against the entity's baseline signature (`ent_sig.cosine_similarity(patch_sig)`).
  4. Marks `is_vacant = True` if cosine similarity $< 0.35$ (`vacancy_similarity_max`).
  5. Progression:
     - If `is_vacant == True`: increments `vacant_streak`. Streak 1 transitions entity to `PENDING_REMOVAL`; streak 2 ($\ge \text{vacancy\_visits\_for\_removal}$) confirms `CONFIRMED_REMOVED`.
     - If `is_vacant == False`: suppresses removal on visit 1 (retains `CONFIRMED_PRESENT` with explanation: *"Non-detection without verified vacancy cannot confirm removal; patch appearance still resembles object"*); requires 3 consecutive unobserved visits to confirm removal.

---

### Question 2: Which components are established methods?
**Answer:**  
Almost every individual component is an **established technique**:
1. **RANSAC planar homography with SIFT/ORB features:** Classic multiple-view geometry (Hartley & Zisserman, 2004).
2. **Condition number and determinant checks:** Standard numerical linear algebra practices for matrix inversion stability.
3. **Quadrilateral convexity testing via cross products:** Textbook computational geometry.
4. **Kuhn-Munkres (Hungarian) bipartite matching:** Standard data association in multi-target tracking since 1955.
5. **Patch-based appearance similarity via color histograms:** Standard baseline in computer vision and visual tracking.
6. **Bounding box IoU occlusion guards:** Standard heuristic in tracking-by-detection (e.g., DeepSORT, ByteTrack).
7. **Multi-visit confirmation counters:** Standard track management logic (Tentative $\rightarrow$ Confirmed $\rightarrow$ Deleted).

---

### Question 3: What is the closest prior-art implementation?
**Answer:**  
The closest prior-art implementations span two distinct literatures:
1. **In 3D Robotics / SLAM:**  
   *Rareș Ambruș et al. ("Meta-rooms: Building and Maintaining Long Term Spatial Models in a Dynamic World", IEEE/RSJ IROS 2014)*.  
   In Meta-rooms, spatial models of rooms are updated across long periods. An object is removed from the persistent map **only after active space carving / ray tracing** confirms that the space previously occupied by the object is now empty (free space), conditioned on valid ICP robot localization.
2. **In 2D / Planar Change Detection:**  
   *Aparna Taneja et al. ("Geometric and Photometric Change Detection", IEEE CVPR 2011, IEEE TPAMI 2015)*.  
   Warped reference and query images using estimated homographies/geometry and verified changes by computing photometric consistency of patches against an appearance background model, explicitly discarding unaligned or occluded regions.

---

### Question 4: Does LT-Mem already provide equivalent behaviour?
**Answer:**  
**No, LT-Mem does not provide equivalent behaviour, but operates on a different level of the problem.**
- **LT-Mem's Approach:** LT-Mem focuses on 3D multi-session SLAM on mobile robots. It manages object persistence through a *volatility-aware reasoning layer* with *Tri-Memory* (Live, Delta, Meta) and applies policies (*overwrite*, *hold*, *multi-hypothesis*) conditioned on semantic object classes and session-to-session evidence scoring for long-term VQA.
- **Where LT-Mem Differs:** LT-Mem does not formulate a 2D planar homography quality gate (condition number, convexity, inlier ratio), nor does it perform monocular patch appearance cosine similarity tests against an empty background to verify vacancy. Conversely, ObjectMemory does not perform 3D multi-session SLAM or VLM question-answering.

---

### Question 5: Does any existing paper combine registration reliability with vacancy verification?
**Answer:**  
**Yes, in the broader literature, but not in this specific lightweight monocular workbench configuration.**
- In **3D SLAM and volumetric mapping**, combining localization reliability with ray-traced vacancy verification has been standard practice for over a decade (*Ambruș et al., 2014*; *Hornung et al., OctoMap 2013*; *Adkins et al., ObVi-SLAM 2024*).
- In **2D planar visual monitoring**, combining explicit algebraic homography conditioning (condition number, convexity, reprojection error) with patch appearance cosine similarity vacancy checks to guard an auditable object state machine is **a distinctive systems synthesis**, but **not a fundamentally new machine learning algorithm**.

---

### Question 6: What is our strongest defensible technical distinction?
**Answer:**  
Our strongest defensible distinction is **not** an individual algorithmic primitive, but rather a **cohesive, auditable systems architecture**:
> **"A Decoupled, Registration-Gated Visual Memory System that Prevents State Corruption Under Perceptual Uncertainty while Preserving an Auditable Forensic Event Trail."**

Specifically, the system concurrently guarantees:
1. **Raw Evidence Preservation:** Storing all raw detector outputs independently of geometric registration success, ensuring that unaligned frames are permanently auditable and visualizable without polluting the persistent object state graph.
2. **Registration-Gated Absence Freezing:** Explicitly freezing absence streaks and suppressing false removals whenever algebraic registration criteria fail.
3. **Auditable Causal Certificates:** Writing exact mathematical metrics ($d_{\text{px}}$ vs. $T_{\text{effective}}$, condition number, inlier ratio, patch cosine similarity) into every persisted change event.

---

### Question 7: What evidence is still missing?
**Answer:**  
To support these claims in a peer-reviewed publication, four empirical pieces of evidence are missing:
1. **Head-to-head fault-tolerance benchmark against standard trackers:** Compare ObjectMemory against ByteTrack and BoT-SORT under severe camera shake and registration degradation, quantitatively proving that ObjectMemory achieves **0% false removals** while baselines delete or corrupt tracks.
2. **Quantitative vacancy verification ablation:** Run Ablation 2 (with vs. without patch appearance similarity) to demonstrate the exact reduction in false removal rate ($FPR_{\text{removal}}$) during temporary detector dropouts.
3. **Cross-scene multi-workbench dataset validation:** Expand beyond the single physical workbench dataset to at least 5 distinct physical workbench setups (varying surface textures, illumination, and object clutter) with 5-seed statistical variance reporting ($p < 0.05$).
4. **Illumination robustness evaluation:** Test the handcrafted 148-d RGB+HSV histogram under severe lighting temperature shifts (or replace it with a frozen DINOv2 patch descriptor).

---

### Question 8: Should we retain the current research title?
**Answer:**  
**We should refine the title.**  
The current title (*"ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes"*) is overly generic and risks confusing reviewers who may expect a 6-DoF multi-camera 3D SLAM or foundation model paper.

**Recommended Refined Titles:**
1. **Primary Recommendation (Systems / Robotics Focus — Target: IEEE RA-L / IROS):**  
   `"ObjectMemory: Fault-Tolerant Visual Memory with Auditable State Dynamics for Planar Workspaces"`  
   *(Directly highlights the fault-tolerant registration gating, auditable state machine, and specific planar workspace domain).*
2. **Alternative Recommendation (Vision / Change Detection Focus — Target: WACV / ECCV Workshop):**  
   `"Evidence-Gated Object Memory: Disentangling Perceptual Uncertainty from Long-Term Visual Change Detection"`  
   *(Emphasizes the core technical mechanism: decoupling raw sensor evidence from spatial state confirmation).*

---

## 4. Final Test Suite Verification Summary

```
======================================================================
TEST VERIFICATION REPORT (Read-Only Execution)
======================================================================
Command Executed:  cd C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory; python -m unittest discover -s tests -p "test_*.py"
Execution Result:  Exit Code 0 (SUCCESS)
Execution Time:    126.079 seconds
Total Tests:       337
Passed Tests:      337
Failed Tests:      0
Errors:            0
Skipped Tests:     0
Final Status:      OK
======================================================================
```

**Explanation of earlier activity log:**
- The earlier command that produced exit code 1 was invoked from `C:\Users\Rishh\.gemini\antigravity` (the default root working directory of `run_command` when no workspace path is specified). Because no `tests/` directory exists in the root antigravity directory, Python's `unittest` module raised `ImportError: Start directory is not importable: 'tests'` and terminated with code 1.
- Executing the command inside `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory` properly discovers the test directory and completes with **exit code 0 across all 337 tests**. No application code, schemas, thresholds, or test assertions were modified.
