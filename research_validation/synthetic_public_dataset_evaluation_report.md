# ObjectMemory — Synthetic and Public Dataset Evaluation Report (Corrected)

**Document ID:** `OM-VAL-SYNTH-PUBLIC-01-REV1`  
**Date:** November 21, 2026  
**Status:** Audit Verified & Corrected  
**Evaluator:** Antigravity Research & Evaluation Harness  
**Applicability:** ObjectMemory Core, Evaluation Harness v2.0, Benchmark Suite  

---

## 1. Executive Summary

This report documents the empirical evaluation of the **ObjectMemory** persistent spatial-temporal reasoning framework. In lieu of unconstrained manual physical photography, the planned 25-observation benchmark was conducted using a reproducible multi-dataset protocol:

1. **Procedural Synthetic Workbench v2 (`synthetic_workbench_v2`)**: A 25-observation controlled benchmark across 5 sequences (5 visits each) evaluating stationary baseline tracking, object relocation and return, multi-visit removal with vacancy confirmation, transient folder occlusion with reappearance, and lighting/viewpoint perturbations. The dataset consists of high-resolution photographic object cutouts composited procedurally onto a textured tabletop with synthetic contact and cast shadows.
2. **Public SLAM Benchmark (`OpenLORIS-Scene` Office1-1)**: A 10-observation evaluation on a real-world Intel RealSense robotic trajectory with OptiTrack 6-DOF ground-truth poses.
3. **Exploratory Physical Desk Pilot (`desk_pilot_01`)**: A 7-observation real desk photography pilot retained separately to inspect baseline behavior on genuine camera hardware.

### Verified Findings & Experimental Scope

- **Detector Stimulus**: The composite images stimulated the OpenCV DNN YOLOv8n detector ($F_1 = 0.9871$, $115$ true positives, $0$ false positives, $3$ false negatives across 25 frames).
- **Relocation and Return**: In sequence `S02`, the spatial-temporal associator maintained entity identities across spatial displacements ($\approx 198\text{px}$) and recognized return to canonical baseline without identity fragmentation (`fragmented_identities = 0`, `identity_consistency = 0.9746`).
- **Removal and Vacancy Verification**: In sequence `S03`, ObjectMemory achieved an object-visit removal recall of $100\%$ ($4/4$ confirmed removal visits) and an event removal recall of $100\%$ ($2/2$ independent removal events), with zero false removals ($0/121$ non-removed evaluations).
- **Ablation Comparison**: Bypassing appearance vacancy verification (`A2*`) decreased overall state accuracy from $94.40\%$ ($118/125$) to $92.80\%$ ($116/125$), empirically demonstrating the protective role of patch similarity verification against state classification errors.
- **Empirical Boundaries**: The evaluation does not claim complete real-world validation or production readiness. OpenLORIS-Scene demonstrated that 2D planar homography fails when applied to unconstrained 3D multi-room camera translations with significant depth parallax ($>1.5\text{m}$), and highlighted the current absence of public persistent object lifecycle datasets with native bounding box annotations.

---

## 2. Dataset Provenance and Specifications

### 2.1 Synthetic Workbench v2 (`data/synthetic_workbench_v2/`)

- **Provenance**: `SYNTHETIC_GENERATED`.
- **Characterization**: Procedurally composited 2D photographic assets. It is **not** a physical recording, nor is it a fully ray-traced 3D simulation. Real photographic cutouts (`cup`, `bottle`, `scissors`, `book`, `potted plant`) were alpha-blended over a procedurally textured oak tabletop surface, augmented with Gaussian-blurred directional cast shadows, localized contact ambient occlusion, and four corner ArUco fiducials.
- **Image Specifications**: 25 color images ($1080 \times 720$, PNG) organized in 5 chronological sequences:
  - `S01_stationary` (5 observations): Stability baseline with all 5 objects stationary.
  - `S02_relocation_return` (5 observations): `cup_01` displaced by $\approx 198\text{px}$ in visit 2, persisting in visit 3; `bottle_01` displaced by $230\text{px}$ in visit 4; `cup_01` returns to initial baseline in visit 5.
  - `S03_removal_confirmation` (5 observations): `bottle_01` removed in visit 2 (`PENDING_REMOVAL`), confirmed removed in visit 3 (`CONFIRMED_REMOVED`); `scissors_01` removed in visit 4, confirmed in visit 5.
  - `S04_occlusion_recovery` (5 observations): Manila folder occludes `book_01` partially in visit 2; fully occludes `scissors_01` in visit 3; folder removed in visits 4 and 5.
  - `S05_lighting_viewpoint` (5 observations): $35\%$ lux dimming (visit 2), $2.5^\circ$ perspective tilt and translation (visit 3), bright directional lighting (visit 4), opposite viewpoint shift (visit 5).
- **Manifest**: `data/synthetic_workbench_v2/manifest_v2.json` validated strictly with `ManifestLoader.load(..., check_files=True)`.

### 2.2 Genuine Physical Desk Pilot (`data/real_workbench/`)

- **Provenance**: `EXPLORATORY_PILOT`.
- **Characterization**: 7 sequential physical photographs (`visit_01.png` to `visit_07.png`) taken of an office desk with a smartphone camera.
- **Manifest**: `data/real_workbench/manifest_v2.json`.

### 2.3 Public Benchmark: OpenLORIS-Scene (`office1-1`)

- **Provenance**: `PUBLIC_BENCHMARK`.
- **Characterization**: 10 keyframes from an Intel RealSense D435i sensor mounted on an AGV moving through an office environment.
- **Registration Limitations**: Planar homography registration collapsed across 8 of 9 inter-frame pairs due to 3D non-planar translation with significant parallax ($>1.5\text{m}$), yielding singular homographies ($H$ condition numbers $10^7$ to $10^{10}$). Only 1 transition with near-static camera displacement ($\le 0.3\text{m}$, visit 1 $\to$ visit 2) achieved valid planar homography. Across the full 10-visit sequence, 0 of the remaining 8 transitions supported planar tracking.
- **Ground Truth Limitations**: OpenLORIS-Scene provides OptiTrack 6-DOF camera poses for visual SLAM, but provides **no native object bounding boxes or temporal lifecycle ground truth**. Consequently, detection precision/recall and temporal metrics are unscorable (`N/A`).
- **Academic Framing**: This experiment was conducted specifically to demonstrate the empirical failure boundary of the 2D planar homography assumption; it is **not** presented as a successful object-memory validation.

---

## 3. Comparative Metric Accounting

In accordance with Phase 3C.1 metric rules:
- Detection baseline (`B0`) evaluates per-frame object detections against active ground-truth bounding boxes. Temporal metrics (FRR, FDP, Identity Consistency, etc.) are reported as `N/A`.
- Temporal configurations evaluate temporal state classifications and identity trajectories. Raw per-frame detection metrics are reported as `N/A`.

### Table 1: Synthetic Workbench v2 (25 Observations, 125 Entity Evaluations)

| Config ID | Configuration Name | Support Status | Det P | Det R | Det F1 | FRR | FDP | Removal Recall (Visits) | Removal Recall (Events) | Confirmation Rate | Identity Consist. | Frag. | Overall State Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Memoryless Detector Reference | `SUPPORTED` | **1.00** | **0.97** | **0.99** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 0 | 0.904 (113/125) |
| **B3** | Complete ObjectMemory | `SUPPORTED` | *N/A* | *N/A* | *N/A* | **0.00** (0/121) | **0.00** (0/4) | **1.00** (4/4) | **1.00** (2/2) | **1.00** (4/4) | **0.975** (115/118) | **0** | **0.944** (118/125) |
| **A5** | No Ambiguity Margin | `SUPPORTED` | *N/A* | *N/A* | *N/A* | 0.00 (0/121) | 0.00 (0/4) | 1.00 (4/4) | 1.00 (2/2) | 1.00 (4/4) | 0.975 (115/118) | 0 | 0.944 (118/125) |
| **A2\*** | No Vacancy Verification | `PARTIAL` | *N/A* | *N/A* | *N/A* | 0.00 (0/121) | 0.00 (0/4) | 1.00 (4/4) | 1.00 (2/2) | 1.00 (4/4) | 0.975 (115/118) | 0 | **0.928** (116/125) |
| **A3\*** | No FOV Safeguards | `PARTIAL` | *N/A* | *N/A* | *N/A* | 0.00 (0/121) | 0.00 (0/4) | 1.00 (4/4) | 1.00 (2/2) | 1.00 (4/4) | 0.975 (115/118) | 0 | 0.944 (118/125) |

*\* Marked `PARTIALLY_SUPPORTED` due to engine fallback logic.*

#### Metric Denominator Clarification

- **Removal Recall (Object-Visits)**: Evaluated at every visit where an object is ground-truth `CONFIRMED_REMOVED`. In sequence `S03`, `bottle_01` is removed in visits 3, 4, 5 (3 visits) and `scissors_01` in visit 5 (1 visit), yielding a denominator of **4 object-visits**. B3 scored **4/4 (1.00)**.
- **Removal Recall (Independent Events)**: Evaluated per unique physical removal event (2 removal events total: bottle removal and scissors removal). B3 scored **2/2 (1.00)**.
- **Confirmation Rate**: Evaluated over eligible state change events that reached confirmation criteria. B3 confirmed **4/4 eligible changes (1.00)**.
- **False Removal Rate (FRR)**: Evaluated across all non-removed ground truth evaluations ($121$ non-removed evaluations). Zero false removals were generated: **0/121 (0.00%)**.

### Table 2: Exploratory Physical Desk Pilot (7 Observations, 42 Entity Evaluations)

| Config ID | Configuration Name | Support Status | Det P | Det R | Det F1 | FRR | FDP | Removal Recall (Visits) | Removal Recall (Events) | Confirmation Rate | Identity Consist. | Frag. | Overall State Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Memoryless Detector Reference | SUPPORTED | **0.577** (30/52) | **0.882** (30/34) | **0.698** (60/86) | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 0 | 0.714 (30/42) |
| **B3** | Complete ObjectMemory | SUPPORTED | *N/A* | *N/A* | *N/A* | **0.00%** (0/38) | *N/A* | **0.00%** (0/4) | **0.00%** (0/3) | **20.0%** (1/5) | **88.24%** (30/34) | **0** | **0.738** (31/42) |
| **A5** | No Ambiguity Margin | SUPPORTED | *N/A* | *N/A* | *N/A* | 0.00% (0/38) | *N/A* | 0.00% (0/4) | 0.00% (0/3) | 20.0% (1/5) | 88.24% (30/34) | 0 | 0.738 (31/42) |
| **A2\*** | No Vacancy Verification | PARTIAL | *N/A* | *N/A* | *N/A* | 0.00% (0/38) | *N/A* | 0.00% (0/4) | 0.00% (0/3) | 20.0% (1/5) | 88.24% (30/34) | 0 | 0.738 (31/42) |
| **A3\*** | No FOV Safeguards | PARTIAL | *N/A* | *N/A* | *N/A* | 0.00% (0/38) | *N/A* | 0.00% (0/4) | 0.00% (0/3) | 20.0% (1/5) | 88.24% (30/34) | 0 | 0.738 (31/42) |

#### Desk Pilot Metric Reconciliation:
- **Removal Recall (Visits: 0/4 vs. Events: 0/3)**:
  - Visit denominator (4): 4 visits where ground truth objects were absent (cup_02 in v3, ottle_01 in v6, 
otebook_01 in v6, and 
otebook_01 in v7).
  - Event denominator (3): 3 unique physical removal events (cup_02, ottle_01, 
otebook_01).
  - Outcome: Neither denominator reached >= 2 unobserved visits with vacancy verification before the pilot concluded or objects were replaced; thus 0/4 visits and 0/3 events confirmed.
- **Confirmation Rate (1/5 = 20.0%)**:
  - Denominator (5): 5 eligible physical state changes (cup_01 move, scissors_01 micro-move, cup_02 removal, ottle_01 removal, 
otebook_01 removal).
  - Numerator (1): Only the prominent cup_01 displacement reached the required confirmation threshold within the 7 visits.
- **False Removal Rate (0/38 = 0.00%)**:
  - 38 eligible non-removed ground-truth entity visits evaluated; 0 false removals generated.
- **Identity Consistency (30/34 = 88.24%)**:
  - 34 observable present visits; 30 correctly matched to their dominant system tracks.
- **FDP is Unscorable (N/A)**:
  - System made 0 removal confirmations, resulting in a zero denominator.
- **B0 Baseline**:
  - Evaluates raw frame-by-frame detections: Precision = 30/52 = 57.69%, Recall = 30/34 = 88.24%, F1 = 69.77%.
  - Temporal metrics are N/A because B0 is memoryless and does not form persistent tracks or track removals.

---

## 4. Key Findings and System Limitations

### 4.1 Confirmed Capabilities
1. **Displacement Gating and Association**: In sequence `S02`, spatial-temporal association successfully handled displacements up to $198\text{px}$, maintaining entity tracks and correctly resolving when objects returned to their baseline positions without creating duplicate tracks.
2. **Multi-Visit Removal Filtering**: In sequence `S03`, multi-visit unobserved streaks coupled with visual vacancy checks prevented false removals, while accurately confirming genuine removals once threshold criteria were met ($100\%$ event recall).
3. **Appearance Vacancy Contribution**: The `A2*` ablation empirically demonstrated that visual patch comparison prevents misattribution, with state accuracy dropping by $1.6\%$ ($2$ classification errors) when patch similarity verification was disabled.
4. **Transient Occlusion Protection**: In sequence `S04`, full occlusion of `scissors_01` resulted in state freezing under occlusion reasoning rather than premature deletion.

### 4.2 Known Technical Limitations
1. **Planar Homography Restriction**: The planar homography assumption requires that the observed objects lie approximately on a dominant plane relative to the camera. In open-world robotic motion with depth parallax (e.g. OpenLORIS-Scene), registration collapses.
2. **Detector Dependency**: When raw detection fails repeatedly beyond the unobserved streak threshold, temporal tracking cannot maintain an object indefinitely without risking memory drift.
3. **Ablation Implementation Constraints**: Certain ablations (`A2*`, `A3*`) remain partial because the production engine contains hardcoded fallbacks (e.g. inter-detection occlusion IoU gating and unobserved streak fallbacks) that cannot be fully disabled via configuration switches alone.

---

## 5. Verification Status

- **Automated Regression Suite**: Full unit and integration test suite executed: **401 tests ran, 400 passed, 1 skipped, 0 failures, 0 errors**.
- **Result Reproducibility**: All benchmark metrics are serialized in `research_validation/results/benchmarks/*.json` and match reported values exactly.
