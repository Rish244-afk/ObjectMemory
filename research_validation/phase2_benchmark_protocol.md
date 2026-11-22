# ObjectMemory: Phase 2 Benchmark Protocol
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Protocol Version:** 2.0 (Frozen)  
**Execution Context:** Empirical Research Validation Design

---

## 1. Research Hypothesis & Theoretical Framing

### Formal Research Hypothesis
> "An uncertainty-aware long-term visual memory system that conditions object association and state transitions on registration reliability, visibility assessment, appearance-based vacancy verification and accumulated temporal evidence can reduce unsupported movement and removal decisions compared with conventional detector-based temporal association."

### Independent Variables (Manipulated Factors)
1. **Temporal Memory Configuration:** Baseline models (B0, B1, B2, B3) and ablation variants (A1–A6).
2. **Environmental Challenge Mode:** Controlled variations across physical sequences:
   - Camera viewpoint displacement (in-plane translation, rotation, scale change).
   - Illumination shifts (ambient daylight, direct task lamp, low-light shadow).
   - Temporary partial and heavy physical occlusions.
   - Genuine object physical relocation (within-plane displacement).
   - Genuine object physical removal and subsequent physical reappearance.
   - Extreme camera viewpoint shifts resulting in out-of-frame objects or registration failure.

### Dependent Variables (Observed Responses)
1. **State Decision Accuracy:** Correctness of epistemic states (`CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `CONFIRMED_REMOVED`, `PENDING_REMOVAL`, `AMBIGUOUS_IDENTITY`).
2. **Error Rates:** False removal rate, false movement rate, identity switch count, duplicate track count.
3. **Temporal Dynamics:** Delay (number of visits) to confirm genuine removal.
4. **Registration Reliability:** Registration success rate, false acceptance rate, mean reprojection error.

### Primary Outcome Metrics
- **False Removal Proportion (FRP):** Ratio of physically present entity evaluations falsely declared `CONFIRMED_REMOVED` to total evaluations of present entities.
- **Unsupported Movement Rate (UMR):** Ratio of stationary entity visits incorrectly transitioned to `CONFIRMED_MOVED` to total stationary visits.
- **Removal Recall (RR):** Fraction of genuine physical removals successfully confirmed as `CONFIRMED_REMOVED`.
- **Identity Consistency (IDC):** Fraction of multi-visit tracks maintaining a single unique persistent identifier without ID switches or duplicate tracks.

### Secondary Outcome Metrics
- **Removal Confirmation Latency (RCL):** Mean number of consecutive visits elapsed between physical object removal and emission of `CONFIRMED_REMOVED`.
- **Identity Ambiguity Abstention Rate (IAAR):** Fraction of ambiguous visual encounters flagged as `AMBIGUOUS_IDENTITY` rather than erroneous identity mergers.
- **Registration False Acceptance Rate (RFAR):** Fraction of frames with invalid camera alignment ($> 4.0$ px reprojection error) incorrectly accepted as valid.
- **Historical Retrieval Accuracy (HRA):** Accuracy of querying past canonical positions and event timelines from persistent storage following system restart.

### Experimental Assumptions
1. **Planar Reference Surface:** The monitored environment possesses a dominant planar substrate (workbench, table, floor) on which objects rest.
2. **Object Rigidity:** Foreground objects do not undergo substantial non-rigid volumetric deformation between visits (e.g. tools, electronics, containers).
3. **Discrete Observation Visits:** Environmental monitoring proceeds via sequential observation visits ($v_0, v_1, \dots, v_T$) separated by intervals where environmental changes may occur.
4. **Ground Truth Independence:** Ground truth annotations are established by human inspection of physical ground reality, independent of any model predictions.

### Threats to Validity & Safeguards
- **Threat 1: Data Contamination & Circular Tuning:** Tuning thresholds (`vacancy_similarity_max`, `movement_displacement_threshold`) on evaluation sequences produces over-optimistic results.
  *Safeguard:* Strict train/tune/test split. Thresholds audited in `phase2_configuration_audit.md` are frozen prior to test-set execution.
- **Threat 2: Synthetic-to-Real Domain Gap:** Synthetic benchmarks generate ideal crops and artificial affine warps that hide lighting and parallax artifacts.
  *Safeguard:* Physical benchmark sequences are captured with real cameras under genuine optical blur, shadows, and perspective distortions.
- **Threat 3: Perspective Parallax Breakdown:** Severe camera translation induces 3D depth parallax that breaks planar homography.
  *Safeguard:* Evaluator quality gating detects inlier degradation and safely flags registration failure; protocol tests whether raw detections are preserved without false removal.

---

## 2. Controlled Physical Benchmark Protocol

### Benchmark Purpose & Separation from Pilot Data
The previously collected seven-image pilot (`office_pilot`) used synthetic/mocked fixtures and initial exploratory captures. It is **strictly separated** from this physical benchmark. This protocol governs the collection of genuine, photographic workbench datasets.

### Physical Scope & Sequence Matrix
A minimum of **5 complete physical multi-visit sequences** (minimum 6 visits per sequence, totaling $\ge 35$ visits) must be collected:

| Sequence ID | Primary Challenge Mode | Min Visits | Target Object Count | Description of Environmental Trajectory |
|---|---|---|---|---|
| **`PHYS-01-STAT`** | Baseline & Lighting Drift | 6 visits | 6 objects | Stationary objects under 3 illumination conditions (morning daylight, overhead fluorescent, low-angle task lamp) with minor camera shake. |
| **`PHYS-02-MOVE`** | Movement & Return-to-Baseline | 8 visits | 5 objects | Systematic relocation: Object A moved $+80$ px; Object B moved $+150$ px; Object A returned to original baseline; Object C subjected to minor coordinate drift ($< 15$ px). |
| **`PHYS-03-REM`** | Removal, Occlusion & Reappearance | 8 visits | 6 objects | Object A partially occluded by a binder; Object B physically removed; Object B absent for 3 visits; Object B returned to a new location; Object C removed. |
| **`PHYS-04-VIEW`** | Out-of-Frame & Viewpoint Extremes | 7 visits | 6 objects | Camera translates horizontally such that 2 objects leave the field of view; camera returns; camera tilts past overlap threshold ($< 15\%$ overlap) causing registration failure. |
| **`PHYS-05-DIST`** | Identical Distractor Ambiguity | 6 visits | 6 objects | Includes two identical screwdrivers and two identical white mugs placed within $60$ px of each other, testing Hungarian ambiguity margin gating. |

### Physical Workbench Environment Setup
1. **Workbench Surface:** A matte wooden or laminated tabletop ($120 \times 80\text{ cm}$) with natural surface texture.
2. **Object Diversity:** 10 diverse everyday physical objects representing common COCO classes and varied visual properties:
   - Matte electronics: Multimeter, computer mouse, calculator.
   - Cylindrical containers: Ceramic coffee mug, metal water bottle.
   - High-contrast tools: Screwdriver (yellow/black), pliers (red handle), wire stripper.
   - Planar stationery: Hardcover notebook, small cardboard box.
3. **Fiducial Ground Truth Frame (Optional Calibration Baseline):** 4 ArUco markers placed at table corners to establish high-precision millimeter ground truth poses for camera trajectory verification, independent of the natural feature SIFT registration.
4. **Camera Placement & Acquisition:**
   - Handheld or tripod-mounted mirrorless/smartphone camera (1080p minimum, fixed 24mm–35mm equivalent focal length).
   - Fixed exposure and white-balance lock per sequence to isolate physical lighting changes from camera auto-exposure hunting.
   - For each visit $v_t$, capture a single high-resolution image ($1920 \times 1080$ or $1280 \times 720$), recorded in chronological order with ISO 8601 timestamps.

### Physical Capture Workflow
```
[Physical State Setup] -> [Capture Image v_t] -> [Record Physical Ground Truth Log]
         |
         v
[Apply Physical Transformation (e.g. Move Tool A by 10cm)]
         |
         v
[Capture Image v_{t+1}] -> [Record Physical Ground Truth Log]
```

### Physical Ground-Truth Logging Procedure
At each visit $v_t$, the physical experimenter records the ground reality in a structured log:
1. Physical existence of each object on the table (`PRESENT`, `REMOVED`).
2. Physical position coordinates (measured via workbench grid ruler in millimeters).
3. Physical visibility status from camera viewpoint (`VISIBLE`, `PARTIALLY_OCCLUDED`, `HEAVILY_OCCLUDED`, `OUT_OF_FRAME`).
4. Event occurring between $v_{t-1}$ and $v_t$ (`STATIONARY`, `MOVED`, `REMOVED`, `REAPPEARED`).

---

## 3. OpenLORIS-Scene Protocol

### Role in Research Validation
OpenLORIS-Scene is an unconstrained multi-session robotic SLAM dataset captured by a RealSense D435i camera mounted on a wheeled mobile robot.
- **What it Validates:** Real-world camera motion, optical motion blur, dynamic lighting shifts across sessions, natural feature matching degradation, and registration failure handling.
- **What it Does NOT Supply:** OpenLORIS does not provide object-level persistence, movement, or removal labels. It is strictly used as an empirical benchmark for **registration robustness, viewpoint overlap evaluation, and sensor failure gating**.

### Sequence Selection & Matched Viewpoint Grouping
Using the Phase 1C synchronization audit:
- **Baseline Pair:** `office1-1` (Reference traversal) and `office1-2` / `office1-3` (Query traversals on same day).
- **Multi-Day Cross-Session Pair:** `office1-1` paired with `office1-6` (Recorded on a later date under daylight shifts).

### Keyframe Extraction & Viewpoint Grouping Protocol
To avoid evaluating arbitrary disparate frames where the robot is in completely different rooms:
1. **Pose-Based Viewpoint Clustering:** Using synchronized OptiTrack ground-truth poses $(x, y, z, q_w, q_x, q_y, q_z)$, query frames are matched to reference frames whose camera optical axes intersect the same region within:
   $$\Delta \mathbf{p} \le 0.40\text{ m},\quad \Delta \theta \le 25^\circ$$
2. **Keyframe Subsampling:** Extract frames at $1.0\text{ Hz}$ (or every $0.25\text{ m}$ of robot motion) to prevent redundant autocorrelation.
3. **Leakage Prevention Rule:** Frame splitting must be performed **by continuous sequence segment**, never by random frame shuffle. Traversal segments from the same spatial turn or viewpoint cluster must reside entirely in either the tuning split or the test split.

### Registration Metrics Evaluated on OpenLORIS
- **Registration Success Rate (RSR):** Percentage of matched-viewpoint frame pairs that pass all 7 quality gates.
- **Inlier Ratio Distribution:** Median, 5th, and 95th percentiles of SIFT/ORB inliers across lighting variations.
- **Mean Reprojection Error ($\bar{e}_{\text{reproj}}$):** Distribution across successful registrations.
- **Planar Degeneracy Rejection:** Number of times the homography quality gate correctly rejects 3D non-planar traversals (triggering `AlignmentStatus.HOMOGRAPHY_ESTIMATION_FAILED` or `EXCESSIVE_REPROJECTION_ERROR`).

---

## 4. Multi-Dimensional Annotation Schema

The benchmark enforces an explicit multi-dimensional annotation contract where physical reality, observational quality, and temporal changes are strictly decoupled.

### Field Specification

```json
{
  "sequence_id": "PHYS-03-REM",
  "visit_id": "visit_04",
  "timestamp": "2026-11-14T10:15:00Z",
  "annotations": [
    {
      "gt_entity_id": "tool_multimeter_01",
      "class_label": "multimeter",
      "canonical_bbox": [245.0, 180.0, 365.0, 310.0],
      "query_bbox": [280.0, 195.0, 400.0, 325.0],
      "visibility": "VISIBLE",
      "observation_quality": "CLEAR",
      "session_presence": "PRESENT",
      "change_status": "STATIONARY",
      "expected_epistemic_state": "CONFIRMED_PRESENT",
      "provenance": "MANUAL_INSPECTION_RULER_VERIFIED"
    }
  ]
}
```

### Enumerated Attribute Dimensions

#### Dimension 1: Visibility Status (`visibility`)
- `VISIBLE`: Unobstructed optical line of sight; bounding box fully inside camera frame ($> 90\%$ visible).
- `PARTIALLY_OCCLUDED`: Foreground obstacle covers between $10\%$ and $70\%$ of the object's surface area.
- `HEAVILY_OCCLUDED`: Foreground obstacle covers $> 70\%$ of the surface area, preventing reliable detection.
- `OUT_OF_FRAME`: Object is physically on the table, but the camera viewpoint frustum does not cover its coordinates.
- `UNKNOWN`: Visibility cannot be established with certainty.

#### Dimension 2: Identity Association Quality (`identity`)
- `MATCHED`: Uniquely identifiable object with unambiguous appearance and spatial correspondence.
- `AMBIGUOUS`: Visually indistinguishable from another object present in the scene within close proximity.
- `UNMATCHED`: Novel object appearing for the first time without prior history.

#### Dimension 3: Observation Quality (`observation_quality`)
- `CLEAR`: Sharp focus, adequate illumination, no significant motion blur.
- `MOTION_BLURRED`: Camera motion or object movement induces noticeable edge degradation.
- `LOW_CONFIDENCE`: Extreme low-light noise, intense specular glare, or severe shadow.
- `REGISTRATION_FAILED`: Global camera alignment failed; frame coordinates cannot be projected.

#### Dimension 4: Ground Reality Presence (`session_presence`)
- `PRESENT`: Object is physically resting on the workbench surface.
- `ABSENT_CONFIRMED`: Object was physically picked up and removed from the room.
- `UNKNOWN`: Object status outside camera range is unverified.

#### Dimension 5: Physical Change Status (`change_status`)
- `STATIONARY`: Object has not moved physically since the baseline observation ($\Delta d \le 10\text{ mm}$).
- `MOVED`: Object was physically picked up and placed in a different location ($\Delta d > 10\text{ mm}$).
- `REMOVED`: Object was physically taken away from the workbench.
- `REAPPEARED`: Object previously removed has been returned to the workbench.
- `UNKNOWN`: Change status cannot be determined.

### Invariant Ground-Truth Rules
1. **Never Infer Removal from Detector Miss Alone:** A missing bounding box in a query image does NOT warrant an annotation of `REMOVED`. If the object is physically on the table, `session_presence` MUST be annotated as `PRESENT`, and `visibility` annotated as `VISIBLE` or `OCCLUDED`.
2. **Independent Provenance:** Every annotation file must record the human annotator ID and inspection method (`provenance: "MANUAL_INSPECTION"`).

---

## 5. Evaluation Metrics Formulation

Every metric is defined with its exact mathematical formula and explicit denominator. We strictly distinguish **False Discovery Proportion (FDP)** from **False Positive Rate (FPR)**.

### A. Detection Quality Metrics (Per Visit)
- **True Positives ($TP_{\text{det}}$):** Detected bounding box matched to ground-truth box with $\text{IoU} \ge 0.45$ and matching class label.
- **False Positives ($FP_{\text{det}}$):** Detection with no matching ground-truth box ($\text{IoU} < 0.45$ or mismatched class).
- **False Negatives ($FN_{\text{det}}$):** Ground-truth object with no matching detection.
$$\text{Precision}_{\text{det}} = \frac{TP_{\text{det}}}{TP_{\text{det}} + FP_{\text{det}}},\quad \text{Recall}_{\text{det}} = \frac{TP_{\text{det}}}{TP_{\text{det}} + FN_{\text{det}}},\quad F1_{\text{det}} = \frac{2 \cdot P_{\text{det}} \cdot R_{\text{det}}}{P_{\text{det}} + R_{\text{det}}}$$

### B. Track Continuity & Identity Metrics
- **Identity Switches ($ID_{\text{sw}}$):** Number of times a continuous ground-truth object track is assigned a different system entity ID $e_{\text{sys}}$ between visit $v_{t-1}$ and $v_t$.
- **Duplicate Track Rate ($DTR$):** Number of concurrent active system entities mapped to the same single physical ground-truth object divided by total ground-truth objects:
  $$DTR = \frac{N_{\text{duplicate\_tracks}}}{N_{\text{gt\_objects}}}$$
- **Identity Consistency ($IDC$):**
  $$IDC = \frac{\sum_{i=1}^{N_{\text{gt}}} \mathbb{I}(\text{track}_i \text{ has 0 ID switches and 0 duplicate tracks})}{N_{\text{gt}}}$$

### C. Movement Decision Metrics
Evaluated over all entity-visit observations where camera registration succeeded:
- **True Relocation ($TP_{\text{move}}$):** Object physically moved ($\text{change\_status} == \text{MOVED}$), and system emitted transition to `CONFIRMED_MOVED`.
- **False Relocation ($FP_{\text{move}}$):** Object is stationary ($\text{change\_status} == \text{STATIONARY}$), but system emitted transition to `CONFIRMED_MOVED` (due to coordinate drift, framing truncation, or registration jitter).
- **Missed Relocation ($FN_{\text{move}}$):** Object physically moved, but system failed to transition to `CONFIRMED_MOVED`.
$$\text{Precision}_{\text{move}} = \frac{TP_{\text{move}}}{TP_{\text{move}} + FP_{\text{move}}},\quad \text{Recall}_{\text{move}} = \frac{TP_{\text{move}}}{TP_{\text{move}} + FN_{\text{move}}}$$
- **Unsupported Movement Rate ($UMR$):**
  $$UMR = \frac{FP_{\text{move}}}{N_{\text{stationary\_evaluations}}}$$

### D. Removal Decision Metrics
- **True Removal ($TP_{\text{rem}}$):** Object is physically absent ($\text{session\_presence} == \text{ABSENT\_CONFIRMED}$), and system reached `CONFIRMED_REMOVED`.
- **False Removal ($FP_{\text{rem}}$):** Object is physically present on the workbench ($\text{session\_presence} == \text{PRESENT}$), but system declared `CONFIRMED_REMOVED` (due to detector miss, occlusion, or bad camera alignment).
- **Missed Removal ($FN_{\text{rem}}$):** Object is physically absent, but system failed to reach `CONFIRMED_REMOVED` (remains stuck in `CONFIRMED_PRESENT` or `PENDING_REMOVAL` indefinitely).
$$\text{Recall}_{\text{rem}} = \frac{TP_{\text{rem}}}{TP_{\text{rem}} + FN_{\text{rem}}}$$
- **False Removal Rate ($FRR$ / False Discovery Proportion):**
  $$FRR = \frac{FP_{\text{rem}}}{TP_{\text{rem}} + FP_{\text{rem}}} \quad (\text{FDP of removal declarations})$$
- **False Removal on Present Rate ($FRPR$ / False Positive Rate):**
  $$FRPR = \frac{FP_{\text{rem}}}{N_{\text{physically\_present\_evaluations}}}$$

### E. Temporal Dynamic Metrics
- **Removal Confirmation Latency ($RCL$):** For all true removals $i \in TP_{\text{rem}}$, let $t_{\text{physical}}$ be the visit index when the object was taken away, and $t_{\text{confirmed}}$ be the visit index when the system recorded `CONFIRMED_REMOVED`:
  $$RCL = \frac{1}{|TP_{\text{rem}}|} \sum_{i \in TP_{\text{rem}}} (t_{\text{confirmed}, i} - t_{\text{physical}, i})$$
  (For 2-visit vacancy confirmation, expected optimal $RCL = 2.0\text{ visits}$).

### F. Camera Registration & Failure Gating Metrics
- **Registration Success Rate ($RSR$):**
  $$RSR = \frac{N_{\text{accepted\_registrations}}}{N_{\text{total\_registration\_attempts}}}$$
- **Registration False Acceptance Rate ($RFAR$):**
  $$RFAR = \frac{N(\text{registration accepted with actual error } > 4.0\text{ px})}{N_{\text{accepted\_registrations}}}$$

### G. Statistical Reporting Guidelines
1. **Sequence-Level Breakdown:** All metrics must be reported per sequence in addition to aggregate totals.
2. **Uncertainty Intervals:** For aggregate metrics, report $95\%$ bootstrap confidence intervals across sequence repetitions ($N = 1000$ bootstrap iterations).
3. **No Arbitrary Combined Score:** Do NOT compute an ad-hoc weighted average score. Report detection, association, movement, and removal metrics as independent scientific dimensions.
