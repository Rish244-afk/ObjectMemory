# Phase 5: Multi-Visit Temporal Error Analysis and Failure Modes

> **Project**: ObjectMemory Studio  
> **Audit Source**: `scripts/temporal_error_analysis_results.json`  
> **Regression Tests**: `tests/test_temporal_failure_modes.py` (10 tests, 100% pass)  
> **Physical Dataset**: 7-visit desk sequence (`data/real_workbench/`)  

---

## 1. Executive Summary & Separation of Evidence

To adhere to rigorous scientific standards, **engineered test counts are strictly separated from physical-scene measurements**:

1. **Controlled Engineered Tests (Unit Fixtures)**:
   - Evaluates the core algorithms against synthetic, edge-case, and boundary conditions with exact programmatic ground truth.
   - **Result**: **10 out of 10 failure mode scenarios PASSED (100%)**.
   - Proves algorithmic correctness of causal monotonicity, registration gating, cross-class association, and state machine transitions.
2. **Physical Scene Measurements (7 Real Workbench Visits)**:
   - Evaluates the complete multi-visit pipeline (`RealImagePipelineRunner`) running on real photographs across 7 visits (`visit_01.png` to `visit_07.png`).
   - Reflects actual physical conditions including lighting variations, camera FOV shifts, and detector prediction jitter.
   - **Result**: 0 false removals on control objects; registration failure gating successfully halted absence progression during low-inlier frames; identified camera crop shift as an operational challenge for planar homography.

---

## 2. Temporal Error Taxonomy Matrix

| Error / Decision Category | Engineered Test Suite Count | Physical Desk Sequence (YOLOv8n) | Physical Desk Sequence (YOLO-World) | Primary Root Cause / Architectural Safeguard |
| :--- | :---: | :---: | :---: | :--- |
| **False Movement Events** | **0** | 2 | 1 | Induced by camera FOV/crop shift between V3 (~780x636) and V4-V7 (~584x576), causing ~98px projective shift. |
| **False Removal Events** | **0** | **0** | **0** | **Zero unwarranted removals** on stationary control objects (`potted plant`, `book`). |
| **Missed Movement Events** | **0** | 1 | 1 | In `visit_05`, registration inlier ratio fell to 0.18 (< 0.20 threshold). Movement was safely gated. |
| **Missed Removal Events** | **0** | **0** | **0** | Verified vacancy logic correctly progressed `cup_02` and `bottle_01` to `CONFIRMED_REMOVED`. |
| **Duplicate Entities** | **0** | 1 | **0** | YOLOv8n experienced edge clutter on desk corner; YOLO-World produced 0 duplicate entities. |
| **Incorrect Associations** | **0** | 0 | 0 | Kuhn-Munkres matching with 148-d visual signature similarity prevented identity cross-talk. |
| **Registration-Gated Decisions**| 1 | 5 | 5 | Gating triggered whenever inlier ratio $< 0.20$ or homography is unverified, blocking false claims. |
| **Uncertain / Deferred Decisions**| 1 | 2 | 0 | Single-frame detection drops transition to `PENDING_REMOVAL`, deferring confirmed physical changes. |

---

## 3. Detailed Audit of the 10 Failure Modes

All 10 failure modes specified in Task 2 were tested using dedicated integration fixtures in `tests/test_temporal_failure_modes.py`:

### Mode 1: One Missed Detection Must Not Confirm Removal
- **Requirement**: An object missing for a single visit must not be declared removed.
- **Engine Behavior**: In `visit_3`, when the object is absent, the engine records a vacant streak of 1 and transitions the entity from `CONFIRMED_PRESENT` to `PENDING_REMOVAL`.
- **Database Verification**: `store.get_entity(id).current_state == EpistemicState.PENDING_REMOVAL`. Exactly 1 `ChangeEvent` recorded with `new_state = PENDING_REMOVAL`.
- **Status**: **PASS**.

### Mode 2: Multiple Missed Detections Under Unreliable Registration
- **Requirement**: Missing detections when registration fails (inlier ratio $< 0.40$ or homography is null) must never accumulate absence evidence.
- **Engine Behavior**: Across 3 consecutive unaligned visits, the engine evaluates `_is_registration_reliable(visit.registration) == False`. It leaves `vacant_streak` at 0 and maintains `CONFIRMED_PRESENT`.
- **Database Verification**: `vacant_streak == 0`, `current_state == EpistemicState.CONFIRMED_PRESENT`, 0 transitions recorded.
- **Status**: **PASS**.

### Mode 3: Low-Confidence Detection Preserves Historical Canonical Coordinates
- **Requirement**: Unreliable or low-confidence detections must not corrupt verified historical observations.
- **Engine Behavior**: In `_touch_entity`, canonical bounding boxes are only updated if `is_spatial_reliable == True`. Previous `Observation` records in SQLite are append-only and immutable.
- **Database Verification**: `store.get_entity(id).canonical_bbox` retains the baseline canonical coordinates; previous observation records retain original confidence (0.95).
- **Status**: **PASS**.

### Mode 4: Label Changes Must Not Duplicate Persistent Entities
- **Requirement**: When a detector alternates between `book`, `notebook`, and `laptop` across visits at the same spatial coordinates, the engine must maintain a single entity.
- **Engine Behavior**: The conservative cross-class associator matches spatial proximity ($\text{IoU} \ge 0.50$, centroid distance $< 35\text{px}$) and 148-d appearance signatures.
- **Database Verification**: `len(store.list_entities()) == 1`, entity ID remains `ent_v1_1`, all 3 observation records attached to the single entity.
- **Status**: **PASS**.

### Mode 5: Similar-Looking Objects Must Not Merge Without Sufficient Evidence
- **Requirement**: Two near-identical objects competing within the ambiguity margin must not be incorrectly merged into a single entity.
- **Engine Behavior**: When candidate affinity difference falls below `ambiguity_margin` (0.10), the associator sets `is_ambiguous = True` and refuses assignment.
- **Database Verification**: 2 distinct persistent entities maintained in SQLite; neither is falsely overwritten.
- **Status**: **PASS**.

### Mode 6: Moved Object vs. Camera Motion
- **Requirement**: Camera translation $H$ shifts all raw pixel coordinates; the engine must distinguish global camera shifts from physical object relocation.
- **Engine Behavior**: Stationary object's canonical coordinates remain static ($\Delta \le 35\text{px}$), maintaining `CONFIRMED_PRESENT`. Truly moved object has canonical displacement ($120\text{px} > 35\text{px}$), transitioning to `CONFIRMED_MOVED` with recorded `displacement_distance = 120.0px`.
- **Database Verification**: `stat_obj` remains `CONFIRMED_PRESENT`; `move_obj` transitions to `CONFIRMED_MOVED` with `displacement_distance = 120.0px`.
- **Status**: **PASS**.

### Mode 7: Reappearance Preserves Complete Entity History
- **Requirement**: An object that progresses to `CONFIRMED_REMOVED` and later reappears must restore its original entity ID and audit chain.
- **Engine Behavior**: When re-observed at its prior canonical position, the engine restores the entity to `CONFIRMED_PRESENT`.
- **Database Verification**: Entity ID `ent_v1_1` preserved; SQLite contains the full transition history: `CONFIRMED_PRESENT` $\rightarrow$ `PENDING_REMOVAL` $\rightarrow$ `CONFIRMED_REMOVED` $\rightarrow$ `CONFIRMED_PRESENT`.
- **Status**: **PASS**.

### Mode 8: Uncertain Observations Not Reported as Certain Physical Changes
- **Requirement**: An occluded or partially visible object must not trigger false removal or false movement.
- **Engine Behavior**: When another detected object overlaps the canonical region ($\text{IoU} > 0.20$), the engine flags `VisibilityStatus.PARTIALLY_OCCLUDED` and halts removal progression.
- **Database Verification**: Entity remains `CONFIRMED_PRESENT`; 0 change events fired.
- **Status**: **PASS**.

### Mode 9: Singular or Degenerate Homography Does Not Corrupt Coordinates
- **Requirement**: Non-finite homography matrices ($NaN$, determinant = 0) must return `None` and halt spatial updates.
- **Engine Behavior**: `transform_bbox_to_canonical` checks `np.isfinite(H)` and singular values; returns `None` on degenerate matrices.
- **Database Verification**: Persistent canonical bounding box in SQLite is untouched.
- **Status**: **PASS**.

### Mode 10: Repeated Ingestion of Same Visit is Idempotent
- **Requirement**: Reprocessing an existing visit session must not create duplicate entities, observations, or change events.
- **Engine Behavior**: The engine detects existing observations for `visit.session_id` and builds an idempotent audit report (`is_reprocessed = True`).
- **Database Verification**: Entity count, observation count, and change event count are identical before and after reprocessing.
- **Status**: **PASS**.

---

## 4. Physical Desk Sequence Error Analysis

During the 7-visit desk sequence (`data/real_workbench/`), several real-world environmental phenomena were observed:

### 4.1 Camera Crop & Resolution Shift (V3 vs. V4)
- **Manifest Documented Reality**: Visits 1–3 were captured at 780x636 resolution. Visits 4–7 were captured at 584x576 resolution with a tighter camera crop on the center of the desk.
- **Homography Impact**: Feature-based planar registration between the reference frame (V1) and V4–V7 experiences higher perspective distortion. In `visit_05`, the inlier ratio fell to 0.1809 ($< 0.20$ threshold).
- **Engine Safeguard Response**: The registration safeguard activated as designed:
  - It declared `AlignmentStatus.INSUFFICIENT_INLIERS`.
  - It gated spatial updates, preventing erroneous object relocations and false removals.
  - *Trade-off*: In `visit_05`, genuine movement of `cup_01` was missed because the engine prioritizes epistemic safety over aggressive assertion under unaligned conditions.

### 4.2 Pixel Drift vs. Movement Threshold
- In `visit_06`, the book exhibited a measured canonical displacement of $98.3\text{px}$ due to the compound projective warp between V1 and V6.
- Because $98.3\text{px} > 35\text{px}$ (the movement threshold), the engine triggered a `CONFIRMED_MOVED` event for the book.
- **Research Implication**: Fixed pixel displacement thresholds ($35\text{px}$) are sensitive to camera distance and FOV shifts. Future research should adopt metric-scaled thresholds (e.g., fraction of desk width or calibrated metric distance) rather than raw pixels.

---

## 5. Scope and Limitations of Empirical Findings

> [!CAUTION]
> The error counts presented in Section 2 are specific to the 7-visit physical pilot dataset and the 10 engineered unit tests. They must **not** be interpreted as universal empirical error rates for arbitrary real-world environments.
>
> Generalization across arbitrary lighting, cluttered backgrounds, non-planar surfaces, and heavy 3D occlusions requires future multi-scene validation.
