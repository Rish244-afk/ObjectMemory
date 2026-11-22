# Physical Workbench Benchmark v1 Protocol & Data Capture Guide

**ObjectMemory Studio — Phase 14C**  
**Status**: Ingestion & Evaluation Infrastructure Active (`AWAITING_PHYSICAL_DATA`)  
**Standard Image Format**: JPG or PNG, nominal resolution $1920 \times 1080$.

---

## 1. Experimental Objectives

This benchmark evaluates the complete ObjectMemory pipeline on genuine photographic captures:
$$\text{Image Ingestion} \to \text{Detection} \to \text{Scene Registration} \to \text{Canonical Projection} \to \text{Association} \to \text{Temporal Reasoning} \to \text{Persistence}$$

The experimental goal is to empirically verify:
1. **Object Detection**: Accuracy of YOLOv8n and YOLO-World with letterbox preprocessing on real items.
2. **Co-Registration**: Planar homography estimation across visits under slight viewpoint and illumination changes.
3. **Cross-Visit Association**: Stable track maintenance without identity switches or track bifurcation.
4. **Epistemic State Transitions**: Correct detection of persistent relocations, vacancy confirmation delay (2-visit rule), and invariance to nuisance lighting or transient occlusion.

---

## 2. Standard Physical Setup & Equipment

### 2.1 Workbench Area
- **Surface**: Plain, matte workbench or table with moderate surface texture (e.g., light wood grain, cutting mat, or neutral desk pad) to provide natural SIFT/ORB features for feature-based registration.
- **Active Area**: Approximately $80\text{ cm} \times 50\text{ cm}$.

### 2.2 Calibration Fiducials (ArUco Markers)
- **Dictionary**: ArUco `DICT_4X4_50`.
- **Markers**: 4 corner markers with IDs `[0, 1, 2, 3]`:
  - `ID 0`: Top-Left corner
  - `ID 1`: Top-Right corner
  - `ID 2`: Bottom-Right corner
  - `ID 3`: Bottom-Left corner
- **Placement**: Taped flat to the perimeter corners of the active workspace. Markers must remain stationary across all baseline sequences.
- **Purpose**: Provides ground-truth homography registration benchmarks and reprojection error measurement independent of visual feature density.

### 2.3 Camera Configuration
- **Mount**: Rigid tripod or overhead boom arm looking down at an oblique angle ($45^\circ \text{--} 60^\circ$).
- **Camera Settings**:
  - Focus: Manual Focus (locked on center workspace; do NOT use continuous autofocus).
  - Exposure: Manual Exposure (shutter speed and aperture locked; do NOT use auto-exposure).
  - White Balance: Manual / Preset Daylight (locked; do NOT use auto-white-balance).
  - Resolution: $1920 \times 1080$ (Full HD) preferred.

---

## 3. Five Canonical Objects

| Object ID | Class Label | Suggested Item | Dimensions (Approx) |
| :--- | :--- | :--- | :--- |
| `cup_01` | `cup` | White ceramic coffee mug with handle | $8\text{ cm}$ diam, $10\text{ cm}$ height |
| `book_01` | `book` | Paperback novel (matte navy cover) | $13\text{ cm} \times 20\text{ cm}$ |
| `scissors_01` | `scissors` | Metal craft scissors with red handles | $18\text{ cm}$ length |
| `bottle_01` | `bottle` | Clear or stainless steel reusable bottle | $7\text{ cm}$ diam, $22\text{ cm}$ height |
| `phone_01` | `cell phone` | Dark smartphone in matte case | $7.5\text{ cm} \times 15\text{ cm}$ |

---

## 4. Seven Evaluation Scenarios (5 Visits Each)

Capture each scenario as a sequence of 5 visits. Save each captured image to `data/physical_workbench_v1/images/` using the canonical filename pattern:
`images/{sequence_id}_visit_{01..05}.jpg`

### S01 — Stationary Baseline
- **Visits 1 to 5**: All 5 objects placed at initial positions and left completely stationary.
- **Target Metrics**: 0 false movements, 0 false removals, 0 identity switches.

### S02 — Temporary Movement and Return
- **Visits 1 & 2**: Baseline positions.
- **Visit 3**: Move `cup_01` $>35\text{ px}$ to a temporary location.
- **Visit 4**: Return `cup_01` back to its exact baseline position.
- **Visit 5**: Leave `cup_01` stationary at restored baseline.
- **Target Metrics**: Detect `object_moved` at V3, detect `return_to_baseline` at V4, maintain single persistent track.

### S03 — Persistent Relocation
- **Visit 1**: Baseline positions.
- **Visit 2**: Move `phone_01` $>50\text{ px}$ to a new position.
- **Visits 3, 4, 5**: Leave `phone_01` stationary in its new position.
- **Target Metrics**: Transition to `CONFIRMED_MOVED` at V2, maintain single continuous track without creating duplicate entities.

### S04 — Permanent Removal
- **Visits 1 & 2**: Baseline positions.
- **Visit 3**: Physically remove `bottle_01` from the room.
- **Visits 4 & 5**: `bottle_01` remains absent.
- **Target Metrics**: Transition to `PENDING_REMOVAL` at V3, transition to `CONFIRMED_REMOVED` at V4 (verifying the 2-visit confirmation policy).

### S05 — Partial Occlusion
- **Visit 1**: Baseline positions.
- **Visit 2**: Place an opaque card over ~30% of `book_01`.
- **Visit 3**: Slide the card to cover ~70% of `book_01`.
- **Visit 4**: Remove the card completely.
- **Visit 5**: Stable unoccluded scene.
- **Target Metrics**: System must NOT trigger false removal or duplicate track; entity remains `CONFIRMED_PRESENT`.

### S06 — Illumination Stress
- **Visit 1**: Nominal room lighting (~450 lx).
- **Visit 2**: Dim room lights / pull blinds (~120 lx).
- **Visit 3**: Side desk lamp on (harsh directional shadow).
- **Visit 4**: Overhead direct spotlight.
- **Visit 5**: Restored nominal lighting.
- **Target Metrics**: 0 false movements despite strong shadow shifts.

### S07 — Viewpoint Perturbation
- **Visit 1**: Nominal camera pose.
- **Visit 2**: Small tripod pan (~$5^\circ$ left).
- **Visit 3**: Small tripod tilt (~$5^\circ$ downward).
- **Visit 4**: Small camera height adjustment ($\sim 5\text{--}10\text{ cm}$).
- **Visit 5**: Tripod returned to nominal position.
- **Target Metrics**: Planar homography success rate, registration inlier ratio, and degradation boundaries.

---

## 5. Ingestion and Evaluation Commands

1. **Validate Dataset Schema and Image Completeness**:
   ```powershell
   python scripts/validate_physical_workbench_dataset.py --dataset-dir data/physical_workbench_v1
   ```

2. **Run Decoupled 3-Track Benchmark**:
   ```powershell
   python scripts/evaluate_physical_workbench.py --dataset-dir data/physical_workbench_v1 --output-file scripts/phase14c_physical_workbench_results.json
   ```

3. **Overlay Bounding Box Inspection** (Visual Audit Tool):
   ```powershell
   python scripts/validate_physical_workbench_dataset.py --dataset-dir data/physical_workbench_v1 --overlay-dir data/physical_workbench_v1/overlays
   ```
