# Real-World Dataset Collection and Annotation Protocol

**Official project title:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes

**Official description:** "An AI-based visual memory system that compares scenes over time to detect and explain persistent changes in object locations while filtering out temporary variations such as lighting, occlusion, and viewpoint changes."

---

## 1. Overview and Scientific Objectives

This protocol governs the collection and ground-truth annotation of physical photographs for evaluating `ObjectMemory`. 

To prevent confounding variables from masking or falsely creating state transitions, this protocol explicitly **decouples isolated experimental factors** across controlled multi-visit photographic sequences on a planar work surface.

### Crucial Architectural Reality: Object Classes
The current object detector is a pretrained **YOLOv8 Nano model** running via OpenCV DNN CPU inference, trained on the **COCO 80-class taxonomy**.
- **Important:** COCO does **not** have classes for electronics tools such as `multimeter`, `soldering iron`, or `wire stripper`.
- For genuine physical evaluation without custom detector re-training, choose physical objects corresponding to **explicitly supported COCO classes**:
  1. **`cup`** (e.g. coffee mug, teacup)
  2. **`bottle`** (e.g. spray bottle, water bottle, cleaning solvent container)
  3. **`scissors`** (e.g. standard workbench scissors or shears)
  - Other compatible everyday workbench items: `mouse`, `keyboard`, `book`, `clock`, `cell phone`.

---

## 2. Decoupled Controlled Experiments

Rather than combining relocation, occlusion, removal, camera tilt, and lighting shift in a single visit, physical testing should follow these isolated test procedures:

### Experiment A: Baseline & Unchanged-Scene Control
- **Purpose:** Quantify false-positive change detection rate, false relocation rate, and detection stability under zero physical change.
- **Protocol:**
  - Place 3 distinct objects (e.g., `cup`, `bottle`, `scissors`) on the textured workbench surface.
  - Photograph Visit 1 ($V_1$).
  - Without moving the camera, changing lighting, or touching any object, photograph Visit 2 ($V_2$).
- **Expected Physical Observation:** All objects `VISIBLE` with `annotated_event = UNCHANGED`.
- **Expected System State:** All objects remain `CONFIRMED_PRESENT`.
- **Failure Criterion:** Any entity transitions to `CONFIRMED_MOVED`, `PENDING_REMOVAL`, or `CONFIRMED_REMOVED`.

---

### Experiment B: Pure Spatial Relocation (Constant Camera & Lighting)
- **Purpose:** Test spatial displacement threshold ($\Delta p > 35\text{px}$) and canonical relocation tracking without confounding viewpoint or photometric changes.
- **Protocol:**
  - **Fixed:** Camera mounted on a tripod or fixed stand; constant ambient room lighting.
  - **Changed:** Move `cup` by at least $10\text{--}15\text{ cm}$ (guaranteeing $>50\text{px}$ displacement in a $1920\times1080$ frame). Leave `bottle` and `scissors` stationary.
- **Sequence:**
  - $V_1$: Initial baseline.
  - $V_2$: `cup` relocated.
  - $V_3$: `cup` remains in its new relocated position.
- **Expected State Transitions:**
  - `cup`: $V_1 \to \text{CONFIRMED\_PRESENT}$, $V_2 \to \text{CONFIRMED\_MOVED}$, $V_3 \to \text{CONFIRMED\_MOVED}$.
  - Stationary objects: remain `CONFIRMED_PRESENT`.
- **Failure Criterion:**
  - `cup` does not transition to `CONFIRMED_MOVED` at $V_2$ (false negative relocation).
  - A stationary object transitions to `CONFIRMED_MOVED` (false relocation).
  - `cup` identity is lost and treated as a new object while old position is marked removed (broken identity association).

---

### Experiment C: Temporary Partial Occlusion Followed by Uncovering
- **Purpose:** Verify that an unobservable object does **not** accumulate removal evidence and maintains `CONFIRMED_PRESENT`.
- **Protocol:**
  - **Fixed:** Camera viewpoint, illumination, object positions.
  - **Changed:** In $V_2$, partially or completely cover `scissors` with a notebook, clipboard, or cardboard sheet. In $V_3$, remove the cover.
- **Sequence:**
  - $V_1$: `scissors` visible (`CONFIRMED_PRESENT`).
  - $V_2$: `scissors` covered (`physical_observation = PARTIALLY_OCCLUDED` or `FULLY_OCCLUDED`, `bbox = null`, `expected_state = CONFIRMED_PRESENT`).
  - $V_3$: `scissors` uncovered (`physical_observation = VISIBLE`, `expected_state = CONFIRMED_PRESENT`).
- **Failure Criterion:**
  - `scissors` transitions to `PENDING_REMOVAL` or `CONFIRMED_REMOVED` during $V_2$ (failure to gate removal on observability).

---

### Experiment D: Consecutive Absence Followed by Reappearance
- **Purpose:** Validate the two-visit vacancy streak requirement for `CONFIRMED_REMOVED` and recovery upon return.
- **Protocol:**
  - **Fixed:** Camera viewpoint, lighting.
  - **Changed:** Remove `bottle` from the workbench during $V_2$ and $V_3$. Return `bottle` in $V_4$.
- **Sequence:**
  - $V_1$: `bottle` visible (`CONFIRMED_PRESENT`).
  - $V_2$: `bottle` absent (1st reliable absence; `expected_state = PENDING_REMOVAL`).
  - $V_3$: `bottle` absent (2nd consecutive reliable absence; `expected_state = CONFIRMED_REMOVED`).
  - $V_4$: `bottle` returned (`expected_state = CONFIRMED_PRESENT`).
- **Failure Criterion:**
  - Declaring `CONFIRMED_REMOVED` prematurely at $V_2$ (streak = 1).
  - Failing to confirm removal at $V_3$ (streak = 2).
  - Failing to restore `CONFIRMED_PRESENT` at $V_4$.

---

### Experiment E: Photometric Illumination Variation (Positions Fixed)
- **Purpose:** Validate that moderate illumination shifts do not trigger false state changes.
- **Protocol:**
  - **Fixed:** All object positions; camera viewpoint.
  - **Changed:** Dim or brighten room lighting ($\approx \pm 20\%$) across visits.
- **Expected System State:** All objects remain `CONFIRMED_PRESENT`.
- **Failure Criterion:** Color histogram shift degrades appearance affinity enough to break tracking or trigger false removal.

---

### Experiment F: Camera / Viewpoint Variation (Positions Fixed)
- **Purpose:** Verify planar homography co-registration and canonical coordinate mapping under perspective shift.
- **Protocol:**
  - **Fixed:** All physical object positions; constant room illumination.
  - **Changed:** Shift handheld camera position ($5^\circ\text{--}15^\circ$ angle shift, small translation).
  - **Requirement:** Work surface must contain rich planar texture (e.g. cutting mat grid, printed circuit board, wood grain) ensuring $\ge 12$ RANSAC inliers and inlier ratio $\ge 0.20$.
- **Expected System State:** Homography accurately aligns frames; canonical coordinates do not shift by $>35\text{px}$; all objects remain `CONFIRMED_PRESENT`.
- **Failure Criterion:** Registration failure triggers spatial gating, or inaccurate homography causes static objects to shift $>35\text{px}$ in canonical coordinates and falsely declare `CONFIRMED_MOVED`.

---

## 3. Directory Structure & File Formats

Store sequences under `data/real_workbench/` (or dedicated subdirectories):

```text
data/real_workbench/
├── manifest.json
└── visits/
    ├── visit_01.jpg
    ├── visit_02.jpg
    ├── visit_03.jpg
    └── visit_04.jpg
```

- **Format:** JPEG or PNG (`.jpg`, `.png`).
- **Resolution:** $1280\times720$ to $1920\times1080$ recommended.
- **Path convention:** Image paths in `manifest.json` are relative to the directory containing `manifest.json`.

---

## 4. Manifest Schema & Annotation Rules

The manifest file `manifest.json` is validated by `RealWorkbenchDataset.validate_dataset_integrity()`:

### Manifest Fields
- `dataset_name` (`string`): Unique sequence name.
- `description` (`string`): Sequence description.
- `is_template` (`boolean`): Set to `true` **only** for unpopulated templates. Must be omitted or `false` for real evaluations.
- `metadata` (`object`): Sensor, lighting, resolution, and coordinate conventions.
- `visits` (`array`): Chronologically ordered visit list ($\ge 2$ visits required).

### Visit Frame Fields
- `visit_id` (`string`): Unique visit identifier. Duplicate IDs will cause a validation failure.
- `session_name` (`string`): Descriptive session title.
- `timestamp` (`string`): Strictly monotonically increasing ISO 8601 UTC timestamp (e.g. `"2026-03-01T10:00:00Z"`).
- `image_path` (`string`): Relative path to photograph. Must point to an existing, readable image.
- `annotations` (`array`): Ground-truth physical entity annotations.

### Annotation Object Fields & Integrity Invariants
- `entity_id` (`string`): Persistent entity identifier (e.g. `"cup_01"`). Must remain identical across visits for the same physical object.
- `entity_label` (`string`): Must be a valid supported class from COCO (e.g. `"cup"`, `"bottle"`, `"scissors"`). Must be consistent across all visits for the same `entity_id`.
- `bbox` (`[x_min, y_min, x_max, y_max]` or `null`):
  - Top-left origin $(0, 0)$ in raw image pixel coordinates.
  - **Invariants:**
    - If `physical_observation == "VISIBLE"`, `bbox` **must not be null**, must satisfy $x_{min} < x_{max}$ and $y_{min} < y_{max}$, and must be within image dimensions.
    - If `physical_observation == "ABSENT"`, `bbox` **must be null**.
    - If `annotated_event == "ABSENT"`, `physical_observation` **must be `"ABSENT"`** and `bbox` **must be null**.
- `physical_observation` (`enum`): `VISIBLE`, `PARTIALLY_OCCLUDED`, `FULLY_OCCLUDED`, `OUT_OF_FRAME`, `ABSENT`.
- `annotated_event` (`enum`): `UNCHANGED`, `RELOCATED`, `ABSENT`, `NEWLY_APPEARED`.
- `expected_state` (`enum`): `CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`, `AMBIGUOUS_IDENTITY`.

---

## 5. Execution & Safeguards

1. **Evaluation Blocking:** If `is_template: true` or image paths contain `"placeholder"`, the pipeline CLI blocks with `BLOCKED_PENDING_ANNOTATIONS`.
2. **Ground-Truth Isolation:** Annotations are never passed to the detector, registrar, or temporal engine. They are solely evaluated post-hoc by `RealBenchmarkEvaluator`.
3. **Execution Command:**
   ```bash
   python tools/run_real_pipeline.py
   ```
