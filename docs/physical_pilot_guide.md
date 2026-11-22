# ObjectMemory Physical Pilot Guide

**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Purpose:** Exact manifest schema, supported labels, and data-collection checklist for the first genuine physical pilot  
**Sources inspected:** `src/objectmemory/datasets/real_dataset.py`, `src/objectmemory/benchmarks/real_evaluator.py`, `tools/run_real_pipeline.py`

---

## 1. Manifest Format — Exact Required Fields

The pipeline reads `data/real_workbench/manifest.json`.  
The loader is [`RealWorkbenchDataset.load_from_directory`](../src/objectmemory/datasets/real_dataset.py#L314-L371).

> **Important:** The loader parses timestamps and bounding boxes during loading, before any template or readiness check. Timestamps must be valid ISO 8601 and bboxes must be numeric arrays even in a template file.

### Top-level keys

| Key | Required | Type | Notes |
| :--- | :--- | :--- | :--- |
| `dataset_name` | Yes | string | Used in report headers |
| `description` | No | string | Free text; not evaluated |
| `is_template` | **Yes** | boolean | **Must be `false` for real data.** Any `true` value blocks evaluation immediately. |
| `metadata` | No | object | Informational only; not parsed by evaluator |
| `visits` | Yes | array | Ordered list of visit objects (min 2) |

### Per-visit object

| Key | Required | Type | Notes |
| :--- | :--- | :--- | :--- |
| `visit_id` | Yes | string | Unique across all visits. Duplicate IDs raise `ValueError`. |
| `timestamp` | Yes | string | ISO 8601 UTC, e.g. `"2026-10-15T09:00:00Z"`. Each visit must be strictly later than the previous. |
| `image_path` | Yes | string | Path relative to the manifest directory. File must exist and decode via OpenCV. Must **not** contain the word `"placeholder"`. |
| `session_name` | **No** | string | Defaults to `visit_id` if omitted. |
| `annotations` | Yes (for evaluation) | array | At least one visit must have annotations. |

### Per-annotation object

| Key | Required | Type | Notes |
| :--- | :--- | :--- | :--- |
| `entity_id` | Yes | string | Stable across all visits. The same physical object must use the same `entity_id` every visit, including after reappearance. Defaults to `entity_label` if omitted. |
| `entity_label` | Yes | string | Exact COCO class name (see §2). Must be the same value for the same `entity_id` in every visit. |
| `bbox` | Conditional | `[x_min, y_min, x_max, y_max]` or `null` | See §3 for exact rules per observation type. |
| `physical_observation` | Yes | string | One of five values (§3). |
| `annotated_event` | No | string or null | One of four values (§3). Recommended. |
| `expected_state` | No | string or null | One of six values (§3). Strongly recommended — see §4. |
| `notes` | No | string | Free text for your records. |

---

## 2. Supported Detector Labels

The model is **YOLOv8 nano** trained on **COCO 80 classes**.  
Source: `COCO_CLASSES_80` in [`src/objectmemory/detection/detector.py`](../src/objectmemory/detection/detector.py).

**`entity_label` must exactly match one of these strings** (case-sensitive, including spaces):

```
 0  person          20  elephant        40  wine glass      60  dining table
 1  bicycle         21  bear            41  cup             61  toilet
 2  car             22  zebra           42  fork            62  tv
 3  motorcycle      23  giraffe         43  knife           63  laptop
 4  airplane        24  backpack        44  spoon           64  mouse
 5  bus             25  umbrella        45  bowl            65  remote
 6  train           26  handbag         46  banana          66  keyboard
 7  truck           27  tie             47  apple           67  cell phone
 8  boat            28  suitcase        48  sandwich        68  microwave
 9  traffic light   29  frisbee         49  orange          69  oven
10  fire hydrant    30  skis            50  broccoli        70  toaster
11  stop sign       31  snowboard       51  carrot          71  sink
12  parking meter   32  sports ball     52  hot dog         72  refrigerator
13  bench           33  kite            53  pizza           73  book
14  bird            34  baseball bat    54  donut           74  clock
15  cat             35  baseball glove  55  cake            75  vase
16  dog             36  skateboard      56  chair           76  scissors
17  horse           37  surfboard       57  couch           77  teddy bear
18  sheep           38  tennis racket   58  potted plant    78  hair drier
19  cow             39  bottle          59  bed             79  toothbrush
```

**Practical objects for a desk or workbench pilot:**  
`bottle`, `cup`, `scissors`, `book`, `laptop`, `keyboard`, `mouse`, `cell phone`, `remote`, `clock`, `vase`, `bowl`, `backpack`

> **Not in COCO (detector will not reliably fire):** multimeter, oscilloscope, soldering iron, PCB, screwdriver, pliers, breadboard. Do not use these as `entity_label` values.

---

## 3. Observation Types, Bounding Boxes, and Scoring

### `physical_observation` — your independent observation of physical reality

| Value | Physical meaning | `bbox` | How evaluator scores this visit |
| :--- | :--- | :--- | :--- |
| `VISIBLE` | Object fully visible and unobstructed | **Required** — non-null. Enclose the full object extent. | Normal accuracy check: system state vs. `expected_state`. |
| `PARTIALLY_OCCLUDED` | Partially hidden, but you can see part of it and know it is present | **Required** — non-null. Enclose **only the visible portion**. | Normal accuracy check. |
| `FULLY_OCCLUDED` | Entirely hidden (e.g. under a cloth), but you know it is present | **Must be null.** | State is **frozen at prior state** (no removal evidence accrues). See §4. |
| `OUT_OF_FRAME` | Outside camera field of view, but physically present | **Must be null.** | State is **frozen at prior state** (no removal evidence accrues). |
| `ABSENT` | Genuinely removed from the scene | **Must be null.** | Absence evidence accrues. See §4 for progression rules. |

> **Validator enforcement:** `ABSENT`, `FULLY_OCCLUDED`, and `OUT_OF_FRAME` with a non-null `bbox` are rejected with a descriptive error. `VISIBLE` with a null `bbox` is also rejected. `PARTIALLY_OCCLUDED` accepts either null or non-null — but providing the visible-portion box is strongly recommended, as it enables detection matching.

### `annotated_event` — the cross-visit change you observed

| Value | Use when |
| :--- | :--- |
| `UNCHANGED` | Object is in the same position as the previous visit |
| `RELOCATED` | Object was physically moved to a different position |
| `NEWLY_APPEARED` | Object was absent or unknown, now present (use with `VISIBLE`) |
| `ABSENT` | Object was present, now absent (must pair with `physical_observation: ABSENT`) |

---

## 4. Occlusion and the Expected-State Policy

> This section is the most important for correct annotation. Read it carefully.

### The evaluator's derivation rule (source: `real_evaluator.py` line 434)

When `expected_state` is **omitted**, the evaluator derives the expected state from `physical_observation` using this causal policy:

```
PARTIALLY_OCCLUDED / FULLY_OCCLUDED / OUT_OF_FRAME
   → FREEZE: expected state = prior state, streak unchanged
                                           (removal evidence does NOT accumulate)

VISIBLE / RELOCATED / NEWLY_APPEARED
   → expected state = CONFIRMED_PRESENT (or CONFIRMED_MOVED if RELOCATED)
     streak reset to 0

ABSENT
   → streak += 1
     streak == 1 (from CONFIRMED_PRESENT/CONFIRMED_MOVED) → PENDING_REMOVAL
     streak >= 2 (from PENDING_REMOVAL)                   → CONFIRMED_REMOVED
```

### Two distinct guarantees — and why they are different

| Guarantee | What it means | When it holds |
| :--- | :--- | :--- |
| **Prevents false CONFIRMED_REMOVED** | The system will not advance the removal streak or produce CONFIRMED_REMOVED for an occluded object. | Guaranteed by both the engine's IoU-based occlusion detection and the evaluator's policy freeze for FULLY_OCCLUDED/PARTIALLY_OCCLUDED/OUT_OF_FRAME. |
| **Retains CONFIRMED_PRESENT** | The system actively asserts the object is present. | **Only holds if the prior state was already CONFIRMED_PRESENT.** The freeze preserves the prior — it does not upgrade from INSUFFICIENT_EVIDENCE to CONFIRMED_PRESENT. |

### The first-visit occlusion case

If an object is `FULLY_OCCLUDED` or `OUT_OF_FRAME` in **Visit 1** (the first time it appears in the sequence), the prior state is `INSUFFICIENT_EVIDENCE`. The evaluator's freeze produces `INSUFFICIENT_EVIDENCE`, not `CONFIRMED_PRESENT`. In this case, omitting `expected_state` will cause the evaluator to score the system against `INSUFFICIENT_EVIDENCE`.

**If you know the object is present and you want the system to be scored as correct for holding `CONFIRMED_PRESENT`, provide `expected_state: "CONFIRMED_PRESENT"` explicitly.** When `expected_state` is present, the evaluator uses it directly and skips derivation.

### Concrete example: bottle occluded in V2, having been confirmed present in V1

| Visit | physical_observation | prior state | Derived expected_state (if omitted) | Explicit expected_state (recommended) |
| :--- | :--- | :--- | :--- | :--- |
| V1 | VISIBLE | INSUFFICIENT_EVIDENCE | CONFIRMED_PRESENT | `CONFIRMED_PRESENT` |
| V2 | FULLY_OCCLUDED | CONFIRMED_PRESENT | **CONFIRMED_PRESENT** (frozen) | `CONFIRMED_PRESENT` |
| V3 | ABSENT | CONFIRMED_PRESENT (from V2) | PENDING_REMOVAL | `PENDING_REMOVAL` |
| V4 | ABSENT | PENDING_REMOVAL (from V3) | CONFIRMED_REMOVED | `CONFIRMED_REMOVED` |

> In V2, the frozen derived state happens to equal the correct ground truth because V1 produced `CONFIRMED_PRESENT`. The system will not be penalised for `CONFIRMED_REMOVED` (the freeze prevents that). However, it **could** be penalised if it actually produces `PENDING_REMOVAL` in V2 — the freeze in the policy says `expected = CONFIRMED_PRESENT`, not `expected = anything-but-CONFIRMED_REMOVED`. Specifying `expected_state` explicitly makes this comparison unambiguous.

### Practical rule for annotation

| Situation | What to write |
| :--- | :--- |
| Object visible at this visit | `"expected_state": "CONFIRMED_PRESENT"` (or `"CONFIRMED_MOVED"` if relocated) |
| Object occluded but was CONFIRMED_PRESENT last visit | `"expected_state": "CONFIRMED_PRESENT"` — write it explicitly |
| Object occluded but was CONFIRMED_MOVED last visit | `"expected_state": "CONFIRMED_MOVED"` — write it explicitly |
| First absence after being present | `"expected_state": "PENDING_REMOVAL"` |
| Second+ consecutive absence | `"expected_state": "CONFIRMED_REMOVED"` |
| Object reappears after removal | `"expected_state": "CONFIRMED_PRESENT"` |

---

## 5. Persistent Identity Across Visits — Including Reappearance

The `entity_id` field is the sole identifier the evaluator uses to track a physical object across visits. It is **not reset or changed** when an object disappears and returns.

- Use the same `entity_id` (e.g. `"bottle_01"`) in every visit where that physical object is annotated, including after it reappears.
- The `annotated_event` value `"NEWLY_APPEARED"` describes the physical event (absent → present), not an identity change.
- Using the same `entity_id` with `NEWLY_APPEARED` is fully compatible with the parser (`RealVisitAnnotation.from_dict`, line 126) and the evaluator's entity-mapping logic (line 508).

**Example: bottle disappears at V3, returns at V5**

```json
{ "entity_id": "bottle_01", "annotated_event": "ABSENT",        ... }  // V3
{ "entity_id": "bottle_01", "annotated_event": "ABSENT",        ... }  // V4
{ "entity_id": "bottle_01", "annotated_event": "NEWLY_APPEARED",... }  // V5 — same ID
```

---

## 6. Validator Safeguards — What the System Checks

Source: [`validate_dataset_integrity()`](../src/objectmemory/datasets/real_dataset.py#L187-L282) and [`is_evaluation_ready()`](../src/objectmemory/datasets/real_dataset.py#L284-L310).

The following always block evaluation with an explicit reason:

- `"is_template": true` → immediate block
- `"placeholder"` anywhere in an `image_path` → immediate block
- Fewer than 2 visits
- Empty or missing `visit_id`
- Non-monotonic or identical timestamps between consecutive visits
- `image_path` does not exist on disk
- Image file cannot be decoded by OpenCV
- `VISIBLE` with `bbox: null`
- `ABSENT`, `FULLY_OCCLUDED`, or `OUT_OF_FRAME` with a non-null `bbox`
- `annotated_event: ABSENT` with `physical_observation` ≠ `ABSENT`
- Degenerate box: `x_max ≤ x_min` or `y_max ≤ y_min`
- Box coordinates exceed actual image pixel dimensions
- Same `entity_id` with two different `entity_label` values across visits
- Duplicate `visit_id` values

**No accuracy, precision, recall, or F1 numbers are reported while evaluation is blocked.**

---

## 7. Complete 5-Visit JSON Template (Copy and Fill In)

This is the file at [`data/real_workbench/manifest.json`](../data/real_workbench/manifest.json).  
Replace every `REPLACE:` value. Replace `0, 0, 1, 1` bboxes with real pixel measurements.  
Change `"is_template": true` to `"is_template": false` and rename image paths when ready.

The template covers all five evaluation scenarios:  
**control (unchanged) · relocation · temporary full occlusion · consecutive absence (removal) · reappearance**

```json
{
  "dataset_name": "pilot_01",
  "description": "Five-visit physical workbench sequence — replace all REPLACE: values",
  "is_template": true,
  "metadata": {
    "sensor": "REPLACE: e.g. iPhone 15 rear camera, fixed overhead tripod",
    "lighting_conditions": "REPLACE: e.g. ceiling LED, blinds closed",
    "resolution": [1920, 1080],
    "coordinate_convention": "[x_min, y_min, x_max, y_max] raw pixels, (0,0) = top-left"
  },
  "visits": [
    {
      "visit_id": "visit_01",
      "session_name": "Baseline — all objects visible",
      "timestamp": "2026-10-15T09:00:00Z",
      "image_path": "visits/visit_01.jpg",
      "annotations": [
        {
          "entity_id": "cup_01",
          "entity_label": "cup",
          "bbox": [120, 80, 220, 200],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Control — does not move"
        },
        {
          "entity_id": "scissors_01",
          "entity_label": "scissors",
          "bbox": [340, 150, 440, 270],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Will be relocated at visit_02"
        },
        {
          "entity_id": "bottle_01",
          "entity_label": "bottle",
          "bbox": [560, 100, 620, 310],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Will be removed at visit_03"
        }
      ]
    },
    {
      "visit_id": "visit_02",
      "session_name": "Scissors relocated; bottle fully occluded; cup unchanged",
      "timestamp": "2026-10-15T10:00:00Z",
      "image_path": "visits/visit_02.jpg",
      "annotations": [
        {
          "entity_id": "cup_01",
          "entity_label": "cup",
          "bbox": [120, 80, 220, 200],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Unchanged control"
        },
        {
          "entity_id": "scissors_01",
          "entity_label": "scissors",
          "bbox": [700, 300, 800, 420],
          "physical_observation": "VISIBLE",
          "annotated_event": "RELOCATED",
          "expected_state": "CONFIRMED_MOVED",
          "notes": "Moved to new position (≥15 cm from V1 position). Measure new bbox."
        },
        {
          "entity_id": "bottle_01",
          "entity_label": "bottle",
          "bbox": null,
          "physical_observation": "FULLY_OCCLUDED",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Covered by clipboard. bbox MUST be null. Prior was CONFIRMED_PRESENT so freeze produces CONFIRMED_PRESENT. Specifying expected_state explicitly prevents silent misclassification."
        }
      ]
    },
    {
      "visit_id": "visit_03",
      "session_name": "Bottle uncovered and removed (first absence); scissors stationary; cup unchanged",
      "timestamp": "2026-10-15T11:00:00Z",
      "image_path": "visits/visit_03.jpg",
      "annotations": [
        {
          "entity_id": "cup_01",
          "entity_label": "cup",
          "bbox": [120, 80, 220, 200],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Unchanged control"
        },
        {
          "entity_id": "scissors_01",
          "entity_label": "scissors",
          "bbox": [700, 300, 800, 420],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_MOVED",
          "notes": "Same new position from V2. expected_state is CONFIRMED_MOVED, not CONFIRMED_PRESENT."
        },
        {
          "entity_id": "bottle_01",
          "entity_label": "bottle",
          "bbox": null,
          "physical_observation": "ABSENT",
          "annotated_event": "ABSENT",
          "expected_state": "PENDING_REMOVAL",
          "notes": "First reliable absence (streak=1). bbox MUST be null. System: CONFIRMED_PRESENT → PENDING_REMOVAL."
        }
      ]
    },
    {
      "visit_id": "visit_04",
      "session_name": "Bottle still absent — removal confirmed; scissors and cup unchanged",
      "timestamp": "2026-10-15T12:00:00Z",
      "image_path": "visits/visit_04.jpg",
      "annotations": [
        {
          "entity_id": "cup_01",
          "entity_label": "cup",
          "bbox": [120, 80, 220, 200],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Unchanged control"
        },
        {
          "entity_id": "scissors_01",
          "entity_label": "scissors",
          "bbox": [700, 300, 800, 420],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_MOVED",
          "notes": "Same new position from V2."
        },
        {
          "entity_id": "bottle_01",
          "entity_label": "bottle",
          "bbox": null,
          "physical_observation": "ABSENT",
          "annotated_event": "ABSENT",
          "expected_state": "CONFIRMED_REMOVED",
          "notes": "Second consecutive absence (streak=2). bbox MUST be null. System: PENDING_REMOVAL → CONFIRMED_REMOVED."
        }
      ]
    },
    {
      "visit_id": "visit_05",
      "session_name": "Bottle returned — reappearance",
      "timestamp": "2026-10-15T13:00:00Z",
      "image_path": "visits/visit_05.jpg",
      "annotations": [
        {
          "entity_id": "cup_01",
          "entity_label": "cup",
          "bbox": [120, 80, 220, 200],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Unchanged control"
        },
        {
          "entity_id": "scissors_01",
          "entity_label": "scissors",
          "bbox": [700, 300, 800, 420],
          "physical_observation": "VISIBLE",
          "annotated_event": "UNCHANGED",
          "expected_state": "CONFIRMED_MOVED",
          "notes": "Same new position from V2."
        },
        {
          "entity_id": "bottle_01",
          "entity_label": "bottle",
          "bbox": [560, 100, 620, 310],
          "physical_observation": "VISIBLE",
          "annotated_event": "NEWLY_APPEARED",
          "expected_state": "CONFIRMED_PRESENT",
          "notes": "Same entity_id 'bottle_01' as all previous visits. Identity is preserved by entity_id across the full sequence. NEWLY_APPEARED is the annotated_event — it does not create a new entity. System: CONFIRMED_REMOVED → CONFIRMED_PRESENT."
        }
      ]
    }
  ]
}
```

> **Before you run evaluation:**  
> 1. Replace all bbox coordinates with real pixel measurements from your photographs.  
> 2. Replace all `image_path` values with your actual filenames.  
> 3. Replace timestamps with the real times you took each photograph.  
> 4. Set `"is_template": false`.

---

## 8. Physical Capture Checklist

### Camera setup (once, before all visits)

- [ ] Mount the camera on a **fixed tripod or overhead rig**. Do not move it between visits.
- [ ] Confirm the **full workbench surface is in frame** with margin to spare.
- [ ] Confirm **no tracked object is clipped** by the frame boundary.
- [ ] **Lock focal length and exposure** (on a smartphone: tap-to-lock focus and brightness).
- [ ] Use **identical lighting** for every visit (same bulbs, same blinds setting).
- [ ] Photograph a ruler or tape measure in the first frame to understand scale (optional but useful).
- [ ] Note the image resolution (`width × height` pixels) — you need this to verify bbox bounds.

### Scenario sequence (5 visits)

| Visit | Physical action | Object roles |
| :--- | :--- | :--- |
| **V1 — Baseline** | Place all objects; photograph | cup=control, scissors=stationary, bottle=stationary |
| **V2 — Relocation + occlusion** | Move scissors ≥15 cm; cover bottle with clipboard | scissors=RELOCATED, bottle=FULLY_OCCLUDED, cup=UNCHANGED |
| **V3 — First absence** | Remove clipboard; remove bottle from room | scissors=UNCHANGED at new spot, bottle=ABSENT (streak 1), cup=UNCHANGED |
| **V4 — Confirmed removal** | Do nothing; bottle still absent | scissors=UNCHANGED, bottle=ABSENT (streak 2), cup=UNCHANGED |
| **V5 — Reappearance** | Return bottle to workbench | scissors=UNCHANGED, bottle=NEWLY_APPEARED, cup=UNCHANGED |

### Per-visit annotation steps (after all photographs are taken)

1. Open each image in a tool that shows pixel coordinates (GIMP, ImageJ, Paint with ruler).
2. For each visible or partially occluded object, record exact `[x_min, y_min, x_max, y_max]` from actual pixel measurements.
3. Do not estimate — measure from the image file.
4. For FULLY_OCCLUDED / OUT_OF_FRAME / ABSENT objects: write `"bbox": null`.
5. Fill in the manifest JSON with real timestamps, filenames, and measured coordinates.
6. Set `"is_template": false` and remove the word `"placeholder"` from all `image_path` values.

---

## 9. Actual CLI Output — Template Dataset

Command executed (exit code 0):

```
python tools/run_real_pipeline.py
```

Actual output with the current template manifest (`is_template: true`, placeholder image paths):

```
======================================================================================
 ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes
 Phase 6: Real-Image Validation and Detector Integration
======================================================================================

[1/4] Initializing CPU Object Detector...
      Model:        OpenCVDNN_YOLOV8_ONNX
      Backend:      OpenCV DNN (CPU)
      Classes:      80 classes
      Weights:      models\yolov8n.onnx (12.21 MB)

[2/4] Loaded real-image dataset: 'TEMPLATE_pilot_01'
      Visits found: 5
      Evaluation status: BLOCKED_PENDING_ANNOTATIONS
      Status note: Dataset contains placeholder image paths or template annotations.
                   Evaluation is blocked until real physical photographs and manual
                   annotations replace the template placeholders.

[3/4] Processing Imagery & Profiling CPU Performance...
      No physical dataset frames found; executing CPU inference validation on
      'demo_alignment_overlay.png'...
      -> Detections found: 17 (all "sports ball" — synthetic diagnostic image)
      -> CPU Inference Latency: 128.3 ms

[4/4] Real-Image Evaluation Status:
======================================================================================
 STATUS: BLOCKED_PENDING_ANNOTATIONS
 Reason: Real workbench dataset does not contain physical multi-visit images and
         manual ground-truth annotations.
======================================================================================
```

> The `WARN: Targets are not supported by the new graph engine for now` message is an OpenCV DNN stderr notice; it does not affect functionality.  
> The 128.3 ms latency figure is from a **synthetic diagnostic image** (`demo_alignment_overlay.png`), not a real photograph.  
> **No accuracy, precision, recall, or F1 numbers appear** when the dataset is blocked.

---

## 10. Known Limitations Before Physical Data Exists

| Limitation | Detail |
| :--- | :--- |
| **35 px movement threshold not physically calibrated** | Engineering parameter; tested only on synthetic black frames with mock homographies. Measure actual registration residuals on real photos before adjusting. |
| **Registration not validated on real images** | SIFT/ORB inlier ratio threshold (0.35) and homography estimation tested only on synthetic frames. |
| **Detector does not cover workbench tools** | COCO 80 classes do not include multimeter, soldering iron, PCB, screwdriver, etc. |
| **Detector uses direct aspect-ratio stretch** | Input resized to 640×640 regardless of aspect ratio; annotated bboxes for non-square images will scale anisotropically. |
| **Latency figure is from a synthetic image** | The 128 ms CPU inference figure was measured on `demo_alignment_overlay.png`. Real photographs at different resolutions will produce different figures. |
| **No physical accuracy reported yet** | `accuracy`, `removal_recall`, `false_confirmed_removal_rate` will be populated for the first time from pilot data. |
