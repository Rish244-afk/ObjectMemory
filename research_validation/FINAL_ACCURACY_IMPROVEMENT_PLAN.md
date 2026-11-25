# ObjectMemory — Final Accuracy Improvement Feasibility Plan

**Dataset:** `data/physical_pilot_v2/manifest_v2.json` (30 observations, 26 byte-unique images, 183 annotations, 177 visible GT boxes)  
**Frozen Baselines:** `YOLOv8n B3` (`30.60%`, `56/183`), `YOLOWorld 7-Class B3` (`31.69%`, `58/183`), `Track B Oracle` (`76.50%`, `140/183`)

---

## 1. Exact Cause of Current Low Accuracy

Per `research_validation/detection_failure_analysis/DETECTION_FAILURE_ANALYSIS_REPORT.md`:
- **Registration & Association are NOT bottlenecks:** `24/24` query frames registered (`AlignmentStatus.SUCCESS`), and `id_switches = 0` across all runs.
- **Temporal Reasoning works when detections exist:** Track B Oracle achieved `140/183` (**76.50%**), including `24/24` (**100%**) on `V02`–`V05` for `stationary`, `occlusion`, and `return_to_baseline` (36 of the 43 Track B mismatches are Visit-1 `INSUFFICIENT_EVIDENCE` initialization).
- **Overhead Top-Down Detection Recall is the primary bottleneck:**
  - `YOLOv8n` (`conf=0.25`): `48/177` (27.12%) recall — detects `scissors` (`30/30`) and `banana` (`3/3`), but misses `book` (`0/28`), `bottle` (`0/26`), `cell phone` (`1/30`), and drops `potted plant` (`9/30`) and `cup` (`5/30`), causing 6 false removals (`FDP=100%`).
  - `YOLOWorld` (`7 classes`, `conf=0.15`): `54/177` (30.51%) recall — detects `cell phone` (`29/30`) and `bottle` (`22/26`), but misses `book` (`0/28`), `scissors` (`0/30`), `cup` (`0/30`), `banana` (`0/3`, `max conf 0.1378`), and drops `potted plant` (`3/30`).

---

## 2. Feasibility of Fine-Tuning on the Existing Dataset

- **Environment status:** `torch`, `ultralytics`, and `onnx` export packages are **not installed** in the current Python `3.11.9` environment (only `onnxruntime 1.31.0` and `scikit-learn 1.9.1` are present).
- **Dataset sufficiency:** `physical_pilot_v2` contains **30 images (only 26 byte-unique frames)** of **1 physical desk, 1 lighting condition, and 1 physical instance per object class**, and `banana` appears in only **3 frames of 1 scenario** (`addition/V03`–`V05`).
- **Verdict:** Fine-tuning on `physical_pilot_v2` alone is **not scientifically sufficient** for generalizable detector improvement:
  1. Fine-tuning on 26 frames of one desk risks memorizing that specific desk background and those 7 exact object instances.
  2. `banana` (3 frames in 1 scenario) cannot be placed in both train and held-out scenario test sets.
  3. Meaningful fine-tuning would require combining external overhead/tabletop training data (e.g., rotated COCO subsets / additional multi-desk captures) with strict scenario-level separation.

---

## 3. Annotation Verification & Class Mapping

- **Format:** `manifest_v2.json` uses `PIXEL_XMIN_YMIN_XMAX_YMAX` `[x_min, y_min, x_max, y_max]` on `1600×1200` EXIF-oriented frames (`0` missing images, `0` out-of-bounds boxes).
- **Classes (7 COCO-aligned labels):** `potted plant` (30 visible), `book` (28), `bottle` (26), `scissors` (30), `cell phone` (30), `cup` (30), `banana` (3).
- **Conversion for YOLO training (if ever executed):** Normalize to `[class_id, x_center/W, y_center/H, box_w/W, box_h/H]` without altering `manifest_v2.json`.

---

## 4. Scenario-Level Train / Validation / Test Split Proposal (No Leakage)

To prevent frame leakage from duplicate/shared baseline images (`1d43944a02b2`: `removal/V01`, `addition/V02`, `occlusion/V04`; `ee864de0bd2f`: `addition/V01`, `occlusion/V01`, `occlusion/V05`), deduplicate shared hashes across splits:
- **Train (12 unique frames):** `stationary` (`V01`–`V05`, 5), `movement` (`V01`–`V05`, 5), `addition` (`V03`–`V04`, 2). *(Shared `addition/V01`–`V02` excluded.)*
- **Validation (5 unique frames):** `removal` (`V02`–`V05`, 4) + `addition/V05` (1).
- **Held-Out Test (9 unique frames):** `return_to_baseline` (`V01`–`V05`, 5), `occlusion` (`V02`–`V03`, 2), plus 1 copy each of shared `removal/V01` and `occlusion/V01`.
- **Known Split Limitation:** `banana` only exists in `addition/V03`–`V05`, and `occlusion/V02`–`V03` is the only towel-occlusion scenario, so single-pilot splits cannot simultaneously train and test those exact conditions on unseen desks.

---

## 5. Recommended Model, Pretrained Weights & Overhead Augmentation

- **Architecture:** `YOLOv8n` / `YOLOv8s` (`ultralytics`, COCO-pretrained weights `yolov8n.pt` / `yolov8s.pt`) exported to ONNX for `OpenCVDNNDetector`, **or** zero-shot evaluation of higher-capacity pretrained COCO ONNX models (`yolov8s.onnx` / `yolov8m.onnx`) without fine-tuning.
- **Overhead-Realistic Augmentations (if fine-tuning with external tabletop data):**
  - Full 360° / 90°–180°–270° rotation + horizontal & vertical flips (overhead scenes are orientation-invariant, directly addressing top-down `book`, `bottle`, `cup`, and `scissors` appearances).
  - Scale jitter (`0.5×–1.5×`), brightness/contrast/hue jitter, and random shadow/occlusion patches.

---

## 6. Evaluation Protocol & Metric Integrity

- **Keep `overall_state_accuracy` (`correct_state_evaluations / total_entity_evaluations`, `183` denominator) and all frozen `B3` definitions unchanged.**
- **Evaluate fairly against frozen baselines** (`YOLOv8n B3`, `YOLOWorld 7-Class B3`, `Track B Oracle`) on the exact same held-out sequences using:
  - Detection Recall & Precision (`IoU >= 0.50`, per-class and overall)
  - `overall_state_accuracy`
  - `identity_consistency`
  - `false_removal_rate` & `false_removal_count`
  - `removal_recall` & `event_removal_recall`
  - `confirmation_rate` (movement/addition/return-to-baseline)
  - `historical_retrieval_accuracy`

---

## 7. Expected Risks & Limitations

1. **Single-desk overfitting:** Fine-tuning on 26 frames of 1 desk will not generalize to other rooms/desks (e.g., ETH ASL Lounge).
2. **80% target realism:** Because Track B Oracle itself scores **76.50% (`140/183`)** under the current Visit-1 `INSUFFICIENT_EVIDENCE` initialization convention, an automatic detector cannot exceed `76.50%` on `overall_state_accuracy` without changing the evaluation/initialization policy (which is prohibited). Reaching `~65–75%` state accuracy (approaching the `76.50%` oracle ceiling) is the realistic upper bound for detector improvements alone.

---

## 8. Recommended Implementation Experiment

**Evaluate a higher-capacity pretrained COCO ONNX detector (`yolov8s.onnx` or `yolov8m.onnx` from official Ultralytics COCO weights, zero custom training on the 26-frame pilot)** in an isolated benchmark script on `data/physical_pilot_v2/manifest_v2.json` alongside `YOLOv8n B3`, `YOLOWorld 7-Class B3`, and `Track B Oracle`. This avoids single-desk overfitting and train/test leakage while directly testing whether stronger pretrained features close the overhead detection gap toward the `76.50%` oracle ceiling.
