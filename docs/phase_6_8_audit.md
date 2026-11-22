# Phase 6.8 Audit: Final Evaluation Semantics — Reconciliation and Claim Verification

**Official Project Title:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Official Project Description:**  
> "An AI-based visual memory system that compares scenes over time to detect and explain persistent changes in object locations while filtering out temporary variations such as lighting, occlusion, and viewpoint changes."

**Date:** October 10, 2026  
**Audit type:** Evidence Reconciliation — corrects unsupported or imprecise claims in Phase 6.8  
**Active Test Suite Status:** 140 / 140 tests passing  

---

## Audit Scope

Each claim in the original Phase 6.8 report was re-examined against:
- Actual source code in `src/objectmemory/`.
- Actual test implementations in `tests/`.
- Actual command output from the three required commands.

Every claim below is labeled with one of:
- **[Verified by source inspection]** — confirmed by reading the file cited.
- **[Verified by executed test]** — a specific, named test passes and is cited by function name.
- **[Supported only by synthetic testing]** — evidence exists but comes from procedurally generated images, not physical photographs.
- **[Not yet verified]** — no supporting source or test evidence at this time.

---

## 1. Movement Threshold Evidence (35 px)

### 1A. Test cited: `test_localization_jitter_small` (Test 1A, `test_temporal_robustness.py` lines 84–99)

| Property | Value |
| :--- | :--- |
| Input type | **Synthetic** — `np.zeros((480, 640, 3), dtype=np.uint8)` black frames |
| Image dimensions | 640 × 480 pixels |
| Reference bbox | `BoundingBox(100, 100, 200, 200)` — centroid (150, 150) |
| Query bbox (Visit 2) | `BoundingBox(108, 106, 208, 206)` — centroid (158, 156) |
| Measured centroid shift | $\sqrt{8^2 + 6^2} = 10.0$ pixels |
| Registration | Mock identity homography; `inlier_ratio=0.88` (mock, not computed) |
| Movement classification | No movement (< 35 px threshold) |
| Expected result | `CONFIRMED_PRESENT` maintained |
| Test passes | **Yes [Verified by executed test]** |

### 1B. Test cited: `test_localization_jitter_exceeding_threshold_documents_limitation` (Test 1B, lines 101–116)

| Property | Value |
| :--- | :--- |
| Input type | **Synthetic** — `np.zeros((480, 640, 3), dtype=np.uint8)` black frames |
| Image dimensions | 640 × 480 pixels |
| Reference bbox | `BoundingBox(100, 100, 200, 200)` — centroid (150, 150) |
| Query bbox (Visit 2) | `BoundingBox(140, 100, 240, 200)` — centroid (190, 150) |
| Measured centroid shift | $\sqrt{40^2 + 0^2} = 40.0$ pixels |
| Registration | Mock identity homography; `inlier_ratio=0.88` (mock, not computed) |
| Movement classification | Relocation (> 35 px threshold) |
| Expected result | `CONFIRMED_MOVED` |
| Test passes | **Yes [Verified by executed test]** |

### Assessment: Corrected Claims

The following claims from the original Phase 6.8 report are **not supported** by the above evidence and are corrected here:

1. **"5–25-pixel registration-noise range"** — This range is **not measured** in any executed test. The tests use mock identity homographies with no keypoint computation and no measured residual error. The range was stated only as a narrative assertion. **[Not yet verified]**

2. **"Greater-than-80-pixel relocation range"** — No test demonstrates an 80-pixel or larger genuine relocation. Test 1B uses exactly 40 pixels. The 80-pixel value was stated only as a narrative assertion. **[Not yet verified]**

3. **"Physically calibrated 35-pixel threshold"** — The threshold is an **engineering parameter** set in `TemporalEngineConfig` (`movement_displacement_threshold=35.0`, `test_temporal_robustness.py` line 59). All tests use procedurally generated black frames with mock homographies. No physical workbench photograph has been processed through the homography pipeline to empirically validate this choice. The threshold is correctly described as: **an engineering parameter awaiting physical calibration. [Supported only by synthetic testing]**

4. **Registration inlier count / ratio** — All robustness tests use a mocked `RegistrationResult` with `inlier_ratio=0.88` hardcoded. No real SIFT/ORB keypoint computation is performed in the cited tests. The inlier ratio is therefore not measured from actual feature matching. **[Supported only by synthetic testing]**

---

## 2. Ground-Truth Isolation Test (Test 34)

### What the test does

Source: `tests/test_evaluator_adversarial.py` lines 1496–1554  
(`test_ground_truth_isolation_annotation_change_invariance`)

**Run 1 setup:**
- Images `f1`, `f2`: synthetic workbench PNGs (640 × 480) produced by `_generate_synthetic_workbench(w=640, h=480, seed=42)` + `cv2.imwrite`. **Not physical photographs.**
- Detector: `stub1 = ControlledStubDetector({"visit_0": [...], "visit_1": [...]})` — returns pre-specified `Detection` objects, does not call any inference model.
- Annotations: `entity_id="c1"`, `entity_label="cup"`, `expected_state=CONFIRMED_PRESENT`.
- Store: isolated `SQLiteMemoryStore(tmp_path / "run1.db")`.

**Run 2 setup:**
- Same images (`f1`, `f2`) — identical bytes.
- Detector: `stub2 = ControlledStubDetector(...)` — a fresh instance of the same stub config, starting with `call_count=0`.
- Annotations: `entity_id="fake_id_xyz"`, `entity_label="fake_label"`, `expected_state=CONFIRMED_REMOVED`, `physical_observation=ABSENT`.
- Store: isolated `SQLiteMemoryStore(tmp_path / "run2.db")`.

**What is checked:**
- `len(ents1) == len(ents2)` — entity count is the same (both = 1 detected cup).
- `ents1[0].current_state == ents2[0].current_state` — both `CONFIRMED_PRESENT`.
- `ents1[0].label == ents2[0].label` — both "cup", not "fake_label".
- `len(obs1) == len(obs2)` — observation count identical.
- `len(evts1) == len(evts2)` — change event count identical.
- `report1.change_metrics.accuracy != report2.change_metrics.accuracy` — scores differ, proving annotations affected only the evaluator.

**What the test proves — [Verified by executed test]:**
- Evaluator-side annotation changes (`entity_id`, `entity_label`, `expected_state`, `physical_observation`) do **not** alter the system's stored entities, state labels, observations, or change events.
- The evaluator reads system predictions and compares them to annotations without writing back to the store.

**What the test does not prove:**
- Isolation with a real ONNX inference model. The detector in both runs is a deterministic stub returning hard-coded `Detection` objects.
- Isolation under concurrent multi-threaded access or adversarial store tampering.
- That the evaluator cannot access store data during `evaluate()` — it receives the `runner` which holds the store; the test merely shows it does not write contradicting state through the store API.

**No defect found in Test 34 as currently implemented.** The two stubs (`stub1`, `stub2`) are separate instances with `call_count=0` each, and the stores are separate files.

---

## 3. Detector Evidence and Preprocessing

### 3A. The demo image used for latency measurement

Source: `tools/run_real_pipeline.py` lines 122–131

The CLI's `[3/4]` section, when no physical dataset images exist, falls into an `else` branch that checks for `demo_alignment_overlay.png` and runs inference on it.

`demo_alignment_overlay.png` is the **synthetic diagnostic alignment overlay** generated by the registration benchmark tool. It is **not a photograph of a physical scene**. **[Not yet verified as real-world detector performance]**

The reported latency (125–150 ms across runs) is therefore described correctly as: **CPU inference latency measured on a synthetic diagnostic image. It is not real-world detector validation.**

### 3B. Preprocessing implementation — confirmed by source inspection

Source: `src/objectmemory/detection/detector.py` lines 167–184 and 218–230.

The YOLOv8 ONNX preprocessing path uses:

```python
blob = cv2.dnn.blobFromImage(
    image_bgr,
    scalefactor=1.0 / 255.0,
    size=self.input_size,      # (640, 640) for YOLOv8
    mean=(0.0, 0.0, 0.0),
    swapRB=True,               # BGR -> RGB channel swap
    crop=False,                # No cropping; direct resize
)
```

This performs:
- Bilinear resize of the input BGR image to $(640, 640)$, potentially distorting the aspect ratio.
- Channel swap from BGR to RGB (`swapRB=True`).
- Pixel normalization by dividing by 255, shifting to $[0, 1]$.

Coordinate inverse scaling (lines 218–230):
```python
x_scale = w_orig / float(in_w)   # in_w = 640
y_scale = h_orig / float(in_h)   # in_h = 640
x_min = max(0.0, (cx - 0.5 * w) * x_scale)
y_min = max(0.0, (cy - 0.5 * h) * y_scale)
box_w = min(float(w_orig) - x_min, w * x_scale)
box_h = min(float(h_orig) - y_min, h * y_scale)
```

**Confirmed: direct anisotropic stretching** to $(640, 640)$ followed by direct linear coordinate scaling. No letterboxing. **[Verified by source inspection]**

### 3C. Non-square coordinate conversion test

Test: `test_detector_coordinate_scaling_mathematical_accuracy` (Test 10, `test_evaluator_adversarial.py`)  
This test exercises the coordinate scaling formula directly with controlled inputs across multiple aspect ratios (1920×1080, 1280×720, 800×600). It does **not** call the ONNX model; it tests the formula only. **[Verified by executed test — synthetic inputs]**

---

## 4. Metric Zero-Denominator Conditioning

### 4A. Numerator and denominator implementation

Source: `src/objectmemory/benchmarks/real_evaluator.py` lines 601–628.

| Metric | Numerator | Denominator | Returns `None` when |
| :--- | :--- | :--- | :--- |
| `accuracy` | `correct_preds` | `total_evals` | `total_evals == 0` |
| `false_confirmed_removal_rate` | `false_removals` | `total_evals` | `total_evals == 0` |
| `false_confirmed_removal_on_present_rate` | `false_removals` | `physically_present_count` | `physically_present_count == 0` |
| `removal_recall` | `correct_removals` | `expected_removals` | `expected_removals == 0` |
| `relocation_recall` | `correct_relocations` | `expected_relocations` | `expected_relocations == 0` |
| `abstention_rate` | `abstentions` | `total_evals` | `total_evals == 0` |

For `RealDetectionMetrics` (lines 309–312):

| Metric | Returns `None` when |
| :--- | :--- |
| `precision` | `total_pred == 0` |
| `recall` | `total_gt == 0` |
| `f1_score` | `precision is None or recall is None or (prec + rec) == 0` |
| `mean_iou` | No matched boxes (`ious` list is empty) |

**[Verified by source inspection]**

### 4B. Text formatting

`format_text_summary()` (lines 133–159) renders all `None` metrics as `N/A` using:
```python
f'{dm.precision:.3f}' if dm.precision is not None else 'N/A'
```
**[Verified by source inspection]**

### 4C. JSON serialization

> **Finding: No explicit JSON export method exists in `RealImageEvaluationReport`.**

The class is a `@dataclass` but does not define `to_dict()`, `to_json()`, or `asdict()`-based export. The only export surface is `format_text_summary()` which is text. Standard `dataclasses.asdict()` would serialize `Optional[float]` fields that are `None` to JSON `null` correctly, but this is not called anywhere in the current codebase.

**The original Phase 6.8 report's claim "JSON export serializes undefined metrics as JSON `null`" is not verified by any existing code path or test. [Not yet verified]**

No test was added for JSON serialization. This is a gap.

### 4D. Zero-denominator test

Test 33: `test_undefined_metric_denominators_return_none` — verifies `removal_recall is None`, `relocation_recall is None`, and `accuracy == 1.0` for a static present sequence. **[Verified by executed test — synthetic inputs]**

Both zero-denominator (Test 33) and non-zero-denominator (Tests 5, 11, 12, etc.) cases exist. **[Verified by executed test]**

---

## 5. Actual Command Output

### Command 1: `python -m unittest tests/test_evaluator_adversarial.py -v`

Exit code: **0** (confirmed)  
Actual output line:
```
Ran 34 tests in 16.146s
OK
```
All 34 tests passed. No failures, errors, or skips.

### Command 2: `python -m unittest discover -s tests -v`

Exit code: **0** (confirmed; a prior run with `2>&1` pipe reported code 1 due to OpenCV stderr in PowerShell — this is a shell stderr-redirect artifact, not a Python failure)  
Actual output line:
```
Ran 140 tests in 40.390s
OK
```
All 140 tests passed. No failures, errors, or skips.

### Command 3: `python tools/run_real_pipeline.py`

Exit code: **0** (confirmed via `$LASTEXITCODE`)  
Status line: `STATUS: BLOCKED_PENDING_ANNOTATIONS`  
Reason: `Real workbench dataset does not contain physical multi-visit images and manual ground-truth annotations.`  
Inference latency (on `demo_alignment_overlay.png`): **~125 ms** (varies run-to-run; 124–150 ms observed).

The pipeline is correctly blocked. **[Verified by executed test]**

> [!NOTE] The OpenCV `WARN: Targets are not supported by the new graph engine for now` message is printed to stderr by the OpenCV DNN module when setting the target on newer OpenCV builds. It does not affect functionality and the model falls back to the default OpenCV graph engine correctly.

---

## 6. Summary: Claims Corrected in This Reconciliation

| Claim (original Phase 6.8 report) | Correction | Label |
| :--- | :--- | :--- |
| "5–25-pixel registration-noise range" measured in tests | Not measured; stated as narrative. Tests use mock identity homographies. | **Not yet verified** |
| "> 80-pixel genuine relocation" measured in tests | Not measured; no test uses 80 px+. Largest synthetic shift is 40 px. | **Not yet verified** |
| "Physically calibrated 35-pixel threshold" | The threshold is an engineering parameter. All evidence is synthetic. | **Supported only by synthetic testing** |
| "149.9 ms inference latency on real-world image" | Image is `demo_alignment_overlay.png`, a synthetic diagnostic overlay, not a real photograph. | **Supported only by synthetic testing** |
| "JSON export serializes undefined metrics as `null`" | No JSON export method or path exists in the current codebase. | **Not yet verified** |
| "Registration inlier ratio measured in robustness tests" | All robustness tests mock `inlier_ratio=0.88` directly; no real keypoint matching performed. | **Supported only by synthetic testing** |

---

## 7. What Remains Correct After Reconciliation

| Claim | Label |
| :--- | :--- |
| YOLOv8n ONNX preprocessing is direct $(640, 640)$ stretch via `blobFromImage(crop=False)`, not letterboxed | **[Verified by source inspection]** |
| BGR→RGB swap via `swapRB=True`; normalization by 1/255 | **[Verified by source inspection]** |
| Coordinate inverse scaling: `x_scale = w_orig / 640`, `y_scale = h_orig / 640` with NMS via `cv2.dnn.NMSBoxes` | **[Verified by source inspection]** |
| Temporal policy: single absence → `PENDING_REMOVAL`; two consecutive reliable absences → `CONFIRMED_REMOVED` | **[Verified by source inspection + executed tests]** |
| Occlusion (`FULLY_OCCLUDED`, `PARTIALLY_OCCLUDED`) freezes removal evidence accumulation | **[Verified by source inspection + executed tests]** |
| `FULLY_OCCLUDED`, `OUT_OF_FRAME`, `ABSENT` require `bbox is None`; schema rejects violations | **[Verified by source inspection + executed tests]** |
| Metric denominators are correct; `None` returned when denominator = 0 | **[Verified by source inspection + executed tests — synthetic inputs]** |
| `format_text_summary()` renders `None` as `N/A` | **[Verified by source inspection]** |
| Ground-truth isolation: annotation changes have no effect on stored system entities (Test 34) | **[Verified by executed test — synthetic inputs]** |
| `VisitInput` has no `ground_truth_entities` or `expected_states` fields (Test 7) | **[Verified by executed test]** |
| Pipeline correctly blocks evaluation when physical photographs and annotations are absent | **[Verified by executed test — actual CLI run]** |
| Inference executes on CPU without crash; loads 80-class COCO taxonomy | **[Verified by source inspection + synthetic image inference]** |

---

## 8. Evidence Still Required for Genuine Physical Evaluation

The following evidence cannot be generated synthetically and must await genuine physical photographs with independent manual annotations:

1. **Physical threshold calibration:** Record actual registration residual (reprojection error in pixels) and detector bounding box jitter on multiple real workbench frames across varying viewpoints. Confirm that combined jitter is below 35 px for stationary objects.
2. **Real-world relocation range:** Confirm that deliberate object displacements (mug moved 10 cm, etc.) produce measured canonical centroid displacements significantly above 35 px.
3. **Detector performance on workbench objects:** Record precision, recall, and F1 on physical photographs of labeled workbench objects using the COCO 80-class taxonomy. Document which object classes are represented and which are not.
4. **JSON export validation:** If a machine-readable export of evaluation results is needed, implement and test `to_dict()` / JSON serialization for `RealImageEvaluationReport` with explicit handling of `Optional[float]` → JSON `null`.
5. **Multi-visit temporal accuracy on physical data:** Once photographs and manual annotations exist, run the full evaluation pipeline and report actual `accuracy`, `removal_recall`, and `false_confirmed_removal_rate` from physical sequences.

See `docs/real_dataset_collection.md` for the complete data collection protocol and annotation template.
