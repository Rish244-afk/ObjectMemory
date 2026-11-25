# YOLO26m Implementation, Physical-Pilot Evaluation, and Live-Monitoring Report

## 1. Model Source and Exact Version
- **Model**: Official pretrained `yolo26m.pt` (COCO 80-class object detector)
- **Official Source URL**: `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt`
- **Ultralytics Version**: `8.4.218`
- **Architecture Summary**: `YOLO26m` fused (`130` layers, `20,411,132` parameters, `68.4 GFLOPs`, `Detect` head with `end2end: False`)
- **License**: `AGPL-3.0` (Ultralytics YOLO26)
- **Deployment Format**: Exported via Ultralytics (`format="onnx"`, `imgsz=640`, `opset=12`, `simplify=False`) to `models/yolo26m.onnx` (`81,900,125` bytes / `78.1 MB`), input `images: [1, 3, 640, 640]`, output `output0: [1, 84, 8400]`.

## 2. Weight Hash
- **PyTorch Checkpoint (`models/yolo26m.pt`)**: `44,255,705` bytes, SHA-256 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`
- **Exported ONNX (`models/yolo26m.onnx`)**: `81,900,125` bytes, SHA-256 `992d8f3a0ff47b0ce19a6b889f6ac2dc14c2088a955e7505775acead012fbbd8`

## 3. Hardware and Inference Runtime
- **CPU / OS**: `13th Gen Intel Core i7-1355U` on Windows (`Python 3.11.9`)
- **Runtime Selection**:
  - `ONNX Runtime 1.31.0` (`CPUExecutionProvider`) was selected over `cv2.dnn` because OpenCV DNN emitted `setPreferableTarget Targets are not supported by the new graph engine for now`, showed `max_diff=0.0121` against ORT, and was slower (`1315.5 ms` vs `1032.0 ms` in initial parity check).
  - `YOLO26mDetector` ONNX Runtime decoding matched `YOLO('models/yolo26m.onnx').predict(...)` identically on `movement/V01.jpeg` (`scissors 0.8974`, `cup 0.7921`, `potted plant 0.6324`, `book 0.5102`, `bottle 0.4705`, `vase 0.2730`).
- **YOLOv8m Baseline**: Preserved on `OpenCVDNNDetector` (`models/yolov8m.onnx`, CPU) as the production default; `YOLOv8n` (`models/yolov8n.onnx`) preserved as an alternative.

## 4. Integration Architecture
- **Separate Adapter (`src/objectmemory/detection/yolo26.py`)**: `YOLO26mDetector(ObjectDetector)` using `letterbox_image(..., (640, 640))`, RGB `[0, 1]` blob preprocessing, ONNX Runtime inference, `(1, 84, 8400)` decoding, OpenCV NMS (`0.45`), `inverse_transform_cxcywh`, `COCO_CLASSES_80` mapping, lazy session loading, SHA-256/version logging, and explicit `DetectorUnavailableError` without silent fallback.
- **Registry (`src/objectmemory/detection/registry.py`)**: Registered `"yolo26m"` (`is_default=False`), keeping `"yolov8m"` as `is_default=True` and `"yolov8n"` as alternative.
- **Safe Switching (`src/objectmemory/api/workspace_manager.py`)**: `_get_detector` unloads and evicts non-default cached detector instances when switching models to avoid holding multiple large ONNX sessions simultaneously.

## 5. Physical-Pilot Comparison (`data/physical_pilot_v2/manifest_v2.json`, Frozen B3: `conf=0.25, nms=0.45, iou=0.50`)

| Metric | YOLOv8m (Frozen Default) | YOLO26m (Optional Alternative) |
|---|---:|---:|
| **Detector-Only TP / GT Visible** | `160 / 177` | `161 / 177` |
| **Detector-Only Recall** | `90.40%` (`0.9040`) | `90.96%` (`0.9096`) |
| **Detector-Only Precision (Target Classes)** | `95.24%` (`0.9524`) | `87.50%` (`0.8750`) |
| **Non-Target COCO Predictions** | `18` | `29` |
| **Incorrect Class Predictions (IoU >= 0.50)** | `5` (all `bottle` -> `hair drier`) | `0` |
| **Missed Without Overlap** | `12` | `16` |
| **Duplicate Detections** | `0` | `0` |
| **Target Confidence (min / median / mean / p90 / max)** | `0.2523 / 0.8341 / 0.7597 / 0.9361 / 0.9499` | `0.2525 / 0.6504 / 0.6576 / 0.9287 / 0.9672` |
| **Mean / p95 Inference Latency (30 images)** | `927.04 ms` / `1111.36 ms` | `481.93 ms` / `646.09 ms` |
| **Complete ObjectMemory B3 State Accuracy** | `89.07%` (`163 / 183`) | `90.71%` (`166 / 183`) |
| **B3 Removal Recall / Event Removal Recall** | `0 / 3` (`0.0%`) / `0 / 1` (`0.0%`) | `2 / 3` (`66.67%`) / `1 / 1` (`100.0%`) |
| **B3 False Removal Count / Rate** | `1 / 180` (`0.56%`) | `3 / 180` (`1.67%`) |
| **B3 Spurious / Duplicate Tracks** | `11` | `15` |

*Note: Detector-only metrics and complete ObjectMemory temporal-memory B3 metrics are reported separately above; frozen B3 thresholds (`0.25` / `0.45`) were not altered.*

## 6. Per-Class Comparison (Physical Pilot, IoU >= 0.50)

| Class | GT Visible | YOLOv8m TP / Preds (Rec / Prec) | YOLO26m TP / Preds (Rec / Prec) |
|---|---:|---:|---:|
| `potted plant` | `30` | `27 / 27` (`90.00%` / `100.0%`) | `30 / 30` (`100.0%` / `100.0%`) |
| `book` | `28` | `28 / 36` (`100.0%` / `77.78%`) | `27 / 50` (`96.43%` / `54.00%`) |
| `bottle` | `26` | `12 / 12` (`46.15%` / `100.0%`) | `21 / 21` (`80.77%` / `100.0%`) |
| `scissors` | `30` | `30 / 30` (`100.0%` / `100.0%`) | `30 / 30` (`100.0%` / `100.0%`) |
| `cell phone` | `30` | `30 / 30` (`100.0%` / `100.0%`) | `21 / 21` (`70.00%` / `100.0%`) |
| `cup` | `30` | `30 / 30` (`100.0%` / `100.0%`) | `29 / 29` (`96.67%` / `100.0%`) |
| `banana` | `3` | `3 / 3` (`100.0%` / `100.0%`) | `3 / 3` (`100.0%` / `100.0%`) |

Key class trade-offs:
- **Bottle recall** improved substantially from `46.15%` (`12/26`, with 5 misclassified as `hair drier`) to `80.77%` (`21/26`, `0` misclassifications).
- **Book false positives** increased from `8` (`36` preds for `28` GT) to `23` (`50` preds for `28` GT, precision `54.00%`).
- **Cell phone recall** dropped from `100.0%` (`30/30`) to `70.00%` (`21/30`).

## 7. Six-Scenario Comparison (ObjectMemory B3 Correct State Evaluations)

| Scenario | YOLOv8m Correct Evaluations | YOLO26m Correct Evaluations |
|---|---:|---:|
| `stationary` | `25` | `27` |
| `movement` | `24` | `23` |
| `removal` | `25` | `28` |
| `addition` | `32` | `31` |
| `occlusion` | `28` | `28` |
| `return_to_baseline` | `29` | `29` |
| **Total (out of 183)** | **`163` (`89.07%`)** | **`166` (`90.71%`)** |

## 8. Webcam Recording Comparison (`research_validation/webcam_recordings/live_webcam_32s.avi`)
- **Capture Metadata**: Device `0` (`cv2.VideoCapture(0)`), `1280x720`, MJPG codec, reported `30.0` fps, actual capture `29.99 fps`, `960` frames over `~32.0s` (`live_webcam_32s_meta.json`).
- **Replay Protocol**: Exact same recording replayed at `1.0 Hz` (`32` sampled frames) through `YOLOv8m` and `YOLO26m` (`research_validation/webcam_yolov8m_vs_yolo26m_comparison.json`). Ground-truth accuracy is not claimed for the unannotated webcam stream.

| Metric (32 Sampled Frames) | YOLOv8m | YOLO26m |
|---|---:|---:|
| **Detector Latency (Mean / p95)** | `779.80 ms` | `365.19 ms` |
| **End-to-End Pipeline Latency (Mean)** | `1147.79 ms` | `746.40 ms` |
| **Detector Processing FPS** | `1.28 fps` | `2.74 fps` |
| **End-to-End Processing FPS** | `0.87 fps` | `1.34 fps` |
| **Simulated Real-Time Drop Rate (@ 29.99 fps)** | `97.2%` | `95.6%` |
| **Planar Registration Success** | `31 / 31` | `31 / 31` |
| **Observed Detections** | `person: 6`, `cup: 1` | `person: 7`, `cup: 1` |
| **`person` Continuity / Consecutive IoU** | `18.8%` (`6/32`) / `0.3068` | `21.9%` (`7/32`) / `0.3410` |
| **`person` Confidence (Mean / Std)** | `0.8304` / `0.073` | `0.7650` / `0.130` |

## 9. FPS and Latency
- On 1600x1200 physical-pilot images, `YOLO26m` (`481.93 ms` mean) is **1.92x faster** than `YOLOv8m` (`927.04 ms` mean).
- On 1280x720 webcam frames, `YOLO26m` (`365.19 ms` detector, `746.40 ms` E2E) is **2.14x faster** in detection and **1.54x faster** end-to-end than `YOLOv8m` (`779.80 ms` detector, `1147.79 ms` E2E).

## 10. Memory Usage and CPU Utilization
- **Physical Pilot (`compare_yolov8m_vs_yolo26m_pilot.py`)**:
  - `YOLOv8m`: Process RSS `97.57 MB -> 319.44 MB` (`+221.87 MB`), `tracemalloc` Python peak `67.56 MB`.
  - `YOLO26m`: Process RSS `313.94 MB -> 725.79 MB` (`+411.85 MB`), `tracemalloc` Python peak `68.04 MB`.
- **Webcam Replay (`capture_and_compare_webcam.py`)**:
  - `YOLOv8m`: Mean CPU `59.7%`, RSS `439.42 -> 422.71 MB`, Python peak `12.16 MB`.
  - `YOLO26m`: Mean CPU `97.0%` (ONNX Runtime multi-threaded CPU execution), RSS `675.79 -> 824.14 MB`, Python peak `12.06 MB`.

## 11. Live Pipeline Failure Analysis
1. **CPU Throughput Bottleneck**: Even with `YOLO26m` (`2.74 fps` detector-only, `1.34 fps` E2E), CPU inference + SIFT/homography registration cannot run at 30 fps; real-time auto-capture must use interval sampling (e.g., 5s/10s) and skip overlapping ticks rather than queueing concurrent requests.
2. **Detector Class & Precision Trade-offs**: `YOLOv8m` misclassified the dark bottle as `hair drier` (`5` frames), whereas `YOLO26m` eliminated those `5` misclassifications (`80.77%` bottle recall) but produced more `book` false positives (`50` preds vs `28` GT) and lower `cell phone` recall (`70.0%`).
3. **Epistemic Gating**: Single missed frames are gated through `PENDING_REMOVAL` / `INSUFFICIENT_EVIDENCE` rather than immediately confirming removal, and unreliable registration withholds movement claims.

## 12. Changes Made to the Application
- Added `src/objectmemory/detection/yolo26.py` (`YOLO26mDetector`) and exported `models/yolo26m.onnx` (keeping `models/yolov8m.onnx` and `models/yolov8n.onnx` untouched).
- Registered `"yolo26m"` in `src/objectmemory/detection/registry.py` (`is_default=False`).
- Updated `WorkspaceManager._get_detector` in `src/objectmemory/api/workspace_manager.py` to evict non-default heavy detector sessions on model switch.
- Updated `frontend/src/views/WorkspaceOverviewView.tsx` to persist `selectedDetector` in `localStorage`, display active detector status, guard overlapping webcam auto-capture ticks (tracking `Skipped ticks`), and surface live epistemic status badges (`Detected`, `Not detected`, `Registration unreliable`, latency).
- Added unit/integration tests in `tests/test_yolo26m.py` and evaluation scripts `research_validation/scripts/compare_yolov8m_vs_yolo26m_pilot.py` and `research_validation/scripts/capture_and_compare_webcam.py`.

## 13. Test Results
- **Frontend (`npm run build`: `tsc -b && vite build`)**: Passed (`0` errors, built in `1.15s`).
- **Pytest (`tests/test_yolo26m.py`, `tests/test_detector_registry.py`, `tests/test_identity_and_recovery.py`, `tests/test_comparison_correctness.py`, `tests/test_api_backend.py`, `tests/test_eth_registration_api.py`)**: **48 passed, 0 failed** (`29.75s`).

## 14. Known Limitations
- `YOLO26m` has higher CPU memory footprint (`~412 MB` RSS delta via ONNX Runtime) and higher CPU utilization (`97.0%` during continuous replay) than `YOLOv8m`.
- `YOLO26m` exhibits lower `cell phone` recall (`70.0%`) and higher `book` false positives (`23` FP) on the physical pilot.
- Webcam comparison has no manual ground-truth bounding boxes, so webcam accuracy is not claimed.

## 15. Recommendation
**Keep `YOLOv8m` as the production default** (higher precision `95.24%`, lower memory footprint, `100%` `cell phone` and `book` recall), and **include `YOLO26m` as a ready-to-use optional alternative** for live demonstrations where **~2x faster CPU latency** and **higher `bottle` recall (`80.77%` vs `46.15%`, `0` `hair drier` misclassifications)** are prioritized.
