# ObjectMemory — Detection Failure Analysis & Final Detector Decision

**Dataset:** `data/physical_pilot_v2/manifest_v2.json` (30 images, 177 visible GT annotations across 7 target classes)  
**Raw Breakdown JSON:** `research_validation/detection_failure_analysis/detection_failure_breakdown.json`  
**Visual Diagnostic Grids:**
- `research_validation/detection_failure_analysis/comparison_grid_1_3.png` (`stationary_V01`, `movement_V02`, `removal_V02`)
- `research_validation/detection_failure_analysis/comparison_grid_4_6.png` (`addition_V03`, `occlusion_V02`, `return_to_baseline_V02`)

---

## 1. Per-Class Failure Breakdown (Production Threshold vs `0.01` Diagnostic Probe)

### A. `OpenCVDNNDetector` (`models/yolov8n.onnx`, `prod_conf = 0.25`)

| GT Class | Visible GT | TP (`>=0.25`) | Filtered by Conf (`0.01..0.25`) | Class Mismatch (`IoU>=0.50`) | Low IoU | Genuinely Missed (`<0.01`) | Max Conf (`0.01` probe) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `potted plant` | 30 | **9** | 21 | 0 | 0 | 0 | 0.5123 |
| `book` | 28 | **0** | 20 | 0 | 0 | 8 | 0.1946 |
| `bottle` | 26 | **0** | 7 | 1 (`scissors`) | 0 | 18 | 0.1901 |
| `scissors` | 30 | **30** | 0 | 0 | 0 | 0 | 0.9805 |
| `cell phone` | 30 | **1** | 26 | 0 | 0 | 3 | 0.2689 |
| `cup` | 30 | **5** | 16 | 0 | 1 | 8 | 0.4122 |
| `banana` | 3 | **3** | 0 | 0 | 0 | 0 | 0.8551 |
| **Total** | **177** | **48 (27.12%)** | **90 (50.85%)** | **1 (0.56%)** | **1 (0.56%)** | **37 (20.90%)** | — |

### B. `YOLOWorldDetector` (Explicit 7 Prompts, `prod_conf = 0.15`)

| GT Class | Visible GT | TP (`>=0.15`) | Filtered by Conf (`0.01..0.15`) | Class Mismatch (`IoU>=0.50`) | Low IoU | Genuinely Missed (`<0.01`) | Max Conf (`0.01` probe) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `potted plant` | 30 | **3** | 27 | 0 | 0 | 0 | 0.2200 |
| `book` | 28 | **0** | 24 | 0 | 0 | 4 | 0.0739 |
| `bottle` | 26 | **22** | 4 | 0 | 0 | 0 | 0.4718 |
| `scissors` | 30 | **0** | 29 | 0 | 0 | 1 | 0.0401 |
| `cell phone` | 30 | **29** | 1 | 0 | 0 | 0 | 0.4344 |
| `cup` | 30 | **0** | 16 | 0 | 1 | 13 | 0.0477 |
| `banana` | 3 | **0** | 3 | 0 | 0 | 0 | 0.1378 |
| **Total** | **177** | **54 (30.51%)** | **104 (58.76%)** | **0 (0.00%)** | **1 (0.56%)** | **18 (10.17%)** | — |

---

## 2. Confidence & Object-Specific Root Causes

1. **Why `book` fails (`0/28` in both):** It is a closed, flat, black-and-white graphic-covered *"Friends"* notebook viewed from 90° overhead; neither `YOLOv8n` (`max conf 0.1946`) nor `YOLO-World` (`max conf 0.0739`) has strong top-down visual/CLIP alignment for a closed graphic notebook without visible spine/pages.
2. **Why `cup` fails (`0/30` in YOLO-World, `5/30` in YOLOv8n):** Viewed strictly from above, an empty ceramic mug appears as two concentric circles with a tiny side handle, producing very weak zero-shot CLIP confidence in `YOLO-World` (`max conf 0.0477`, 13/30 genuinely missed even at `0.01`).
3. **Why `scissors` fails in YOLO-World (`0/30`) despite `30/30` in YOLOv8n (`0.77–0.98`):** Thin metal blades + red plastic handles from 90° overhead score only `<= 0.0401` in `YOLOv8s-Worldv2`'s CLIP text-image head, whereas closed-set `YOLOv8n` detects them reliably (`30/30`).
4. **Why `potted plant` recall is `10%` (`3/30` in YOLO-World, `9/30` in YOLOv8n):** Overhead foliage on top of a printed rectangular booklet (`"Technical Education"`) splits/dilutes visual features so scores hover right around the threshold (`0.15–0.22` in YOLO-World, `0.15–0.51` in YOLOv8n).
5. **Why `banana` recall is `0/3` in YOLO-World despite `3/3` (`0.84–0.86`) in YOLOv8n:** `YOLO-World` **did** localize `banana_01` in all `3/3` frames, but with `max conf = 0.1378` — just below `default_confidence_threshold = 0.15`.
6. **Why `bottle` (`0/26` in YOLOv8n vs `22/26` in YOLO-World) and `cell phone` (`1/30` vs `29/30`) flip:** `YOLOv8n` misses top-down dark cylindrical bottles (`18/26` `<0.01`, `1/26` misclassified as `scissors` in `return_to_baseline_V02`) and black phones (`26/30` `<0.25`), whereas `YOLO-World` detects both strongly.

---

## 3. Final Detector Decision & Next Implementation Task

- **Keep `YOLOv8n` (`B3`) as the frozen baseline** for previously published synthetic/ETH/pilot benchmarks (do not alter historical results).
- **Retain explicit-prompt `YOLOWorldDetector` as the primary physical-pilot alternative** (it achieved higher state accuracy `31.69%` vs `30.60%`, higher identity consistency `30.17%` vs `26.82%`, `0` spurious tracks, `100%` event removal recall, and `100%` historical retrieval accuracy — and `YOLOv8n` + `YOLO-World` are strictly complementary across the 7 overhead objects: `YOLOv8n` gets `30/30` `scissors` + `3/3` `banana` while `YOLO-World` gets `29/30` `cell phone` + `22/26` `bottle`).
- **Single next implementation task:** Recapture the 5 duplicate/shared images in `data/physical_pilot_v2/` (`addition/V01`, `addition/V02`, `occlusion/V01`, `occlusion/V04`, `occlusion/V05`), update `manifest_v2.json` for those 5 frames, and finalize the end-semester experimental tables comparing frozen `YOLOv8n B3`, `YOLOWorld (7-class) B3`, and `Track B (Oracle)` (noting overhead detector recall as a documented physical-evaluation limitation).
