# ObjectMemory — Larger Pretrained Detector Experiment Report

**Dataset:** `data/physical_pilot_v2/manifest_v2.json` (6 sequences, 30 observations, 183 annotations, 177 visible GT boxes)  
**Experiment Directory:** `research_validation/larger_detector_experiment/`  
**Scripts & Artifacts:**
- `research_validation/larger_detector_experiment/run_yolov8s_b3.py` → `yolov8s_b3_results.json`
- `research_validation/larger_detector_experiment/run_yolov8m_b3.py` → `yolov8m_b3_results.json`
- Official Ultralytics COCO weights exported to ONNX (`opset=12`, `(1, 3, 640, 640)` RGB → `(1, 84, 8400)`), zero fine-tuning, frozen `B3` config (`conf=0.25`, `nms=0.45`).

---

## 1. Model Compatibility & Resource Requirements

| Model | Weights File | Size | Parameters / GFLOPs | Input / Output | Mean CPU Latency | p95 CPU Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `YOLOv8n` (baseline) | `models/yolov8n.onnx` | `12.2 MB` | ~3.2M / ~8.7 | `(1,3,640,640)` / `(1,84,8400)` | — | — |
| `YOLOv8s` (official) | `.../models/yolov8s.onnx` | `42.79 MB` | `11.16M` / `28.6` | `(1,3,640,640)` / `(1,84,8400)` | **`392.17 ms`** | `418.87 ms` |
| `YOLOv8m` (official) | `.../models/yolov8m.onnx` | `99.00 MB` | `25.89M` / `78.9` | `(1,3,640,640)` / `(1,84,8400)` | **`895.29 ms`** | `1042.10 ms` |

---

## 2. Detection Performance (`IoU >= 0.50`, `conf >= 0.25`)

| Class (GT Visible) | `YOLOv8n` Recall | `YOLOv8s` Recall (Prec) | `YOLOv8m` Recall (Prec) | `YOLOWorld` 7-Class Recall |
| :--- | :---: | :---: | :---: | :---: |
| `potted plant` (30) | `9/30` (30.00%) | `15/30` (50.00% / 100%) | **`27/30` (90.00% / 100%)** | `3/30` (10.00%) |
| `book` (28) | `0/28` (0.00%) | `11/28` (39.29% / 100%) | **`28/28` (100.00% / 77.78%)** | `0/28` (0.00%) |
| `bottle` (26) | `0/26` (0.00%) | `12/26` (46.15% / 100%) | `12/26` (46.15% / 100%) | **`22/26` (84.62%)** |
| `scissors` (30) | `30/30` (100.00%) | `30/30` (100.00% / 100%) | **`30/30` (100.00% / 100%)** | `0/30` (0.00%) |
| `cell phone` (30) | `1/30` (3.33%) | `7/30` (23.33% / 100%) | **`30/30` (100.00% / 100%)** | `29/30` (96.67%) |
| `cup` (30) | `5/30` (16.67%) | `9/30` (30.00% / 90.0%) | **`30/30` (100.00% / 100%)** | `0/30` (0.00%) |
| `banana` (3) | `3/3` (100.00%) | `3/3` (100.00% / 100%) | **`3/3` (100.00% / 100%)** | `0/3` (0.00%) |
| **Overall (177)** | **`48/177` (27.12%)** | **`87/177` (49.15% / 98.86%)** | **`160/177` (90.40% / 95.24%)** | **`54/177` (30.51% / 100%)** |

---

## 3. Fair 5-Way B3 Pipeline Comparison

| Metric | `YOLOv8n B3` | `YOLOv8s B3` | `YOLOv8m B3` | `YOLOWorld` 7-Class B3 | `Track B Oracle` *(Diagnostic Ref Only)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **State Accuracy** | `56/183` (30.60%) | `99/183` (54.10%) | **`163/183` (89.07%)** | `58/183` (31.69%) | `140/183` (76.50%) |
| **Identity Consistency** | `48/179` (26.82%) | `87/179` (48.60%) | **`160/179` (89.39%)** | `54/179` (30.17%) | N/A |
| **False Removal Rate** | `6/180` (3.33%) | `4/180` (2.22%) | **`1/180` (0.56%)** | `2/180` (1.11%) | N/A |
| **False Discovery Prop.** | `6/6` (100.00%) | `4/4` (100.00%) | `1/1` (100.00%) | `2/4` (50.00%) | N/A |
| **Removal Recall** | `0/3` (0.00%) | `0/3` (0.00%) | `0/3` (0.00%) | **`2/3` (66.67%)** | Included in `22/30` |
| **Change Confirmation** | `0/3` (0.00%) | `0/3` (0.00%) | `1/3` (33.33%) | **`2/3` (66.67%)** | N/A |
| **Historical Retrieval** | `0/0` (`None`) | `1/2` (50.00%) | **`2/2` (100.00%)** | **`2/2` (100.00%)** | N/A |
| **Registration Failures** | `0/24` | `0/24` | `0/24` | `0/24` | `0/24` |
| **ID Switches** | `0` (`0/32`) | `0` (`0/61`) | `0` (`0/126`) | `0` (`0/41`) | `0` |
| **Spurious / Duplicate IDs** | `1` / `1` | `4` / `4` | `11` / `11` | `0` / `0` | `0` / `1` |

*(Methodological note: `YOLOv8m B3` (`89.07%`) exceeds `Track B Oracle` (`76.50%`) because `BenchmarkRunner` and `evaluate_physical_track_b` use different Visit-1 initialization scoring conventions; `Track B` is a diagnostic reference only, not an automatic detector.)*

---

## 4. Scenario-Wise State Accuracy (`YOLOv8n` vs `YOLOv8s` vs `YOLOv8m`)

| Scenario | `YOLOv8n B3` | `YOLOv8s B3` | `YOLOv8m B3` |
| :--- | :---: | :---: | :---: |
| `stationary` | `5/30` (16.67%) | `11/30` (36.67%) | **`25/30` (83.33%)** |
| `movement` | `7/30` (23.33%) | `7/30` (23.33%) | **`24/30` (80.00%)** |
| `removal` | `7/30` (23.33%) | `5/30` (16.67%) | **`25/30` (83.33%)** |
| `addition` | `13/33` (39.39%) | `26/33` (78.79%) | **`32/33` (96.97%)** |
| `occlusion` | `15/30` (50.00%) | `26/30` (86.67%) | **`28/30` (93.33%)** |
| `return_to_baseline` | `9/30` (30.00%) | `24/30` (80.00%) | **`29/30` (96.67%)** |

---

## 5. Conclusions & Production Detector Recommendation

1. **Best-performing detector:** **`YOLOv8m` (`yolov8m.onnx`)** performs best by a wide margin, raising detection recall from **27.12% → 90.40%** (`160/177`) and B3 state accuracy from **30.60% → 89.07%** (`163/183`), identity consistency from **26.82% → 89.39%**, and historical retrieval to **100% (`2/2`)**, while cutting false removals from `6` to `1` (`0.56%`).
2. **Meaningfulness vs dataset limitations:** The gain (`+58.47` percentage points in state accuracy across 183 evaluations, zero fine-tuning, zero threshold changes) is **practically decisive** because `YOLOv8m` resolves `book` (`0% → 100%`), `cell phone` (`3.3% → 100%`), `cup` (`16.7% → 100%`), and `potted plant` (`30% → 90%`) using pure pretrained COCO weights. However, `physical_pilot_v2` is a small 30-image single-desk dataset, and `YOLOv8m` still missed `bottle` in `stationary`, `movement`, and `removal` (`12/26` overall, `0/3` removal recall) and produced `11` spurious/duplicate tracks (mostly extra `book` predictions on the printed booklet under the plant).
3. **Inference cost:** `895.29 ms/frame` (~0.9s per visit on CPU, `99 MB` disk) is **acceptable** for ObjectMemory's periodic visit-based inspection workflow (not a 30-FPS video tracker), whereas `YOLOv8n` (`12.2 MB`) remains suitable for lightweight/fast environments.
4. **Recommendation:** **Adopt `YOLOv8m` (`yolov8m.onnx`) as the recommended high-accuracy CPU detector** for physical tabletop deployments (while keeping `YOLOv8n` frozen as the published baseline for prior synthetic/ETH benchmarks).
