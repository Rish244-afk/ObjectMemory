# ObjectMemory — Target-Class YOLO-World B3 Experiment Report

**Dataset:** `data/physical_pilot_v2/manifest_v2.json` (6 sequences, 30 observations, 183 annotations)  
**Experiment Script:** `scripts/run_physical_pilot_yoloworld_target_classes.py`  
**Raw Results JSON:** `research_validation/physical_pilot_v2_yoloworld_target_classes_results.json`  
**Constructor Mechanism Used (unmodified):** `YOLOWorldDetector(default_prompts=["potted plant", "book", "bottle", "scissors", "cell phone", "cup", "banana"])` (`default_confidence_threshold=0.15`), run with frozen `B3` config.

---

## 1. Per-Class Detection Counts & Recall (`IoU >= 0.50`, `conf >= 0.15`)

| Class | Predictions | TP / GT Visible | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: |
| `cell phone` | 29 | `29 / 30` | **96.67%** | 100.00% |
| `bottle` | 22 | `22 / 26` | **84.62%** | 100.00% |
| `potted plant` | 3 | `3 / 30` | **10.00%** | 100.00% |
| `book` | 0 | `0 / 28` | 0.00% | N/A |
| `scissors` | 0 | `0 / 30` | 0.00% | N/A |
| `cup` | 0 | `0 / 30` | 0.00% | N/A |
| `banana` | 0 | `0 / 3` | 0.00% | N/A |
| **Overall** | **54** | **`54 / 177`** | **30.51%** | **100.00%** |

---

## 2. 4-Way Aggregate Comparison

| Metric | YOLOv8n B3 | YOLO-World Default B3 | YOLO-World Target-Class B3 | Track B (Oracle Boxes) |
| :--- | :---: | :---: | :---: | :---: |
| **State Accuracy** | `56 / 183` (**30.60%**) | `22 / 183` (**12.02%**) | `58 / 183` (**31.69%**) | `140 / 183` (**76.50%**) |
| **Identity Consistency** | `48 / 179` (**26.82%**) | `19 / 179` (**10.61%**) | `54 / 179` (**30.17%**) | N/A |
| **False Removal Rate / Count** | `6 / 180` (**3.33%**) | `2 / 180` (**1.11%**) | `2 / 180` (**1.11%**) | N/A |
| **False Discovery Proportion** | `6 / 6` (**100.00%**) | `2 / 4` (**50.00%**) | `2 / 4` (**50.00%**) | N/A |
| **Removal Recall (entity-visit)** | `0 / 3` (**0.00%**) | `2 / 3` (**66.67%**) | `2 / 3` (**66.67%**) | Included in `22 / 30` |
| **Event Removal Recall** | `0 / 1` (**0.00%**) | `1 / 1` (**100.00%**) | `1 / 1` (**100.00%**) | N/A |
| **Change Confirmation Rate** | `0 / 3` (**0.00%**) | `2 / 3` (**66.67%**) | `2 / 3` (**66.67%**) | N/A |
| **Historical Retrieval Accuracy** | `0 / 0` (`None`) | `2 / 2` (**100.00%**) | `2 / 2` (**100.00%**) | N/A |
| **ID Switches** | `0` (`0 / 32`) | `0` (`0 / 13`) | `0` (`0 / 41`) | `0` |
| **Spurious Tracks / Duplicates** | `1` / `1` | `4` / `4` | **`0` / `0`** | `0` / `1` |
| **Registration Failures** | `0 / 24` | `0 / 24` | `0 / 24` | `0 / 24` |

---

## 3. Scenario-Wise Results (YOLO-World Target-Class B3)

| Scenario | State Acc | Identity Consistency | False Removals | Removal Recall | Confirmation Rate | Historical Retrieval | Spurious Tracks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stationary` | `10 / 30` (33.33%) | `9 / 30` (30.00%) | `0 / 30` | `0 / 0` | `0 / 0` | `0 / 0` | 0 |
| `movement` | `6 / 30` (20.00%) | `6 / 30` (20.00%) | `2 / 30` | `0 / 0` | `0 / 1` | `0 / 0` | 0 |
| `removal` | `12 / 30` (40.00%) | `9 / 26` (34.62%) | `0 / 27` | `2 / 3` | `1 / 1` | `0 / 0` | 0 |
| `addition` | `10 / 33` (30.30%) | `10 / 33` (30.30%) | `0 / 33` | `0 / 0` | `0 / 0` | `0 / 0` | 0 |
| `occlusion` | `10 / 30` (33.33%) | `10 / 30` (33.33%) | `0 / 30` | `0 / 0` | `0 / 0` | `0 / 0` | 0 |
| `return_to_baseline` | `10 / 30` (33.33%) | `10 / 30` (33.33%) | `0 / 30` | `0 / 0` | `1 / 1` | `2 / 2` | 0 |

---

## 4. Key Scientific Takeaway

- **Explicitly supplying the 7 target classes** eliminated all spurious tracks (`0`) and duplicate identities (`0`), boosted `cell phone` recall to **96.67% (`29/30`)** alongside `bottle` (**84.62%**, `22/26`), and raised overall B3 state accuracy from **12.02% → 31.69%** (surpassing `YOLOv8n`'s 30.60% while achieving **100% event removal recall** and **100% historical retrieval accuracy**).
- **Remaining gap vs Track B Oracle (76.50%):** `YOLOv8s-Worldv2` (`conf=0.15`) still had **0/30** recall on `scissors`, `cup`, `book`, and `banana` (`0/3`) from an overhead top-down viewpoint — whereas `YOLOv8n` excelled on `scissors` (30/30) and `banana` (3/3) while missing `bottle` (0/26) and `cell phone` (1/30). Overhead top-down detection recall remains the primary limiter relative to the 76.50% oracle upper bound.
