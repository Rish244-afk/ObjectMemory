# ObjectMemory — Detector vs Temporal Reasoning Diagnostic Report

**Dataset:** `data/physical_pilot_v2/manifest_v2.json` (6 sequences, 30 observations, 183 annotations)  
**Saved Result Artifacts (B3 untouched):**
- `research_validation/physical_pilot_v2_b3_results.json` (original B3 `YOLOv8n`, untouched)
- `research_validation/physical_pilot_v2_track_b_oracle_results.json` (Track B Oracle Boxes via `evaluate_physical_track_b`)
- `research_validation/physical_pilot_v2_yoloworld_b3_results.json` (B3 with `YOLOWorldDetector` default prompts/thresholds)

---

## 1. Configuration & Track Definitions (Verified from Code)

- **`B3` (`BenchmarkConfigFactory.create_config("B3")` + `BenchmarkRunner`):** Full `TemporalMemoryEngine` with all safeguards enabled (`min_registration_inlier_ratio=0.15`, `vacancy_visits_for_removal=2`, `movement_displacement_threshold=35.0`, `ambiguity_margin`) using `OpenCVDNNDetector(weights_path="models/yolov8n.onnx")` (`conf=0.25`).
- **`Track B` (`scripts/evaluate_physical_workbench.py:evaluate_physical_track_b`):** Oracle-box diagnostic that bypasses the object detector by passing ground-truth `ann.bbox` (`confidence=1.0`, `VisualSignatureExtractor` crops) directly into `TemporalMemoryEngine` with real `PlanarRegistrar` homography registration.
- **Alternative Detector (`YOLOWorldDetector` + `BenchmarkRunner` B3):** Runs the exact same `BenchmarkRunner` and `B3` temporal engine, swapping `OpenCVDNNDetector` for `YOLOWorldDetector()` with its unmodified defaults (`default_confidence_threshold=0.15`, `DEFAULT_WORLD_CLASSES`, which includes `"bottle"`, `"cup"`, `"book"`, `"scissors"`, `"phone"`, `"notebook"`, but omits `"potted plant"`, `"cell phone"`, and `"banana"`).

---

## 2. Aggregate Comparison

| Metric | B3 (`YOLOv8n`, `conf=0.25`) | Track B (Oracle Boxes) | B3 (`YOLOWorldDetector` defaults) |
| :--- | :---: | :---: | :---: |
| **State Accuracy** | `56 / 183` (**30.60%**) | `140 / 183` (**76.50%**) | `22 / 183` (**12.02%**) |
| **Identity Consistency** | `48 / 179` (**26.82%**) | N/A (Track B uses `id_switches`) | `19 / 179` (**10.61%**) |
| **False Removal Rate / Count** | `6 / 180` (**3.33%**) | N/A | `2 / 180` (**1.11%**) |
| **False Discovery Proportion** | `6 / 6` (**100.00%**) | N/A | `2 / 4` (**50.00%**) |
| **Removal Recall (entity-visit)** | `0 / 3` (**0.00%**) | Included in `22/30` removal seq | `2 / 3` (**66.67%**) |
| **Event Removal Recall** | `0 / 1` (**0.00%**) | N/A | `1 / 1` (**100.00%**) |
| **Change Confirmation Rate** | `0 / 3` (**0.00%**) | N/A | `2 / 3` (**66.67%**) |
| **Historical Retrieval Accuracy** | `0 / 0` (`None`, unscorable) | N/A | `2 / 2` (**100.00%**) |
| **ID Switches** | `0` (`0 / 32` transitions) | `0` | `0` (`0 / 13` transitions) |
| **Spurious Tracks / Duplicates** | `1` / `1` | `0` / `1` (Track B `dup=1`) | `4` / `4` |
| **Registration Failures** | `0 / 24` | `0 / 24` | `0 / 24` |

---

## 3. Scenario-Wise Comparison

| Scenario (`physical_pilot_v2_*`) | B3 (`YOLOv8n`) State Acc | Track B (Oracle) State Acc | `YOLOWorld` State Acc | Key Observation |
| :--- | :---: | :---: | :---: | :--- |
| `stationary` | `5 / 30` (16.67%) | `24 / 30` (80.00%) | `3 / 30` (10.00%) | Track B is 24/24 (100%) on V02–V05 (6 V01s initialize in `INSUFFICIENT_EVIDENCE`). |
| `movement` | `7 / 30` (23.33%) | `20 / 30` (66.67%) | `1 / 30` (3.33%) | `YOLOv8n` missed `bottle_01` 0/5; `YOLOWorld` saw `bottle_01` (`frr=2/30`, `cr=0/1`). |
| `removal` | `7 / 30` (23.33%) | `22 / 30` (73.33%) | `3 / 30` (10.00%) | `YOLOWorld` achieved `rr=2/3` and `cr=1/1` on `bottle_01`; `YOLOv8n` missed `bottle_01` 0/5. |
| `addition` | `13 / 33` (39.39%) | `26 / 33` (78.79%) | `5 / 33` (15.15%) | `YOLOv8n` detected `banana_01`; `YOLOWorld` lacked `"banana"` in `DEFAULT_WORLD_CLASSES`. |
| `occlusion` | `15 / 30` (50.00%) | `24 / 30` (80.00%) | `5 / 30` (16.67%) | Track B is 24/24 (100%) on V02–V05 (`dup=1`). |
| `return_to_baseline` | `9 / 30` (30.00%) | `24 / 30` (80.00%) | `5 / 30` (16.67%) | `YOLOWorld` achieved `cr=1/1` and `hra=2/2` (100%) on `bottle_01`. |

---

## 4. Conclusion: Bottleneck Attribution

1. **Object Detection is the #1 Bottleneck (Detector-Related Failures):**
   - Supplying oracle boxes (Track B) jumps state accuracy from **30.60% (`56/183`) → 76.50% (`140/183`)**, with **0 identity switches**.
   - `YOLOv8n` (`conf=0.25`) misses `book_01` and `bottle_01` in 0/30 frames and intermittently drops `plant_01` and `cup_01`, directly causing all 6 false removals (`FDP = 100%`).
   - `YOLOWorldDetector` proves `bottle_01` can be detected and tracked (`100%` event removal recall, `100%` historical retrieval accuracy), but its unmodified `DEFAULT_WORLD_CLASSES` omits `"potted plant"`, `"cell phone"`, and `"banana"` and uses `"phone"` instead of `"cell phone"`.
2. **Registration is NOT a Bottleneck:** `0 / 24` registration failures across all runs.
3. **Association is NOT a Primary Bottleneck:** `id_switches = 0` across all three runs (`B3`, `Track B`, `YOLOWorld`).
4. **Temporal Reasoning Works as Designed When Detections Exist:** Track B achieves 24/24 (100%) on `V02`–`V05` in `stationary`, `occlusion`, and `return_to_baseline` (36 of the 43 Track B mismatches are simply Visit-1 `INSUFFICIENT_EVIDENCE` initialization).

**Single Most Important Next Improvement:**  
Pass the dataset's target class prompts (`["potted plant", "book", "bottle", "scissors", "cell phone", "cup", "banana"]`) to `YOLOWorldDetector` (via `DetectorConfiguration.target_classes`) — or use a stronger overhead-capable detector — so all 7 physical object classes are reliably detected without false drop-outs.
