# ObjectMemory — Physical Pilot v2 (B3) Benchmark Report

**Dataset:** `data/physical_pilot_v2/manifest_v2.json` (`GENUINE_PHYSICAL`, 6 sequences, 30 observations, 183 annotations)  
**Configuration:** `B3` (`OpenCVDNNDetector` with `models/yolov8n.onnx`, default `conf=0.25`, frozen thresholds)  
**Raw Results JSON:** `research_validation/physical_pilot_v2_b3_results.json`  
**Execution Status:** Completed (`exit code 0`, `runtime_seconds = 32.114`, `execution_errors = {}`, `support_status = SUPPORTED`)

> **Timestamp Note:** Manifest timestamps are synthetic monotonic ISO-8601 ordering timestamps because EXIF capture timestamps are absent in the source JPEGs. They do not represent real recording times.

---

## 1. Console Output & Warnings

- **OpenCV warning (non-fatal):**
  `[ WARN:0@1.063] global net_impl_backend.cpp:345 cv::dnn::dnn5_v20260605::Net::Impl::setPreferableTarget Targets are not supported by the new graph engine for now`
- **Observations processed:** `30 / 30` (0 failed observations, 0 skipped observations).

---

## 2. Aggregate B3 Metrics & Exact Denominators

| Metric | Value | Numerator / Denominator |
| :--- | :---: | :---: |
| `overall_state_accuracy` | **0.3060** | `56 / 183` |
| `identity_consistency` | **0.2682** | `48 / 179` |
| `false_removal_rate` | **0.0333** | `6 / 180` |
| `false_discovery_proportion` | **1.0000** | `6 / 6` |
| `removal_recall` | **0.0000** | `0 / 3` |
| `event_removal_recall` | **0.0000** | `0 / 1` |
| `confirmation_rate` | **0.0000** | `0 / 3` |
| `unsupported_movement_rate` | **0.0000** | `0 / 172` |
| `id_switches` / `id_switch_rate` | **0 / 0.0** | `0 / 32` transitions |
| `fragmented_identities` | **0** | — |
| `spurious_system_tracks` | **1** | — |
| `duplicate_identities` | **1** | — |
| `mean_confirmation_delay_visits` | `None` | `0` confirmed events (unscorable) |
| `historical_retrieval_accuracy` | `None` | `0 / 0` (unscorable) |

---

## 3. Scenario-Wise Results

| Sequence (`physical_pilot_v2_*`) | Obs | State Acc (`correct / evals`) | Identity Consistency (`num / den`) | False Removals (`num / den`) | Confirmation Rate (`num / den`) | Removal Recall (`num / den`) | Spurious Tracks | Uninstantiated GT Entities |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `stationary` | 5 | `5 / 30` (0.1667) | `5 / 30` (0.1667) | `0 / 30` | `0 / 0` | `0 / 0` | 0 | `book_01`, `bottle_01`, `cup_01`, `phone_01`, `plant_01` |
| `movement` | 5 | `7 / 30` (0.2333) | `6 / 30` (0.2000) | `0 / 30` | `0 / 1` | `0 / 0` | 0 | `book_01`, `bottle_01`, `cup_01`, `phone_01` |
| `removal` | 5 | `7 / 30` (0.2333) | `6 / 26` (0.2308) | `2 / 27` | `0 / 1` | `0 / 3` | 0 | `book_01`, `bottle_01`, `cup_01`, `phone_01` |
| `addition` | 5 | `13 / 33` (0.3939) | `12 / 33` (0.3636) | `1 / 33` | `0 / 0` | `0 / 0` | 0 | `book_01`, `bottle_01`, `phone_01` |
| `occlusion` | 5 | `15 / 30` (0.5000) | `12 / 30` (0.4000) | `1 / 30` | `0 / 0` | `0 / 0` | 0 | `book_01`, `bottle_01` |
| `return_to_baseline` | 5 | `9 / 30` (0.3000) | `7 / 30` (0.2333) | `2 / 30` | `0 / 1` | `0 / 0` | 1 | `book_01`, `bottle_01`, `phone_01` |

---

## 4. Failure Diagnosis

1. **Registration Failures:** **None (0 / 24)** — all query visits achieved `AlignmentStatus.SUCCESS` with inlier ratios `0.509`–`1.000` (well above `0.15`). No observations were skipped for insufficient evidence.
2. **Detection Failures (Primary Bottleneck):** `OpenCVDNNDetector` (`yolov8n.onnx` at default `conf=0.25`) failed to detect most overhead desk objects:
   - `book_01` and `bottle_01` were detected in **0 / 30** frames (`uninstantiated_gt_entities`), preventing B3 from observing `bottle_01` movement (`movement`), removal (`removal`), or return-to-baseline (`return_to_baseline`), and preventing `book_01` occlusion tracking (`occlusion`).
   - `phone_01` was detected in only **1 / 30** frames (`occlusion_V02`).
   - `cup_01` was detected in only **5 / 30** frames.
   - `potted plant` was detected near threshold (`0.26`–`0.51`) in only **8 / 30** frames.
   - **Misclassification / Spurious Track:** In `return_to_baseline_V02`, the relocated black water bottle was misclassified as a second `'scissors'` (`conf=0.41`), producing `spurious_system_tracks = 1`.
3. **Incorrect Confirmed Change Events (`false_discovery_proportion = 1.0`, `6 / 6`):**
   - Intermittent detections of `plant_01` and `cup_01` followed by consecutive detector drop-outs under `AlignmentStatus.SUCCESS` caused the temporal engine to emit **6 false `CONFIRMED_REMOVED` evaluations** (`removal`: 2, `addition`: 1, `occlusion`: 1, `return_to_baseline`: 2).
4. **Association Errors:** `id_switches = 0` across 32 tracked transitions (`scissors_01` and `banana_01` were tracked cleanly when detected).

---

## 5. Recommended Next Action

1. **Recapture the 5 duplicate/shared images** (`addition/V01`, `addition/V02`, `occlusion/V01`, `occlusion/V04`, `occlusion/V05`) so all 6 sequences are 100% independent captures.
2. **Run an isolated comparison using `YOLOWorldDetector` (or Track A + Track B oracle evaluation)** alongside `yolov8n` to separate **detector recall limitations on overhead desk imagery** from **ObjectMemory's temporal reasoning accuracy** without modifying frozen B3 baselines or thresholds.
