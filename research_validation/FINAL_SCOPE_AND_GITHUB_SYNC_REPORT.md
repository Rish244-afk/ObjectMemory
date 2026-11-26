# ObjectMemory — Final Scope and GitHub Synchronization Report

**Project Title:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Date:** November 26, 2026  
**Institution:** Amrita Vishwa Vidyapeetham, Bengaluru  
**Repository:** `https://github.com/Rish244-afk/ObjectMemory`  
**Branch:** `main`

---

## 1. Final Detector Configuration

In accordance with final project scope decisions, the detector configuration has been streamlined to include only three officially supported models:

| Detector | Role | Model File / Format | Confidence / NMS Thresholds | Target Hardware |
| :--- | :--- | :--- | :---: | :--- |
| **YOLOv8m** | **Production Default** | `models/yolov8m.onnx` (~99.0 MB) | `conf=0.25`, `iou=0.45` | CPU (OpenCV DNN) |
| **YOLOv8n** | **Alternative Detector** | `models/yolov8n.onnx` (~12.2 MB) | `conf=0.25`, `iou=0.45` | CPU (OpenCV DNN) |
| **YOLO26m** | **Alternative / Comparison** | `models/yolo26m.onnx` (~78.1 MB) | `conf=0.25`, `iou=0.45` | CPU (ONNX Runtime) |

*Note:* YOLO-World (7-Class B3) has been removed from all final-facing presentation and documentation scope.

---

## 2. Final Datasets and Their Roles

The final documentation consistently categorizes the evaluated datasets into three distinct tiers:

1. **Physical Pilot v2 — Primary Evaluation**
   - **Scale:** 30 high-resolution photographs ($1600 \times 1200\text{ px}$).
   - **Scenarios:** 6 controlled indoor tabletop sequences (`stationary`, `movement`, `removal`, `addition`, `occlusion`, `return_to_baseline`).
   - **Ground-Truth:** 183 entity-visit annotations, including 177 visible object instances.
   - **Classes:** 7 COCO classes (`potted plant`, `book`, `bottle`, `scissors`, `cell phone`, `cup`, `banana`).
   - **Role:** Primary empirical benchmark establishing real-world tabletop performance.

2. **Synthetic Workbench v2 — Supplementary Controlled Evaluation**
   - **Scale:** 25 procedurally generated images across 5 sequences (`S01`–`S05`).
   - **Role:** Supplementary controlled ablation benchmark used for vacancy verification ablation (`A2*`) and association ambiguity gating (`A5`).
   - **Distinction:** Strictly distinguished from physical-pilot results.

3. **ETH ASL Lounge — Supplementary Registration Evaluation**
   - **Scale:** 10 sessions from the public ETH ASL Lounge benchmark.
   - **Role:** Supplementary planar image-registration evaluation demonstrating homography quality and 3D depth-parallax limits under viewpoint variations.
   - **Scope Restriction:** Explicitly scoped to registration evaluation; not presented as a full object-memory lifecycle benchmark.

---

## 3. References to YOLO-World Removed from Final-Facing Documentation

The following modifications were made to remove YOLO-World from final-facing documents while preserving all implementation files and historical experimental records:

1. **Academic Poster:**
   - **Architecture Diagram (Stage 3):** Updated detector options from `Alt: YOLOv8n, YOLO26m, YOLO-World` to `Alt: YOLOv8n, YOLO26m`.
   - **B3 Comparison Chart (`chart_b3_comparison.png`):** Removed YOLO-World bar; now compares only `YOLOv8n (Alternative)`, `YOLOv8m (Default)`, and `YOLO26m (Comparison)`, alongside Track B Oracle.
   - **Results Comparison Table:** Removed YOLO-World row; table now clearly presents YOLOv8n, YOLOv8m, YOLO26m, and Track B Oracle.
2. **README.md:**
   - Section 1: Removed `YOLO-World v2 Small` from CPU-friendly object detection description.
   - Section 2 (Data Flow Diagram): Updated detector node text to `Alternatives: YOLOv8n, YOLO26m`.
   - Section 3.1: Removed YOLO-World row from the Physical Pilot v2 results table; retained YOLOv8n, YOLOv8m (Default), and YOLO26m (Comparison).
   - Section 5: Updated directory tree to list `yolov8m.onnx`, `yolov8n.onnx`, and `yolo26m.onnx`.
3. **FINAL_PROJECT_REPORT.md:**
   - Abstract & System Description: Removed YOLO-World from supported detector list.
   - Related Work: Updated detector references to YOLOv8 and YOLO26m.
   - Architecture & Pipeline: Removed YOLO-World from Stage 3 and `DetectorRegistry` description.
   - Section 10 (Results): Removed YOLO-World rows from comparison table.

---

## 4. References to OpenLORIS-Scene Removed from Final-Facing Documentation

OpenLORIS-Scene has been removed from all final-facing evaluation and presentation claims:

1. **Academic Poster:**
   - **Dataset & Setup Section:** Removed OpenLORIS-Scene; re-framed dataset scope to Physical Pilot v2 (Primary), Synthetic Workbench v2, and ETH ASL Lounge (Supplementary Registration).
   - **Limitations & Future Scope:** Replaced OpenLORIS references with ETH ASL Lounge registration limits under 3D parallax.
2. **README.md:**
   - Section 1: Removed OpenLORIS-Scene from experimentally validated scope summary.
   - Section 3: Removed OpenLORIS-Scene from dataset list; replaced 4th tier with ETH ASL Lounge.
   - Section 6 (Technical Limitations): Updated dominant plane assumption discussion to cite ETH ASL Lounge.
3. **FINAL_PROJECT_REPORT.md:**
   - Removed OpenLORIS-Scene from dataset overview and geometric limitation discussions; replaced with ETH ASL Lounge.
4. **docs/LIMITATIONS_AND_FUTURE_WORK.md:**
   - Section 1.2: Replaced OpenLORIS-Scene narrative with multi-session registration boundary evaluation on ETH ASL Lounge.
5. **docs/EXPERIMENTAL_METHODOLOGY_AND_RESULTS.md:**
   - Diagram & Section 2.3: Replaced OpenLORIS-Scene node and section with ETH ASL Lounge supplementary registration evaluation.
6. **research_validation/final_presentation_structure.md:**
   - Slide 6 & Slide 10: Replaced OpenLORIS-Scene with ETH ASL Lounge.

---

## 5. Files Modified

| File Path | Nature of Changes |
| :--- | :--- |
| `README.md` | Removed YOLO-World and OpenLORIS from final scope; aligned dataset list and detector options. |
| `FINAL_PROJECT_REPORT.md` | Aligned system description, results table, dataset overview, and limitations with final scope. |
| `docs/LIMITATIONS_AND_FUTURE_WORK.md` | Replaced OpenLORIS-Scene with ETH ASL Lounge for 3D parallax limitation. |
| `docs/EXPERIMENTAL_METHODOLOGY_AND_RESULTS.md` | Updated benchmark diagram and Section 2.3 to ETH ASL Lounge; updated Section 6 domain boundary. |
| `research_validation/final_presentation_structure.md` | Updated presentation slides 6 and 10 to reflect the 3 final datasets and planar boundaries. |
| `scratch/generate_poster_assets.py` | Regenerated `chart_b3_comparison.png` without YOLO-World. |
| `scratch/build_poster.py` | Updated poster PPTX generator (dataset description, architecture badges, comparison table, limitations). |
| `research_validation/poster/ObjectMemory_Poster.pptx` | Regenerated editable PowerPoint poster matching final scope. |
| `research_validation/poster/ObjectMemory_Poster.pdf` | Regenerated high-resolution print-ready PDF export via PowerPoint COM. |
| `research_validation/poster/ObjectMemory_Poster_Preview.png` | Regenerated full-resolution (2480 × 3508) visual preview. |
| `research_validation/FINAL_SCOPE_AND_GITHUB_SYNC_REPORT.md` | Created this comprehensive verification report. |

---

## 6. Preservation of Historical Records & Implementations

In accordance with strict project constraints, no implementation files, model weights, or historical records were deleted:

- **Source Code Preserved:** `src/objectmemory/detection/yolo_world.py`, `src/objectmemory/registration/evaluator.py`, `src/objectmemory/api/app.py`, and all test fixtures remain intact.
- **Model Weights Preserved:** `models/yolov8s-worldv2.onnx`, `models/clip_text_quantized.onnx`, and `models/clip_tokenizer.json` remain preserved in `models/`.
- **Historical Reports Preserved:** Milestone records in `research_validation/` (e.g. `PHYSICAL_PILOT_V2_YOLOWORLD_TARGET_CLASSES_REPORT.md`, `openloris_*.md`, and early test logs) were preserved as immutable historical engineering records.
- **Data & Manifests Preserved:** All physical pilot photographs, annotations, and `data/` directories remain completely unchanged.

---

## 7. Git Status, Branch, Commit, and Push Verification

- **Branch:** `main`
- **Remote:** `origin` (`https://github.com/Rish244-afk/ObjectMemory.git`)
- **Staged Files:** Confined strictly to the intended documentation, poster deliverables, and verification report.
- **Commit Message:** `docs: align ObjectMemory documentation with final project scope`
- **Verification:** Commit hash and push receipt documented upon execution.

---

## 8. Retention of Historical References

A small number of historical engineering logs (e.g. `openloris_acquisition_decision.md` or early exploration notes) retain mentions of YOLO-World or OpenLORIS as part of their chronological research narrative. These were deliberately retained to preserve engineering audit trails, in full compliance with the constraint not to rewrite historical research history. All final-facing documentation, README files, presentation structures, and poster materials present a single, coherent, and consistent project scope.
