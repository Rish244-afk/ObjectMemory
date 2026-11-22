# Phase 5: Ground-Truth Evaluation Audit and Metric Reconciliation

> **Project**: ObjectMemory Studio  
> **Evaluation Audit Date**: 2026-10-20  
> **Dataset**: `data/real_workbench/manifest.json` (7 real annotated visits, 34 ground-truth physical object instances)  
> **Saved Prediction Source**: `scripts/phase4_evaluation_results.json`  
> **Audit Output**: `scripts/phase5_ground_truth_audit.json`

---

## 1. Executive Summary & Audit Mandate

Phase 5 audited the quantitative claims reported in Phase 4. All metrics were re-derived directly from saved raw predictions and manual ground-truth annotations without fabricating data or assuming prior reports were correct.

### Core Audit Discoveries

1. **Global Aggregate Metrics Verified**:
   - The overall True Positive (TP), False Positive (FP), False Negative (FN), Precision, Recall, and F1 calculations reported in Phase 4 were **mathematically verified and replicated exactly**:
     - **YOLOv8n (IoU @ 0.50)**: TP=30, FP=22, FN=4 $\rightarrow$ Precision: **57.7%**, Recall: **88.2%**, F1: **69.8%**.
     - **YOLO-World (IoU @ 0.50)**: TP=24, FP=5, FN=10 $\rightarrow$ Precision: **82.8%**, Recall: **70.6%**, F1: **76.2%**.
     - **YOLOv8n (IoU @ 0.30)**: TP=31, FP=21, FN=3 $\rightarrow$ Precision: **59.6%**, Recall: **91.2%**, F1: **72.1%**.
     - **YOLO-World (IoU @ 0.30)**: TP=25, FP=4, FN=9 $\rightarrow$ Precision: **86.2%**, Recall: **73.5%**, F1: **79.4%**.
2. **Defect in Phase 4 Per-Class Accounting Identified and Corrected**:
   - In Phase 4, per-class results were keyed by the raw predicted prompt name rather than the canonical ground-truth class. For example, when YOLO-World detected a mug as `"mug"`, that matched the ground-truth `"cup"` in the global matching loop. However, in the per-class table, the TP was credited to `"mug"` ($GT=0, TP=7$), while `"cup"` showed $GT=8, TP=0, FN=1$. The numbers did not sum to $GT$ ($0 + 1 \ne 8$).
   - **Correction**: In Phase 5, all predictions and ground-truth instances are mapped into a standardized canonical taxonomy (`cup`, `bottle`, `book`, `potted plant`, `scissors`). Under this canonical mapping, **$\text{TP} + \text{FN} = \text{GT}$ strictly holds for every single class across both detectors**.
3. **1-to-1 Matching Invariants Confirmed**:
   - Every ground-truth object was matched at most once.
   - Every detector prediction was counted as a True Positive at most once.
   - Predictions were ranked strictly by detector confidence descending prior to greedy bipartite matching.

---

## 2. Quantitative Ground-Truth Recalculation

### 2.1 Reconciled Overall Performance Matrix

| Metric | YOLOv8n (IoU @ 0.50) | YOLO-World (IoU @ 0.50) | YOLOv8n (IoU @ 0.30) | YOLO-World (IoU @ 0.30) |
| :--- | :---: | :---: | :---: | :---: |
| **Evaluated Visits** | 7 | 7 | 7 | 7 |
| **Total Ground Truth ($GT$)** | 34 | 34 | 34 | 34 |
| **True Positives ($TP$)** | 30 | 24 | 31 | 25 |
| **False Positives ($FP$)** | 22 | **5** | 21 | **4** |
| **False Negatives ($FN$)** | 4 | 10 | 3 | 9 |
| **$\mathbf{TP + FN = GT}$ Invariant** | **Verified (30+4=34)** | **Verified (24+10=34)** | **Verified (31+3=34)** | **Verified (25+9=34)** |
| **Precision** | 57.69% | **82.76%** | 59.62% | **86.21%** |
| **Recall** | **88.24%** | 70.59% | **91.18%** | 73.53% |
| **F1-Score** | 69.77% | **76.19%** | 72.09% | **79.37%** |

---

### 2.2 Reconciled Canonical Per-Class Metrics (IoU @ 0.50)

#### YOLOv8n Baseline (Fixed COCO Taxonomy, conf $\ge$ 0.25)
| Canonical Class | Ground Truth ($GT$) | True Positives ($TP$) | False Positives ($FP$) | False Negatives ($FN$) | Precision | Recall | F1-Score | Sum Check ($TP+FN=GT$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bottle`** | 5 | 5 | 0 | 0 | 100.0% | 100.0% | 1.0000 | **5 = 5 (PASS)** |
| **`cup`** | 8 | 7 | 0 | 1 | 100.0% | 87.5% | 0.9333 | **7+1 = 8 (PASS)** |
| **`book`** | 8 | 6 | 0 | 2 | 100.0% | 75.0% | 0.8571 | **6+2 = 8 (PASS)** |
| **`potted plant`** | 7 | 6 | 1 | 1 | 85.7% | 85.7% | 0.8571 | **6+1 = 7 (PASS)** |
| **`scissors`** | 6 | 6 | 0 | 0 | 100.0% | 100.0% | 1.0000 | **6 = 6 (PASS)** |
| *`dining table` (non-GT)* | 0 | 0 | 7 | 0 | 0.0% | 0.0% | 0.0000 | 0 = 0 (PASS) |
| *`laptop` (non-GT)* | 0 | 0 | 7 | 0 | 0.0% | 0.0% | 0.0000 | 0 = 0 (PASS) |
| *`vase` (non-GT)* | 0 | 0 | 7 | 0 | 0.0% | 0.0% | 0.0000 | 0 = 0 (PASS) |
| **Total / Macro** | **34** | **30** | **22** | **4** | **57.69%** | **88.24%** | **0.6977** | **30+4 = 34 (PASS)** |

#### YOLO-World v2 Small (Open Vocabulary with 12 prompts, conf $\ge$ 0.15)
| Canonical Class | Ground Truth ($GT$) | True Positives ($TP$) | False Positives ($FP$) | False Negatives ($FN$) | Precision | Recall | F1-Score | Sum Check ($TP+FN=GT$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bottle`** | 5 | 5 | 0 | 0 | 100.0% | 100.0% | 1.0000 | **5 = 5 (PASS)** |
| **`cup`** | 8 | 7 | 0 | 1 | 100.0% | 87.5% | 0.9333 | **7+1 = 8 (PASS)** |
| **`book`** | 8 | 7 | 4 | 1 | 63.6% | 87.5% | 0.7368 | **7+1 = 8 (PASS)** |
| **`potted plant`** | 7 | 5 | 1 | 2 | 83.3% | 71.4% | 0.7692 | **5+2 = 7 (PASS)** |
| **`scissors`** | 6 | 0 | 0 | 6 | 0.0% | 0.0% | 0.0000 | **0+6 = 6 (PASS)** |
| *`dining table` (non-GT)* | 0 | 0 | **0** | 0 | — | — | — | Cleanly Suppressed |
| *`laptop` (non-GT)* | 0 | 0 | **0** | 0 | — | — | — | Cleanly Suppressed |
| *`vase` (non-GT)* | 0 | 0 | **0** | 0 | — | — | — | Cleanly Suppressed |
| **Total / Macro** | **34** | **24** | **5** | **10** | **82.76%** | **70.59%** | **0.7619** | **24+10 = 34 (PASS)** |

---

## 3. Per-Visit Breakdown and Matching Inspection

Below is the verified per-image contingency audit for all 7 physical visits:

| Visit ID | Ground Truth | YOLOv8n ($TP/FP/FN$) | YOLO-World ($TP/FP/FN$) | Analysis of Discrepancies |
| :--- | :---: | :---: | :---: | :--- |
| **`visit_01`** | 6 | 5 / 3 / 1 | 4 / 1 / 2 | Both models miss small white cup (`cup_02`) occluded behind `cup_01`. YOLO-World also misses `scissors_01`. |
| **`visit_02`** | 5 | 5 / 3 / 0 | 4 / 1 / 1 | YOLOv8n achieves 100% recall. YOLO-World misses `scissors_01`. |
| **`visit_03`** | 5 | 4 / 3 / 1 | 3 / 1 / 2 | Plant bounding box confidence dropped slightly in YOLOv8n (borderline). YOLO-World misses plant and scissors. |
| **`visit_04`** | 5 | 4 / 3 / 1 | 4 / 1 / 1 | Spiral `notebook_01` introduced. Both models detect hardcover `book_01` but miss spiral `notebook_01`. |
| **`visit_05`** | 4 | 4 / 3 / 0 | 3 / 1 / 1 | YOLOv8n achieves 100% recall. YOLO-World misses `scissors_01`. |
| **`visit_06`** | 4 | 3 / 4 / 1 | 2 / 0 / 2 | Hardcover book missed by YOLOv8n; plant and scissors missed by YOLO-World. |
| **`visit_07`** | 5 | 5 / 3 / 0 | 4 / 0 / 1 | Bottle returns. YOLOv8n matches all 5; YOLO-World matches 4 (misses scissors). |

---

## 4. Analysis of False Positives & False Negatives

### 4.1 Root Causes of False Positives
1. **YOLOv8n (22 False Positives)**:
   - **`laptop` (7 FP)**: Across all 7 visits, YOLOv8n consistently flags the right desk corner/support as a laptop (confidence 0.89–0.92).
   - **`dining table` (7 FP)**: In every visit, the wooden workbench surface is detected as a dining table (confidence 0.45–0.55).
   - **`vase` (7 FP)**: In every visit, the ceramic plant pot is dual-detected as both a potted plant and a vase.
   - **`potted plant` (1 FP)**: High-confidence secondary leaf cluster in visit 04.
2. **YOLO-World (5 False Positives)**:
   - **Prompt Co-occurrence / Duplicate Book Detection (4 FP)**: Because the prompts included both `"book"` and `"notebook"`, YOLO-World fired on the navy hardcover book twice—once as `book` (conf ~0.35) and once as `notebook` (conf ~0.23). The greedy matcher matched `book` to `book_01`, leaving `notebook` as an unmatched duplicate FP.
   - **`potted plant` (1 FP)**: Dual leaf detection on plant pot.

### 4.2 Root Causes of False Negatives
1. **Scissors Misses in YOLO-World (6 FN)**:
   - YOLO-World Small failed to detect `scissors_01` in all 6 visits where it was physically present. The low-contrast red plastic handle on dark wood and the thin reflective metallic blades fell below the detection threshold for the 48 MB small model.
2. **Occluded Secondary Cup (`cup_02`) (1 FN)**:
   - Both models merged the small cylindrical white cup positioned directly behind `cup_01` in `visit_01` with `cup_01`.
3. **Spiral Notebook (`notebook_01`) (1 FN)**:
   - In `visit_04`, both models failed to detect the open spiral kraft notebook lying flat in the centre-bottom region.

---

## 5. Verification of Matching Invariants

The audit script `scripts/audit_phase5_ground_truth.py` confirmed:
- **No Duplicate GT Consumption**: Each of the 34 ground-truth objects was associated with at most one detector prediction.
- **No Duplicate Prediction Matches**: No single detection was credited as a True Positive more than once.
- **Strict Greedy Bipartite Matching**: Predictions were sorted by confidence descending before computing IoU against available ground-truth boxes.
- **Prompt Reproducibility**: The prompt list, confidence thresholds, and model artifact hashes are permanently logged in `scripts/phase5_ground_truth_audit.json`.
