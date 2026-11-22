# ObjectMemory — Final Presentation Structure & Slide Deck Outline

**Subject:** Final Project Oral Defense & Technical Presentation  
**Duration:** 20 Minutes Presentation + 10 Minutes Q&A  
**Audience:** Academic Committee, Faculty Assessors, Technical Stakeholders  

---

## Slide 1: Title & Overview
- **Title**: ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes
- **Subtitle**: Bridging the Gap Between Frame-by-Frame Detection and Long-Term Spatial Persistence
- **Presenter**: Engineering Candidate
- **Key Visual**: Split view showing raw frame detections vs. persistent spatial-temporal memory graph.

## Slide 2: The Problem: Memoryless Computer Vision
- **Core Dilemma**: Deep neural networks detect objects frame-by-frame without temporal context.
- **Three Critical Failure Modes**:
  1. *False Removals*: Momentary detection miss $\to$ False alarm deletion.
  2. *Identity Fragmentation*: Object moved and returned $\to$ Mistaken for new object.
  3. *Unexplainable Decisions*: Autonomous systems reporting changes without auditable evidence.
- **Target Domain**: Electronics workbenches, laboratory tables, and industrial workspaces.

## Slide 3: Proposed Architecture
- **Pipeline Overview**:
  1. *Planar Co-Registration*: SIFT/ORB + RANSAC homography normalization ($p_{\text{ref}} \sim H p_{\text{query}}$).
  2. *Lightweight CPU Inference*: OpenCV DNN YOLOv8n ($12.2\text{ MB}$, no GPU requirement).
  3. *Multi-Modal Appearance*: 148-d Color-Texture descriptor with cosine similarity.
  4. *Kuhn-Munkres Spatial Associator*: Optimal bipartite matching with displacement and ambiguity gating.
  5. *Epistemic State Engine*: `CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`.
  6. *Auditable Evidence Ledger*: Chronological transaction log in SQLite.

## Slide 4: Key Mechanism 1 — Displacement Gating & Canonical Homography
- **Why Homography?** Normalizes camera angle and position variations to a canonical reference plane.
- **Quadrant Envelope Projection**: All 4 bounding box vertices are mapped through $H$, forming canonical axis-aligned envelopes.
- **Displacement Gating**: Threshold $\tau_{\text{disp}} = 35.0\text{px}$ prevents minor vibration from registering as physical relocation.

## Slide 5: Key Mechanism 2 — Multi-Visit Vacancy Verification
- **Two-Stage Removal Protocol**:
  - Visit 1 Absent $\to$ Query patch compared with baseline $\to$ If vacant, transition to `PENDING_REMOVAL`.
  - Visit 2 Absent $\to$ Query patch verified vacant $\to$ Transition to `CONFIRMED_REMOVED`.
- **Occlusion Handling**: If the coordinate is occluded by another object or out-of-frame, removal progression is frozen under `INSUFFICIENT_EVIDENCE`.

## Slide 6: Experimental Methodology & Benchmark Suite
- **Three Evidence Tiers**:
  1. *Synthetic Workbench v2* (25 observations, 5 scenarios): Photorealistic composite cutouts on textured wood with contact shadows and ArUco fiducials.
  2. *Genuine Desk Pilot* (7 observations): Real smartphone photographs of an office desk.
  3. *OpenLORIS-Scene* (10 observations): Public indoor robotic trajectory used strictly to evaluate planar registration breakdown under 3D depth parallax (8/9 transitions collapsed; lacks object lifecycle ground truth).
- **Standardized Configurations**: B0 (Memoryless Baseline), B3 (ObjectMemory), A5 (No Ambiguity), A2\* (No Vacancy Verification), A3\* (No FOV Safeguards).

## Slide 7: Experimental Results & Findings
- **Highlight Table**:
  - Synthetic v2 FRR: **$0.00\%$** ($0/121$).
  - Synthetic v2 Event Removal Recall: **$100.0\%$** ($2/2$).
  - Synthetic v2 Confirmation Rate: **$100.0\%$** ($4/4$).
  - Synthetic v2 Identity Consistency: **$97.46\%$** ($115/118$).
  - State Accuracy: **$94.40\%$** (B3) vs. **$90.40\%$** (B0).
- **Ablation Proof**: Disabling appearance vacancy verification (`A2*`) lowered state accuracy from $94.40\%$ to $92.80\%$ ($2$ classification errors).

## Slide 8: Live System Demonstration
- **Video / Live Walkthrough**:
  1. Scene Ingestion: 5 objects detected, 5 entities instantiated.
  2. Relocation: Cup moved $198.9\text{px} \to \text{CONFIRMED\_MOVED}$ logged.
  3. Removal: Bottle removed $\to \text{PENDING\_REMOVAL} \to \text{CONFIRMED\_REMOVED}$.
  4. Occlusion: Folder placed over scissors $\to$ Occlusion frozen $\to$ Recovered upon removal.
  5. Insufficient Evidence: Featureless image $\to \text{INSUFFICIENT\_INLIERS} \to$ Absence freezing.

## Slide 9: Full-Stack Web Application
- **Architecture**: FastAPI backend + React/TypeScript single-page app (SPA).
- **Features**:
  - Interactive bounding box canvas with canonical reprojection toggles.
  - Interactive before/after split slider for visual change inspection.
  - Evidence ledger showing chronological explainable events with confidence scores.
  - Multi-workspace isolation with SQLite database persistence.

## Slide 10: Technical Limitations & Future Horizons
- **Limitations**:
  - Requires dominant planar surface (fails under unconstrained 3D multi-room translation, as shown on OpenLORIS).
  - Upper bound set by detector recall streak limits.
  - Handcrafted descriptors sensitive to extreme colored lighting shifts.
- **Future Directions**:
  - RGB-D 3D point cloud / TSDF registration.
  - Quantized CPU-friendly foundation model embeddings (e.g. INT8 MobileNetV4).
  - Active visual verification via robotic arm manipulation.

## Slide 11: Summary & Conclusion
- **Achievements**:
  - Complete, mathematically grounded visual persistence framework.
  - Zero false removals ($0.0\%$) and zero identity fragmentation ($0$) on controlled benchmark.
  - 401 automated tests passing with 0 regressions.
  - Full-stack web application and reproducible CLI test harness.
- **Q&A Session**.
