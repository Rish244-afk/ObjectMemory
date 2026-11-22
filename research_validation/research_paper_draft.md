# ObjectMemory: Persistent Spatial-Temporal Visual Memory for Repeated Environment Observations

**Abstract**—*Autonomous service robotics and visual inspection systems operating over extended durations must determine whether objects observed across repeated visits are stationary, relocated, or removed. Standard object detectors operate memorylessly, treating missing detections as deletions and re-appearing objects as novel instances. We propose ObjectMemory, an epistemic framework for long-term tabletop scene persistence. ObjectMemory integrates planar feature registration, open-set CPU detection, multi-modal appearance modeling, optimal bipartite spatial association with ambiguity gating, and a conservative multi-visit removal policy validated by appearance vacancy verification. On a 25-observation photorealistic synthetic benchmark, ObjectMemory achieves a 0.0% false removal rate, 100.0% removal event recall, and 97.46% identity consistency. An ablation study demonstrates that bypassing appearance vacancy verification degrades state classification accuracy by 1.6%, isolating the empirical utility of visual background confirmation.*

**Index Terms**—Visual persistence, spatial-temporal memory, change detection, object association, robotic inspection.

---

## I. Introduction

Deploying object detection in physical environments over extended time horizons exposes a fundamental limitation of memoryless computer vision: frame-independent detectors cannot distinguish between an object that has been removed and an object that was merely missed due to brief occlusion, viewpoint variation, or sensor noise. In continuous operational workflows, treating single-frame non-detections as physical removals introduces substantial false discovery rates, while failing to recognize relocated objects causes identity fragmentation.

To address these challenges, we introduce **ObjectMemory**, an architecture for maintaining persistent spatial-temporal state over repeated observations. Rather than assuming that an object's existence depends entirely on current detector confidence, ObjectMemory models persistent identities as entities with explicit epistemic states (`CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`, and `INSUFFICIENT_EVIDENCE`).

---

## II. Methodology

### A. Coordinate Homography and Planar Registration
Each observation $I_t$ is aligned to a canonical reference image $I_{\text{ref}}$ via planar homography $H_t \in \mathbb{R}^{3 \times 3}$:
$$p_{\text{ref}} \sim H_t p_t$$
Features are extracted using SIFT with an ORB fallback. RANSAC filters outliers with a strict geometric evaluator rejecting singular matrices, concave corner envelopes, and low inlier ratios ($<20\%$). Bounding box envelopes are projected into canonical coordinates by mapping all four quadrilateral vertices.

### B. Appearance and Spatial Association
Detected objects are represented by bounding box coordinates $b = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ and a 148-dimensional normalized visual descriptor $v = [v_{\text{color}} \mid v_{\text{texture}}]$, comprising a 128-bin Hue-Saturation histogram and a 20-bin gradient orientation descriptor.

Cross-visit bipartite matching between historical entity set $\mathcal{E}$ and query detections $\mathcal{D}_t$ is computed via the Kuhn-Munkres algorithm over a combined cost matrix:
$$C_{ij} = w_{\text{IoU}} (1 - \text{IoU}(b_i, b_j)) + w_{\text{dist}} \frac{\|c_i - c_j\|_2}{D_{\max}} + w_{\text{app}} (1 - \text{sim}(v_i, v_j))$$
Subject to:
1. Spatial displacement gating: $\|c_i - c_j\| \le \tau_{\text{disp}}$.
2. Ambiguity margin: $\min_{k \ne j} |C_{ik} - C_{ij}| \ge \epsilon_{\text{amb}}$.

### C. Vacancy Verification and Removal Filtering
When an established entity $e \in \mathcal{E}$ is not detected in visit $t$:
1. If the coordinate is occluded by another detection or out of camera view, removal progression is suspended (`INSUFFICIENT_EVIDENCE`).
2. If the canonical bounding box is visible, the query image patch at $b_e$ is cropped and compared against the baseline patch.
3. If patch similarity indicates vacancy ($\text{sim} < \tau_{\text{vac}}$), the entity increments its unobserved streak counter.
4. Transition from `CONFIRMED_PRESENT` to `PENDING_REMOVAL` occurs at streak = 1; transition to `CONFIRMED_REMOVED` requires streak $\ge 2$.

---

## III. Experimental Results

We evaluated ObjectMemory across three distinct benchmark datasets:
1. **Synthetic Workbench v2**: 25 observations across 5 sequences (`S01` stationary, `S02` relocation, `S03` removal, `S04` occlusion, `S05` illumination/viewpoint shifts).
2. **Desk Pilot 01**: 7 physical sequential smartphone images of an office desk.
3. **OpenLORIS-Scene**: 10 keyframes of an indoor robotic traversal with OptiTrack ground truth. Evaluated strictly as a geometric boundary experiment on 2D planar homography under 3D depth parallax (8/9 transitions collapsed into singular homographies). The sequence lacks native object lifecycle ground truth and is not presented as an object-memory validation.

### Table I: Benchmark Performance Across Configurations

| Evaluation Metric | B0 (Memoryless) | B3 (ObjectMemory) | A5 (No Ambiguity) | A2\* (No Vacancy Verif.) |
| :--- | :---: | :---: | :---: | :---: |
| **Detection F1 Score** | **0.987** | *N/A* | *N/A* | *N/A* |
| **False Removal Rate (FRR)** | *N/A* | **0.00%** (0/121) | 0.00% (0/121) | 0.00% (0/121) |
| **False Discovery Prop. (FDP)** | *N/A* | **0.00%** (0/4) | 0.00% (0/4) | 0.00% (0/4) |
| **Removal Recall (Visits)** | *N/A* | **100.0%** (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| **Removal Recall (Events)** | *N/A* | **100.0%** (2/2) | 100.0% (2/2) | 100.0% (2/2) |
| **Confirmation Rate** | *N/A* | **100.0%** (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| **Identity Consistency** | *N/A* | **97.46%** (115/118) | 97.46% (115/118) | 97.46% (115/118) |
| **Fragmented Identities** | **0** | **0** | 0 | 0 |
| **Overall State Accuracy** | 0.904 (113/125) | **0.944** (118/125) | 0.944 (118/125) | **0.928** (116/125) |

*\* Denotes partial configuration due to engine fallback logic.*

---

## IV. Discussion and Ablation Analysis

### A. The Protective Role of Vacancy Verification
Ablation `A2*` evaluates the system when visual patch vacancy checks are forced to always evaluate as vacant. In this regime, state accuracy decreased from $94.40\%$ to $92.80\%$, resulting in two state misclassifications. This demonstrates that multi-visit streaks alone are vulnerable when transient background occlusions or registration errors produce unsegmented object fragments.

### B. Relocation Tracking vs. Re-Identification
In sequence `S02`, an object displaced by $\approx 198\text{px}$ was relocated in visit 2 and returned to its initial coordinate in visit 5. Complete ObjectMemory maintained continuous track identity (`fragmented_identities = 0`) and successfully logged both displacement and return events in the auditable change ledger.

---

## V. Conclusion

ObjectMemory establishes a verifiable bridge between short-term computer vision detections and long-term environmental persistence. By coupling planar coordinate normalization, multi-modal appearance extraction, and visual vacancy verification, the system eliminates false removal reports while reliably identifying object relocation and disappearance across repeated observations.
