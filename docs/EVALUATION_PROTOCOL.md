# ObjectMemory Evaluation Protocol

## 1. Overview and Core Philosophy

ObjectMemory is designed to maintain a persistent, explainable visual memory of physical environments across extended temporal intervals. Unlike standard video tracking or single-frame object detection, ObjectMemory addresses discrete, asynchronous visits where viewpoint, lighting, occlusion, and detector classifications fluctuate.

To ensure scientific integrity and prevent false claims:
1. **Strict Dataset Decoupling**: Synthetic pilot data (AI-generated crops) is strictly separated from physical real-world benchmarks.
2. **Conservative State Transitions**: Movements and state changes are never confirmed without sufficient spatial evidence and reliable registration.
3. **Explicit Epistemic Abstention**: When registration degrades or objects intersect image boundaries, the system explicitly holds state or reports `INSUFFICIENT_EVIDENCE` rather than fabricating change events.

---

## 2. Quantitative Evaluation Metrics

### 2.1 Object Detection Quality (Per-Visit)
Evaluates candidate proposal accuracy against manual ground-truth bounding boxes:
- **Detection Precision**:
  $$\text{Precision} = \frac{TP_{det}}{TP_{det} + FP_{det}}$$
- **Detection Recall**:
  $$\text{Recall} = \frac{TP_{det}}{TP_{det} + FN_{det}}$$
- **Detection F1**:
  $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Mean IoU**: Average Intersection-over-Union between predicted bounding boxes and matched ground-truth boxes ($IoU \ge 0.45$).

### 2.2 Temporal Change Detection Quality (Cross-Visit)
Evaluates system ability to confirm physical environmental alterations:
- **Change Precision**: Ratio of correctly identified transitions (`CONFIRMED_MOVED`, `CONFIRMED_REMOVED`, `REAPPEARED`) to all declared transitions.
- **Change Recall**: Ratio of detected changes to actual physical changes in the ground-truth sequence.
- **Change F1**: Harmonic mean of change precision and recall.
- **False Confirmed Removal Rate**:
  $$\text{FCR Rate} = \frac{\text{False Confirmed Removals}}{\text{Total Physically Present Evaluations}}$$
- **False Movement Rate**: Percentage of stationary objects incorrectly flagged as moved due to coordinate jitter or boundary clipping.

### 2.3 Registration Health & Gating
- **Registration Success Rate**: Percentage of query frames achieving $\ge 20\%$ SIFT inliers and valid planar homography.
- **Mean Reprojection Error**: Root-mean-square reprojection error (in canonical pixels) of matched feature pairs.
- **Gating Enforcement**: If inlier ratio $< 20\%$ or reprojection error $> 5.0\text{ px}$, planar registration is rejected. Observation canonical coordinates are flagged unreliable, and epistemic transitions are deferred.

### 2.4 Operational Latency Profile (CPU Only)
- **Mean Detector Latency** ($ms$): OpenCV DNN inference time per visit.
- **Mean Registration Latency** ($ms$): SIFT feature extraction, FLANN/RANSAC matching, and homography computation time.
- **Mean Total Pipeline Latency** ($ms$): Complete per-visit processing duration.
- **95th Percentile Latency** ($ms$): Upper tail latency.

---

## 3. Ground Truth Verification Workflow

1. **Physical Scene Setup**:
   - Fixed camera or controlled handheld position over planar workbench surface.
   - Natural ambient lighting with realistic variation (morning, afternoon, overhead artificial).
2. **Controlled Perturbations**:
   - Controlled object translations measured with metric ruler ($\Delta \ge 35\text{ px}$).
   - Object removals, temporary occlusions, and return/reappearances.
3. **Double-Blind Annotation**:
   - Independent annotator records bounding boxes and visibility states in `manifest.json`.
   - Annotator notes ambiguities (e.g. scissors obscured under Kraft notebook).
4. **Execution Harness**:
   - Run via `RealBenchmarkEvaluator` or API route `POST /api/evaluation/run`.
   - Results output in JSON/CSV format for auditability.
