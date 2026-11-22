# ObjectMemory — Complete Code-Level Implementation Audit

**Audit Date:** November 5, 2026  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Target System:** ObjectMemory (Long-Term Visual Memory for Detecting Persistent Environmental Changes)  
**Audit Standard:** Read-only inspection of source code, database schemas, API schemas, frontend views, and test suites. Every finding is verified against actual implementation lines.

---

## Executive Summary of Code-Level Findings

| Capability Component | Implementation Status | Primary Implementation Files | Verification Test Suites |
| :--- | :--- | :--- | :--- |
| **A1. Object Detection** | **Verified Implemented** | `src/objectmemory/detection/detector.py`<br>`src/objectmemory/detection/yolo_world.py`<br>`src/objectmemory/detection/letterbox.py` | `tests/test_detection.py`<br>`tests/test_yolo_world_detector.py`<br>`tests/test_letterbox_preprocessing.py` |
| **A2. Image Registration** | **Verified Implemented** | `src/objectmemory/registration/registrar.py`<br>`src/objectmemory/registration/evaluator.py`<br>`src/objectmemory/registration/keyframe.py` | `tests/test_registration.py`<br>`tests/test_keyframe_registration.py`<br>`tests/test_boundary_and_registration.py` |
| **A3. Raw Evidence Preservation** | **Verified Implemented** | `src/objectmemory/domain/models.py`<br>`src/objectmemory/persistence/schema.py`<br>`src/objectmemory/persistence/repository.py` | `tests/test_live_camera_persistence.py`<br>`tests/test_persistence.py` |
| **A4. Object Association & Identity** | **Verified Implemented** | `src/objectmemory/temporal/associator.py`<br>`src/objectmemory/detection/signature.py`<br>`src/objectmemory/baseline/comparator.py` | `tests/test_ground_truth_matcher.py`<br>`tests/test_temporal_cross_class_association.py` |
| **A5. Temporal Reasoning & States** | **Verified Implemented** | `src/objectmemory/temporal/engine.py`<br>`src/objectmemory/domain/states.py`<br>`src/objectmemory/temporal/models.py` | `tests/test_temporal_engine.py`<br>`tests/test_temporal_boundary_reappearance.py`<br>`tests/test_temporal_robustness.py` |
| **A6. Persistent Long-Term Memory** | **Verified Implemented** | `src/objectmemory/persistence/repository.py`<br>`src/objectmemory/persistence/schema.py`<br>`src/objectmemory/api/workspace_manager.py` | `tests/test_persistence.py`<br>`tests/test_api_backend.py` |
| **A7. Live Camera Monitoring** | **Verified Implemented** *(Discrete Snapshot Ingestion)* | `frontend/src/views/WorkspaceOverviewView.tsx`<br>`src/objectmemory/api/workspace_manager.py` | `tests/test_live_camera_persistence.py`<br>`tools/test_live_upload.py` |
| **A8. Dataset Replay** | **Verified Implemented** | `src/objectmemory/datasets/real_dataset.py`<br>`src/objectmemory/pipeline/real_pipeline.py`<br>`scripts/evaluate_physical_workbench.py` | `tests/test_physical_workbench_dataset.py`<br>`tests/test_real_detector_pipeline.py` |

---

## A1. Object Detection

### 1.1 Architecture & Models
The object detection subsystem is decoupled via the abstract base class `ObjectDetector` defined in [`src/objectmemory/detection/detector.py`](src/objectmemory/detection/detector.py#L57-L109).

Two primary detector backends are implemented:
1. **OpenCV DNN YOLOv8n (`OpenCVDNNDetector`)**:
   - **File:** [`src/objectmemory/detection/detector.py`](src/objectmemory/detection/detector.py#L203-L450)
   - **Weights:** Pretrained ONNX model stored at `models/yolov8n.onnx` (SHA-1: `68f864475d06e2ec4037181052739f268eeac38d`).
   - **Taxonomy:** Standard COCO 80-class taxonomy ([`detector.py#L43-L54`](src/objectmemory/detection/detector.py#L43-L54)).
   - **Inference Runtime:** `cv2.dnn.readNetFromONNX` on CPU (`cv2.dnn.DNN_BACKEND_OPENCV`, `cv2.dnn.DNN_TARGET_CPU`).
   - **Integrity Rule:** If weights are missing, it raises an explicit `DetectorUnavailableError` or `FileNotFoundError` without fabricating synthetic detections ([`detector.py#L265-L275`](src/objectmemory/detection/detector.py#L265-L275)).
2. **YOLO-World Open-Vocabulary Detector (`YOLOWorldDetector`)**:
   - **File:** [`src/objectmemory/detection/yolo_world.py`](src/objectmemory/detection/yolo_world.py#L45-L240)
   - **Architecture:** Open-vocabulary object detector with dual text-visual vision-language heads.
   - **Capabilities:** Takes arbitrary comma-separated open-vocabulary category prompts (e.g. `drawing compass`, `pliers`, `soldering iron`) and evaluates candidate regions dynamically.

### 1.2 Image Preprocessing & Coordinate Transformation
- **Letterboxing:** [`src/objectmemory/detection/letterbox.py`](src/objectmemory/detection/letterbox.py#L30-L105) implements uniform aspect-ratio scaling to $640 \times 640$ with symmetric padding.
- **Inverse Coordinate Restoration:** `inverse_transform_cxcywh()` subtracts padding offsets $(dw, dh)$ and divides by the scale factor $r = \min(w_{target}/w, h_{target}/h)$ to restore bounding boxes strictly to the native query frame coordinates $[x_{min}, y_{min}, x_{max}, y_{max}]$.
- **Domain Safeguards:** `BoundingBox` ([`src/objectmemory/domain/models.py#L71-L100`](src/objectmemory/domain/models.py#L71-L100)) validates coordinate bounds:
  - Rejects `NaN`, $+\infty$, $-\infty$.
  - Enforces $x_{min} \le x_{max}$, $y_{min} \le y_{max}$.
  - Enforces $x_{min} \ge 0, y_{min} \ge 0$.
  - Clamps and discards degenerate boxes with zero or negative area.

### 1.3 Confidence Thresholding & Filtering
- Thresholds are passed via `DetectorConfiguration(confidence_threshold=...)`:
  - Default API threshold: `0.25` (configurable up to `0.80`).
  - Default physical workbench evaluation threshold: `0.35`.
- Non-Maximum Suppression (NMS): Evaluated via `cv2.dnn.NMSBoxes` with an IoU threshold of `0.45` (`nms_threshold=0.45`).

---

## A2. Image Registration

### 2.1 Feature Extraction, Matching, and Homography
- **Engine:** `PlanarRegistrar` in [`src/objectmemory/registration/registrar.py`](src/objectmemory/registration/registrar.py#L29-L290).
- **Feature Extractors:**
  - SIFT (`cv2.SIFT_create(nfeatures=2000)`) with $L_2$ norm.
  - Automatic fallback to ORB (`cv2.ORB_create(nfeatures=2000)`) with Hamming norm if SIFT is unavailable in the OpenCV build.
- **Descriptor Matching:** `cv2.BFMatcher` performing $k$-NN matching ($k=2$).
- **Lowe's Ratio Test:** Filters tentative matches using threshold $0.75$ (`ratio_test_threshold = 0.75`).
- **Homography Estimation:** `cv2.findHomography(pts_query, pts_ref, cv2.RANSAC, ransac_reproj_threshold=3.0)`.

### 2.2 Alignment Quality Gating & Failure Classification
In [`src/objectmemory/registration/evaluator.py`](src/objectmemory/registration/evaluator.py#L26-L250), an estimated homography must pass six consecutive geometric and statistical filters to be declared valid:
1. **Finite Matrix Check:** Matrix $H$ must contain only finite numbers (rejects `NaN`, `Inf`).
2. **Singularity & Condition Number:** Rejects matrices with $|\det(H)| < 10^{-8}$ or condition number $\kappa(H) > 10^7$ (`AlignmentStatus.SINGULAR_HOMOGRAPHY`).
3. **Inlier Count & Ratio:**
   - Minimum inlier count: $N_{inliers} \ge 12$ (`min_inliers = 12`).
   - Minimum inlier ratio: $R_{inliers} = N_{inliers} / N_{tentative} \ge 0.20$ (`min_inlier_ratio = 0.20`).
   - Failure: `AlignmentStatus.INSUFFICIENT_INLIERS`.
4. **Reprojection Error:**
   - Computed over all RANSAC inlier keypoint correspondences: $\epsilon_i = \|H p_{query, i} - p_{ref, i}\|_2$.
   - Maximum allowed mean error: $\bar{\epsilon} \le 4.0\text{ px}$ (`max_mean_reprojection_error = 4.0`).
   - Failure: `AlignmentStatus.EXCESSIVE_REPROJECTION_ERROR`.
5. **Convex Quadrilateral Integrity:**
   - Projects the four frame corners $[0,0], [W,0], [W,H], [0,H]$ using $H$.
   - `is_convex_quadrilateral()` evaluates cross-products of consecutive edges: cross products must all be strictly positive ($> 10^{-5}$). Rejects inverted, concave, self-intersecting, or folded quadrilaterals (`AlignmentStatus.IMPLAUSIBLE_WARP_GEOMETRY`).
6. **Area Scale Ratio & Overlap Area:**
   - Warped quadrilateral shoelace area ratio: $\text{scale} \in [0.15, 6.0]$ (`min_scale_ratio = 0.15`, `max_scale_ratio = 6.0`).
   - Overlap area fraction with reference frame: $\text{Overlap} \ge 0.15$ (`min_overlap_fraction = 0.15`).

### 2.3 Distinction Between Registration Failure and Scene Change
- The system **strictly decouples registration failure from environmental change**:
  - When registration fails (`RegistrationResult.success == False`), the registrar assigns a discrete failure status code (`INSUFFICIENT_MATCHES`, `LOW_INLIERS`, `EXCESSIVE_REPROJECTION_ERROR`, `IMPLAUSIBLE_WARP_GEOMETRY`, etc.).
  - In `TemporalMemoryEngine.process_visit` ([`src/objectmemory/temporal/engine.py#L270-L330`](src/objectmemory/temporal/engine.py#L270-L330)), `is_reg_reliable` is evaluated:
    ```python
    is_reg_reliable = (
        visit.registration.success
        and visit.registration.homography is not None
        and (visit.registration.inlier_ratio is None or visit.registration.inlier_ratio >= 0.20)
    )
    ```
  - When `is_reg_reliable == False`, spatial projection is disabled (`canon_box = None`).
  - Spatial comparisons are declared invalid (`is_spatial_reliable = False`).
  - Detections are labeled `UNREGISTERED` and cannot advance absence streaks or trigger relocation/removal transitions.

---

## A3. Raw Evidence Preservation

### 3.1 Data Model and Schema Implementation
1. **Domain Model:** [`src/objectmemory/domain/models.py`](src/objectmemory/domain/models.py#L120-L151) defines:
   ```python
   @dataclass
   class RawDetection:
       detection_id: str
       session_id: str
       timestamp: str
       label: str
       confidence: float
       raw_bbox: BoundingBox
       canonical_bbox: Optional[BoundingBox] = None
       registration_status: str = "UNREGISTERED"
       metadata: Optional[Dict[str, Any]] = None
   ```
2. **Database Table:** [`src/objectmemory/persistence/schema.py`](src/objectmemory/persistence/schema.py#L70-L85) defines `raw_detections` in SQLite Schema v3:
   ```sql
   CREATE TABLE IF NOT EXISTS raw_detections (
       detection_id TEXT PRIMARY KEY,
       session_id TEXT NOT NULL,
       timestamp TEXT NOT NULL,
       label TEXT NOT NULL,
       confidence REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
       raw_bbox_json TEXT NOT NULL,
       canonical_bbox_json TEXT,
       registration_status TEXT NOT NULL DEFAULT 'UNREGISTERED',
       metadata_json TEXT,
       FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT
   );
   CREATE INDEX IF NOT EXISTS idx_raw_detections_session ON raw_detections(session_id);
   ```
3. **Repository Methods:** [`src/objectmemory/persistence/repository.py`](src/objectmemory/persistence/repository.py#L610-L690) implements `add_raw_detection`, `add_raw_detections`, `list_raw_detections_for_session`, and `get_raw_detection`.

### 3.2 Evidence Persistence Pipeline
- In `TemporalMemoryEngine.process_visit` ([`engine.py#L310-L355`](src/objectmemory/temporal/engine.py#L310-L355)), every valid detector bounding box is written to `raw_detections` **prior to and independently of** cross-visit registration or association success.
- If registration succeeds, `registration_status = "REGISTERED"` and `canonical_bbox` contains projected canonical coordinates.
- If registration fails, `registration_status = "UNREGISTERED"` and `canonical_bbox = None`.

### 3.3 Protection Against False Identity / State Pollution
- **No False Persistent Entities:** In `engine.py#L360-L380`, creating a new persistent entity in `entities` requires:
  ```python
  if (is_reg_reliable and det.canonical_bbox is not None) or is_initial_ref:
      # Create persistent entity
  ```
  Unregistered detections **never** instantiate rows in `entities`.
- **No False State Transitions:** Because `is_spatial_reliable = False`, unregistered detections cannot trigger `CONFIRMED_MOVED`, `PENDING_REMOVAL`, or `CONFIRMED_REMOVED`.

### 3.4 API and Frontend Visualization
- **API Serializer:** [`src/objectmemory/api/workspace_manager.py#L680-L730`](src/objectmemory/api/workspace_manager.py#L680-L730) exposes `raw_detections` in `SessionDetailResponse` and `raw_detections_count` in `SessionSummary`.
- **Frontend Fallback:** In `WorkspaceOverviewView.tsx` ([lines 403–425](frontend/src/views/WorkspaceOverviewView.tsx#L403-L425)):
  - When `observations.length === 0 && raw_detections.length > 0`, the UI triggers the `isUnregisteredSession` fallback.
  - Renders raw bounding boxes on the Hero Canvas with dashed amber borders (`#F59E0B`, `strokeDasharray: "6 3"`).
  - Displays a high-contrast warning banner: *"Objects detected, but scene registration failed. Cross-visit change analysis is unavailable."*
  - Inspector strip badges raw detections with amber `UNREGISTERED` chips.
  - History table shows `{count} (unreg)`.

---

## A4. Object Association and Identity

### 4.1 Bipartite Kuhn-Munkres Matching
- **Associator:** `CrossVisitEntityAssociator` in [`src/objectmemory/temporal/associator.py`](src/objectmemory/temporal/associator.py#L35-L350).
- **Matching Algorithm:** Uses Kuhn-Munkres optimal assignment (`scipy.optimize.linear_sum_assignment` wrapped in `solve_optimal_assignment` in `src/objectmemory/baseline/comparator.py#L80-L115`).
- **Cost Function:** Minimizes total assignment cost $C_{i,j} = 1.0 - \text{Affinity}(i, j)$ over active entities $i$ and visit detections $j$.

### 4.2 Composite Affinity Function & Constraints
The affinity matrix $\text{Affinity}_{i,j} \in [0.0, 1.0]$ enforces:
1. **Class Compatibility:** Strict match required (`ent.label == det.label`), unless conservative cross-class fallback is enabled.
2. **Visual Signature Extraction:**
   - `VisualSignatureExtractor` ([`src/objectmemory/detection/signature.py#L25-L160`](src/objectmemory/detection/signature.py#L25-L160)) extracts a **148-dimensional descriptor** from real image crops:
     - 64-d RGB color histogram ($4 \times 4 \times 4$ bins).
     - 64-d HSV color histogram ($8 \times 4 \times 2$ bins).
     - 20-d spatial grid luminance structure ($4 \times 5$ sub-patches).
   - Valid signatures evaluate cosine similarity: $\text{Sim} = \frac{v_{ent} \cdot v_{det}}{\|v_{ent}\| \|v_{det}\|}$.
   - Minimum similarity threshold: $\text{Sim} \ge 0.65$ (`signature_similarity_threshold = 0.65`).
   - If $\text{Sim} \ge 0.65$, affinity receives a collocation bonus: $\text{Affinity} = \min(1.0, \text{Sim} + 0.10 \times \text{IoU})$.
   - If $\text{Sim} < 0.65$, affinity is strictly $0.0$.
3. **Collocated Spatial Fallback (No Appearance Signature):**
   - If visual signature is degenerate or unavailable, spatial matching is allowed **only if registration is verified** (`can_use_spatial == True`).
   - Requires $\text{IoU} \ge 0.40$ (`iou_match_threshold`) or centroid distance $d \le 35.0\text{ px}$ (`distance_match_threshold`).
   - Bounded affinity: $\text{Affinity} = \max(0.40, \min(0.70, \text{IoU}))$.
4. **Ambiguity Gating:**
   - If the difference between the best match and the second-best match is less than `0.12` (`ambiguity_margin = 0.12`), both the entity and detection are declared ambiguous.
   - Ambiguous pairings are **not** assigned; the entity transitions to `AMBIGUOUS_IDENTITY` rather than suffering a false identity merger.
5. **Conservative Cross-Class Fallback:**
   - If an established entity remains unmatched after strict category matching, candidate detections with different labels can match **only if**:
     - $\text{IoU} \ge 0.50$,
     - Signature similarity $\ge 0.75$,
     - Registration is verified.

---

## A5. Temporal Reasoning & Epistemic State Machine

### 5.1 State Definitions
Defined in [`src/objectmemory/domain/states.py`](src/objectmemory/domain/states.py#L1-L90):
- `INSUFFICIENT_EVIDENCE`: Entity newly introduced or unverified.
- `CONFIRMED_PRESENT`: Verified active at its established canonical position.
- `CONFIRMED_MOVED`: Verified persistent relocation exceeding dynamic movement threshold.
- `PENDING_REMOVAL`: First observation of absence with verified vacant patch or unobserved streak.
- `CONFIRMED_REMOVED`: Consecutive confirmed absences with verified vacant patch.
- `AMBIGUOUS_IDENTITY`: Multiple competing candidates preventing deterministic identity resolution.
- `OCCLUDED`: Target position occluded by another detected foreground object.

The transition policy (`is_valid_transition`) strictly forbids direct jumps from `CONFIRMED_PRESENT` to `CONFIRMED_REMOVED`, mandating the intermediate `PENDING_REMOVAL` state.

```
       [Discovery]
            │
            v
   CONFIRMED_PRESENT  <──────────────────────────┐
       │         ▲                               │
(Relocated)   (Returned to                       │
       │       Baseline)                         │
       v         │                               │
    CONFIRMED_MOVED                              │
       │                                         │
 (Absence 1)                                     │
       │                                         │
       v                                         │
    PENDING_REMOVAL  ────────────────────────────┤
       │                            (Reappearance)
 (Absence 2 + Vacant Patch)                      │
       │                                         │
       v                                         │
   CONFIRMED_REMOVED ────────────────────────────┘
```

### 5.2 Dynamic Movement Thresholding & Spatial Reasoning
In [`engine.py#L160-L200`](src/objectmemory/temporal/engine.py#L160-L200), physical movement is distinguished from camera misalignment via dynamic uncertainty expansion:
$$T_{effective} = T_{base} + \Delta u$$
where:
- $T_{base} = 35.0\text{ px}$ (`movement_displacement_threshold`).
- $\Delta u = \min(25.0, 2.0 \times \bar{\epsilon}_{reproj} + 5.0 \times \ln(1.0 + \kappa(H)))$.
- If baseline displacement $d_{base} \le 35.0\text{ px}$: remains `CONFIRMED_PRESENT`.
- If $35.0\text{ px} < d_{base} \le T_{effective}$: displacement falls within registration uncertainty; preserved as `CONFIRMED_PRESENT`.
- If $d_{base} > T_{effective}$, but IoU with baseline $\ge 0.50$ (`drift_iou_threshold`) and step displacement $\le 20.0\text{ px}$: classified as coordinate drift without physical movement; preserved as `CONFIRMED_PRESENT`.
- If $d_{base} > T_{effective}$ and not drift: transitions to `CONFIRMED_MOVED`.

### 5.3 Return to Baseline
If an entity currently in `CONFIRMED_MOVED` is detected such that:
- Step displacement from previous visit $> T_{effective}$ (it physically moved again), AND
- Baseline displacement from original $t_0$ position $\le T_{effective}$ (or $\text{IoU}_{initial} \ge 0.40$),
The engine transitions the entity back to `CONFIRMED_PRESENT` with explanation: *"Object returned to initial baseline position"*.

### 5.4 Absence Reasoning, Occlusion, and Removal Confirmation
When an entity is not detected in visit $t$:
1. **Registration Reliability Guard:**
   - If `not is_reg_reliable` or `overlap_mask is None`:
   - Returns `VisibilityStatus.UNOBSERVED`.
   - `vacant_streak` and removal progression are **frozen**. No state transition occurs.
2. **Field-of-View Overlap Guard:**
   - Canonical bounding box overlap with warped registration mask must exceed `0.70` (`min_overlap_fraction = 0.70`).
   - If $< 0.70$: entity is marked `VisibilityStatus.OUT_OF_FRAME`; absence cannot be confirmed.
3. **Occlusion by Other Detections:**
   - Evaluates IoU between entity's canonical bbox and all other detected objects in the visit.
   - If any IoU $> 0.20$: entity is marked `VisibilityStatus.PARTIALLY_OCCLUDED`; absence streak is not advanced.
4. **Vacancy Verification on Image Patch:**
   - Warped query image patch at the entity's canonical location is compared against the entity's last known visual signature.
   - If cosine similarity $< 0.35$ (`vacancy_similarity_max = 0.35`): the patch is verified vacant (background exposed).
5. **Removal Progression Timeline:**
   - **Absence 1 (Verified Vacant):** `CONFIRMED_PRESENT` $\to$ `PENDING_REMOVAL` (`vacant_streak = 1`).
   - **Absence 2 (Verified Vacant):** `PENDING_REMOVAL` $\to$ `CONFIRMED_REMOVED` (`vacant_streak = 2 >= vacancy_visits_for_removal = 2`).
   - **Patch Unverified (Missed detection, but patch looks like object):**
     - Visit 1 missed: Preserved in current state (suppresses false removal on false-negative detector dropouts).
     - Visit 2 missed: Transitions to `PENDING_REMOVAL`.
     - Visit 3+ missed: Transitions to `CONFIRMED_REMOVED`.

### 5.5 Reappearance Recovery
If an entity in `PENDING_REMOVAL` or `CONFIRMED_REMOVED` is re-detected:
- Matched via Kuhn-Munkres using visual appearance signature.
- Resets `vacant_streak = 0`.
- Transitions back to `CONFIRMED_PRESENT` (if at or near baseline) or `CONFIRMED_MOVED` (if at a relocated position).
- Emits an auditable `ChangeEvent` documenting re-discovery.

---

## A6. Persistent Long-Term Memory

### 6.1 Relational Architecture & Referential Integrity
Implemented in [`src/objectmemory/persistence/repository.py`](src/objectmemory/persistence/repository.py):
- **Engine:** SQLite 3 with strict foreign keys (`PRAGMA foreign_keys = ON;`).
- **Schema Migrations:** Managed incrementally via `PRAGMA user_version = 3`.
- **Immutability & Audit Protection:** Historical tables (`change_events`, `observations`, `raw_detections`, `pairwise_baseline_results`) use `ON DELETE RESTRICT` foreign keys. Attempting to delete a referenced entity or session raises `sqlite3.IntegrityError`.

### 6.2 Persistence Across Restarts
- **Storage Location:** Each workspace lives in an isolated filesystem directory:
  `data/workspaces/<workspace_id>/memory.db` alongside `images/<session_id>.png`.
- **Backend Restart Behavior:** `WorkspaceManager` ([`workspace_manager.py#L225-L260`](src/objectmemory/api/workspace_manager.py#L225-L260)) reopens `memory.db`, automatically runs `apply_migrations`, checks the reference keyframe, and rebuilds the `RealImagePipelineRunner` state without losing historical visits, entities, or event sequences.
- **Frontend Restart Behavior:** React UI queries `/api/workspaces/{id}/observations` and `/api/workspaces/{id}/entities`, rendering the full chronological timeline from persistent SQLite storage.

---

## A7. Live Monitoring Implementation

### 7.1 Architecture: Discrete Snapshot Ingestion
- **Source:** [`frontend/src/views/WorkspaceOverviewView.tsx`](frontend/src/views/WorkspaceOverviewView.tsx#L290-L385).
- **Webcam Stream Acquisition:** Uses browser Web API `navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: 'environment' } })`.
- **Frame Sampling:**
  - Manual Mode: User clicks *"Capture Single Observation Now"*.
  - Periodic Auto-Capture Mode: An active countdown timer (`countdownTimerRef`, interval 5s–60s) samples frames.
  - Video frame is drawn to an offscreen HTML5 `<canvas>` and encoded via `canvas.toBlob(blob, 'image/jpeg', 0.92)`.
  - The Blob is uploaded as a multipart `FormData` payload to `POST /api/workspaces/{id}/observations`.
- **Distinction Notice:** ObjectMemory's live monitoring is **intermittent discrete snapshot processing** (interval-based visit sampling), **not** continuous 30/60 FPS real-time video stream tracking.

---

## A8. Dataset Replay

### 8.1 Implementation & Supported Datasets
The pipeline runner can consume chronological image files from disk through the same code path used by live monitoring:
- **Dataset Loaders:**
  - `RealWorkbenchDataset` in [`src/objectmemory/datasets/real_dataset.py`](src/objectmemory/datasets/real_dataset.py).
  - `PhysicalWorkbenchDataset` in [`src/objectmemory/datasets/physical_workbench.py`](src/objectmemory/datasets/physical_workbench.py).
  - `ControlledTemporalDataset` in [`src/objectmemory/datasets/controlled_dataset.py`](src/objectmemory/datasets/controlled_dataset.py).
- **Replay Execution:**
  - `RealImagePipelineRunner.process_dataset()` ([`real_pipeline.py#L250-L280`](src/objectmemory/pipeline/real_pipeline.py#L250-L280)) iterates chronologically over visit frames, registering each to the reference baseline, performing detection, extracting signatures, and updating the temporal engine.
  - Benchmark evaluation harnesses: `scripts/evaluate_physical_workbench.py` and `tools/run_temporal_benchmark.py`.

---

## A9. Test Coverage & Verification Evidence

All 337 automated unit and integration tests execute and pass cleanly:
```powershell
$env:PYTHONPATH="src"; python -m unittest discover -s tests -p "test_*.py"
Ran 337 tests in 62.41s
OK
```

Key verified test suites:
- `tests/test_live_camera_persistence.py` (5 tests): Verifies raw detection persistence on registration failure, zero false entity creation, absence freezing, and API serialization.
- `tests/test_registration.py` (14 tests): Verifies SIFT/ORB, Lowe's test, RANSAC, reprojection thresholding, and convexity gates.
- `tests/test_temporal_engine.py` (28 tests): Verifies state transitions, displacement thresholding, coordinate drift suppression, return to baseline, and removal progression.
- `tests/test_temporal_boundary_reappearance.py` (12 tests): Verifies boundary clipping vs genuine movement, and reappearance from `PENDING_REMOVAL` / `CONFIRMED_REMOVED`.
- `tests/test_persistence.py` (15 tests): Verifies SQLite tables, foreign key restrictions, schema v1 $\to$ v2 $\to$ v3 migrations, and crash recovery.
