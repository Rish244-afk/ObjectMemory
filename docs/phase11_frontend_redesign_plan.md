# Phase 11 — Frontend Audit and Research-Grade Redesign Plan

**Date**: 2026-10-21  
**Project**: ObjectMemory Studio  
**Workspace**: `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Phase**: Phase 11 — Read-Only Frontend Audit and UI/UX Redesign Specification  
**Status**: READ-ONLY AUDIT & DESIGN BLUEPRINT (No application code modified)

---

## 1. Executive Summary

ObjectMemory Studio is an auditable visual memory system designed for long-term physical object tracking and environmental change detection across repeated visits to a workspace. While the underlying backend architecture—comprising planar SIFT registration, compound canonical homography, Hungarian ground-truth matching, and temporal belief state machines—has reached scientific maturity in Phases 9B through 10A (288 passing unit tests), the current frontend remains in an early MVP state.

This audit provides an exhaustive evaluation of the current frontend implementation and formulates an actionable, research-grade visual redesign. The redesigned interface replaces the current ad-hoc dashboard layout with a **scene-first, professional laboratory workstation** utilizing **dark charcoal surfaces with restrained cyan accents**, rigorous geometric and epistemic visualization, and zero fabricated metrics or misleading claims.

---

## 2. Comprehensive Frontend Audit

### 2.1 Technology Stack & Build Pipeline
- **Core Framework**: React 19.3.1 (`react`, `react-dom`)
- **Language**: TypeScript 6.0.3 with strict type-checking (`tsc -b`)
- **Styling**: Tailwind CSS v4.3.3 via `@tailwindcss/vite`
- **Iconography**: Lucide React 1.60.0 (`lucide-react`)
- **Bundler**: Vite 8.3.4 (build time ~524ms, outputs 362 kB JS, 52 kB CSS)
- **End-to-End Testing**: Playwright 1.64.1 (`frontend/e2e/studio_e2e.spec.ts`)
- **Linter**: Oxlint 1.88.0

### 2.2 Route Architecture & State Management
Currently, the application does not use a routing library (such as React Router); instead, state-driven view switching is orchestrated at the root component level:
- [frontend/src/App.tsx](frontend/src/App.tsx):
  - State `workspaces: WorkspaceSummary[]` and `currentWorkspace: WorkspaceSummary | null`.
  - Active workspace persisted in browser `localStorage` under `objectmemory_active_workspace_id`.
  - State `activeTab: ActiveTab` with 7 tab states:
    1. `'overview'`: [WorkspaceOverviewView.tsx](frontend/src/views/WorkspaceOverviewView.tsx)
    2. `'comparison'`: [SceneComparisonView.tsx](frontend/src/views/SceneComparisonView.tsx)
    3. `'spatial'`: [SpatialMemoryExplorerView.tsx](frontend/src/views/SpatialMemoryExplorerView.tsx)
    4. `'entities'`: [EntityHistoryView.tsx](frontend/src/views/EntityHistoryView.tsx)
    5. `'timeline'`: [TimelineView.tsx](frontend/src/views/TimelineView.tsx)
    6. `'diagnostics'`: [DiagnosticsView.tsx](frontend/src/views/DiagnosticsView.tsx)
    7. `'evaluation'`: [EvaluationLabView.tsx](frontend/src/views/EvaluationLabView.tsx)
  - Unselected workspace state renders [DashboardView.tsx](frontend/src/views/DashboardView.tsx).

### 2.3 Visual Flaws & Architectural Limitations in Current UI
1. **Cluttered Form-First Rather Than Scene-First Layout**:
   In [WorkspaceOverviewView.tsx](frontend/src/views/WorkspaceOverviewView.tsx), the workspace header, summary cards, and upload input form dominate the upper fold. Ingesting imagery requires looking past multiple configuration fields. Once uploaded, the scene image is buried inside a subordinate table row or inspect modal rather than functioning as the hero focal point.
2. **Timeline Disconnection (Events vs Visits)**:
   [TimelineView.tsx](frontend/src/views/TimelineView.tsx) displays an audit trail of `ChangeEventSummary` items. However, there is no **interactive visit scrubber** allowing an operator to step seamlessly across successive visits ($V_1 \to V_2 \to \dots \to V_n$) to observe how the visual environment physically mutated over time.
3. **Misleading Perspective Naming ("2.5D Isometric")**:
   [SpatialMemoryExplorerView.tsx](frontend/src/views/SpatialMemoryExplorerView.tsx) offers a toggle titled `"2.5D Isometric"` which applies a CSS 3D tilt (`rotateX(55deg) rotateZ(-45deg)`) to an SVG canvas. Calling this 2.5D or isometric risks confusing users into believing 3D point cloud or depth reconstruction exists, whereas ObjectMemory operates strictly on **2D planar projective homography** ($H \in \mathbb{R}^{3 \times 3}$).
4. **Color Hierarchy and Garish Accent Overload**:
   The existing UI relies on vibrant orange buttons (`bg-orange-600`), green pills, and yellow warnings scattered without a coherent tonal hierarchy. The redesign requires **dark charcoal surfaces with restrained cyan accents**, aligning with high-precision scientific instruments.
5. **Unused Boilerplate**:
   [frontend/src/App.css](frontend/src/App.css) contains 180 lines of dead Vite starter CSS (`.hero`, `.framework`, `.vite`, `#center`, `#next-steps`) that must be purged.
6. **Uncertainty Presentation Deficiency**:
   In [EvidenceLedger.tsx](frontend/src/components/EvidenceLedger.tsx), certainty is classified with coarse static badges (`CONFIRMED`, `PROVISIONAL`, `UNCERTAIN`). It does not visually decompose the actual constituent physical uncertainties: registration inlier ratio, keyframe distance, detector confidence score, boundary proximity clipping, and patch visual similarity.

---

## 3. Backend API Endpoints and Schema Mapping

The backend is served by FastAPI ([src/objectmemory/api/app.py](src/objectmemory/api/app.py)) with schemas defined in [src/objectmemory/api/schemas.py](src/objectmemory/api/schemas.py) and evaluation router in [src/objectmemory/api/evaluation.py](src/objectmemory/api/evaluation.py).

### Endpoint Inventory
| HTTP Method & Path | Controller / Manager Method | Request Schema | Response Schema | Frontend Client Method |
| :--- | :--- | :--- | :--- | :--- |
| `GET /api/health` | `get_health()` | None | `HealthResponse` | `api.getHealth()` |
| `GET /api/detectors` | `DetectorRegistry.list_available()` | None | `DetectorListResponse` | `api.getDetectors()` |
| `GET /api/workspaces` | `mgr.list_workspaces()` | None | `List[WorkspaceSummary]` | `api.listWorkspaces()` |
| `POST /api/workspaces` | `mgr.create_workspace()` | `WorkspaceCreate` | `WorkspaceResponse` | `api.createWorkspace()` |
| `GET /api/workspaces/{id}` | `mgr.get_workspace_summary()` | Path: `id` | `WorkspaceResponse` | `api.getWorkspace()` |
| `DELETE /api/workspaces/{id}` | `mgr.delete_workspace()` | Path: `id` | `{"status": "DELETED"}` | `api.deleteWorkspace()` |
| `POST /api/workspaces/{id}/observations` | `mgr.process_image_observation()` | `multipart/form-data` | `ObservationSubmissionResult` | `api.submitObservation()` |
| `GET /api/workspaces/{id}/observations` | `mgr.list_sessions()` | Path: `id` | `List[SessionSummary]` | `api.listObservations()` |
| `GET /api/workspaces/{id}/observations/{sid}` | `mgr.get_session_detail()` | Path: `id`, `sid` | `SessionDetailResponse` | `api.getObservationDetail()` |
| `GET /api/workspaces/{id}/entities` | `mgr.list_entities()` | Query: `state` | `List[EntitySummary]` | `api.listEntities()` |
| `GET /api/workspaces/{id}/entities/{eid}` | `mgr.get_entity_detail()` | Path: `id`, `eid` | `EntityDetailResponse` | `api.getEntityDetail()` |
| `GET /api/workspaces/{id}/events` | `mgr.list_change_events()` | Path: `id` | `List[ChangeEventSummary]` | `api.listChangeEvents()` |
| `GET /api/workspaces/{id}/compare` | `mgr.compare_sessions()` | Query: `ref`, `query` | `ComparisonResponse` | `api.compareSessions()` |
| `GET /api/workspaces/{id}/diagnostics` | `mgr.get_diagnostics()` | Path: `id` | `DiagnosticsResponse` | `api.getDiagnostics()` |
| `GET /api/workspaces/{id}/images/{filename}` | `mgr.get_image_file_path()` | Path: `id`, `filename` | Raw image stream (`image/png`, `jpeg`) | `api.getImageUrl()` |
| `GET /api/evaluation/datasets` | `list_datasets()` | None | `List[DatasetMetaResponse]` | `api.listDatasets()` |
| `POST /api/evaluation/run` | `run_evaluation()` | `EvaluationRunRequest` | `EvaluationRunResponse` | `api.runEvaluation()` |
| `GET /api/evaluation/runs` | In-memory runs | None | `List[EvaluationRunResponse]` | `api.listEvaluationRuns()` |
| `GET /api/evaluation/runs/{id}/export` | Export CSV/JSON | Query: `format` | Streamed File | `api.getEvaluationExportUrl()` |

---

## 4. Priority Feature Design Specifications

### Priority 1: Main Visual Memory Workspace
- **Objective**: Establish a scene-first visual workbench layout where high-resolution visit imagery, bounding boxes, and canonical projections are the focal hero element.
- **Existing Implementation Reusable**:
  - [BoundingBoxCanvas.tsx](frontend/src/components/BoundingBoxCanvas.tsx) (handles SVG scaling, natural size detection, and box overlays).
  - Backend ingestion: `POST /api/workspaces/{id}/observations`.
- **Required New Components**:
  - `SceneViewerHero`: 16:9 responsive canvas with smooth pan/zoom HUD, coordinate crosshair readout, and layer toggles (Raw vs Canonical Bounding Boxes, Confidence Labels, Boundary Margin Guides).
  - `WorkspaceControlDock`: Compact side dock housing visit metadata, camera calibration parameters, detector selection (YOLOv8n vs YOLO-World with text prompts), and drag-and-drop ingest tray.
- **States**:
  - *Loading*: Neutral charcoal skeleton with subtle cyan pulse.
  - *Empty*: Blueprint grid lines indicating an uncalibrated workspace awaiting Visit 1 (Baseline reference frame).
  - *Error*: Banner detailing HTTP 413 (>25MB), corrupted image format, or registration rejection.
  - *Uncertain*: Bounding box stroke rendered with dashed cyan line indicating low detector confidence ($<0.40$) or boundary-clipped entity.
- **Relevant Tests**:
  - Playwright: File drop, image preview load, box overlay render, HUD pan/zoom click.

---

### Priority 2: Interactive Timeline for Successive Visits
- **Objective**: Provide an interactive temporal scrubber allowing operators to step sequentially through visits ($V_1, V_2, \dots, V_n$), immediately observing scene changes, new entities, displacements, and registration quality.
- **Existing Implementation Reusable**:
  - `api.listObservations()` returns `SessionSummary[]` in chronological order.
  - `api.getObservationDetail()` fetches all observations and events for a selected visit.
- **Missing Backend Capability**:
  - `SessionSummary` does not currently include registration inlier ratio or keyframe status. The frontend can either retrieve this via cached `SessionDetailResponse` or a lightweight field can be added in a future backend phase. For now, the frontend derives this on demand when a visit node is selected.
- **Required New Components**:
  - `VisitTimelineScrubber`: Horizontal stepped rail at the top or bottom of the scene canvas showing visit nodes ($V_1, V_2, \dots$), timestamps, reference visit anchor icon, keyframe badges, and event count badges.
  - `PlaybackControls`: Step backward (`[KeyLeft]`), step forward (`[KeyRight]`), and auto-play walkthrough mode (stepping every 1.5 seconds).
- **States**:
  - *Loading*: Shimmer line on timeline rail.
  - *Empty*: Single disabled node indicating "Establish Baseline (Visit 1)".
  - *Error*: Red node indicator if image failed to load or registration failed.
  - *Uncertain*: Amber warning dot on visits where registration inlier ratio was degraded ($<0.20$).

---

### Priority 3: Before-and-After Scene Comparison
- **Objective**: High-precision split-screen and swipe-slider inspection between baseline reference and query visit, or between any two arbitrary visits $V_i$ and $V_j$.
- **Existing Implementation Reusable**:
  - [ComparisonSlider.tsx](frontend/src/components/ComparisonSlider.tsx) (touch and mouse drag slider).
  - [SceneComparisonView.tsx](frontend/src/views/SceneComparisonView.tsx) (side-by-side vs slider toggles).
  - Backend: `GET /api/workspaces/{id}/compare?reference_session_id=...&query_session_id=...`.
- **Required Redesign**:
  - Remove orange highlight styling; replace with **Cyan-400 for Query** and **Slate-300 for Reference**.
  - Synchronized dual-canvas zoom and pan: scrolling or zooming in the reference frame mirrors exactly in the query frame.
  - Displacement overlay: SVG vector lines drawn between $p_{\text{ref}}$ and $p_{\text{query}}$ directly on the comparison canvas.
  - Quick-switch toggle buttons: `[Ref]`, `[Query]`, `[Split Slider]`, `[Difference Mask]`.
- **States**:
  - *Loading*: Dual skeleton panels with synchronized spinners.
  - *Empty*: Prompts user: "Select two distinct visits to compute temporal differences."
  - *Error*: Clear message when homography cannot be estimated between the two visits.
  - *Uncertain*: Warning banner displayed if registration inlier ratio $<0.20$, noting that reported pixel displacements carry registration error slack.

---

### Priority 4: Object History and Persistent Identity
- **Objective**: Inspect an individual tracked entity's long-term visual trajectory across visits, distinguishing persistent physical identity from per-frame detector labels.
- **Existing Implementation Reusable**:
  - `api.listEntities()` and `api.getEntityDetail()` in [EntityHistoryView.tsx](frontend/src/views/EntityHistoryView.tsx).
  - Canonical coordinates and observation history array (`ObservationItem[]`).
- **Required New Components**:
  - `EntityCardDeck`: Master entity list with quick filters (State, Class, Displacement).
  - `EntityTrajectoryCanvas`: Planar 2D mini-map showing the entity's path on the canonical workbench ground plane across visits ($V_1 \to V_2 \dots$).
  - `EntityVisualCropGallery`: Client-side cropped thumbnails of the entity extracted from each visit's source image using `raw_bbox`, allowing visual confirmation of identity continuity without requiring thumbnail generation on the backend.
- **States**:
  - *Loading*: Entity card pulse skeletons.
  - *Empty*: "No tracked entities match the filter criteria."
  - *Error*: "Failed to retrieve entity belief history."
  - *Uncertain*: `AMBIGUOUS_IDENTITY` or `INSUFFICIENT_EVIDENCE` state badge with purple border.

---

### Priority 5: Evidence-Backed Change Explanations
- **Objective**: Provide an explainable audit ledger detailing *why* an entity transitioned between epistemic states, citing physical criteria (displacement vs threshold, IoU with baseline, unobserved streak, vacancy similarity).
- **Existing Implementation Reusable**:
  - [EvidenceLedger.tsx](frontend/src/components/EvidenceLedger.tsx).
  - `ChangeEventSummary.supporting_evidence` dictionary provided by the backend.
- **Required Redesign**:
  - Transition from decorative card boxes to a **structured scientific evidence table**:
    - **Physical Criteria Columns**: Displacement distance ($d$), adaptive movement threshold ($T_{\text{mov}}$), patch vacancy similarity ($s_{\text{patch}}$), unobserved streak count ($k_{\text{streak}}$).
    - **Decision Code**: Monospace rule tag (e.g. `RULE_REMOVAL_STREAK_EXCEEDED`, `RULE_DISPLACEMENT_EXCEEDS_ADAPTIVE_THRESHOLD`).
    - **Human-Readable Explanation**: Verbatim explanation from backend `ChangeEventSummary.explanation`.
    - **Audit Payload Modal**: One-click drawer displaying the raw JSON evidence payload for scientific reproducibility.
- **States**:
  - *Empty*: "No state transition events recorded for this entity."
  - *Uncertain*: Clear indication when vacancy verification was prevented due to background resemblance (similarity $>0.50$).

---

### Priority 6: Registration Diagnostics and Uncertainty
- **Objective**: Surface planar alignment fidelity, feature inlier ratios, keyframe graph topology, and spatial coverage without misrepresenting heuristic metrics as calibrated Bayesian probabilities.
- **Existing Implementation Reusable**:
  - [DiagnosticsView.tsx](frontend/src/views/DiagnosticsView.tsx).
  - `api.getDiagnostics()` returning `DiagnosticsResponse`.
- **Required Redesign**:
  - `RegistrationFidelityGauge`: Clean linear bar showing SIFT inliers and inlier ratio ($r \in [0, 1]$), with explicit qualitative bands:
    - $\ge 0.40$: Robust Planar Alignment
    - $0.20 - 0.39$: Degraded Alignment (Movement thresholds widened by registration uncertainty slack)
    - $< 0.20$: Safeguard Interlock Triggered (Spatial changes suppressed)
  - `KeyframeTopologyTree`: Visual display showing whether a visit registered directly to canonical ($V_1$) or via an intermediate keyframe $V_k$ using compound homography ($H_{\text{canonical}} = H_{V_k \to V_1} \cdot H_{V_i \to V_k}$).
  - **Explicit Scientific Disclaimer**: Static card stating:
    > "Planar homography assumes co-planar workbench surface. Inlier ratios measure feature correspondence consistency, not calibrated Bayesian posterior probabilities."

---

### Priority 7: Research Evaluation Dashboard
- **Objective**: Interactive benchmarking suite displaying empirical metrics against ground truth, detector comparisons (YOLOv8n vs YOLO-World), timing profiles, confusion matrices, and audit logs with strict provenance notices.
- **Existing Implementation Reusable**:
  - [EvaluationLabView.tsx](frontend/src/views/EvaluationLabView.tsx).
  - `api.listDatasets()`, `api.runEvaluation()`, `api.listEvaluationRuns()`, `api.getEvaluationExportUrl()`.
- **Required Redesign**:
  - **Dataset Separation Tabs**:
    1. *Synthetic Controlled Benchmark*: 6 sequences, 30 visits, 4-axis decoupled evaluation.
    2. *Natural Detection Subset*: COCO val2017 20-image subset with per-class TP/FP/FN/Precision/Recall tables.
    3. *Physical Desk Pilot*: 7-visit physical sequence with grid approximation bounds.
  - **Scientific Provenance Banner**: Prominent, unmissable alert documenting dataset nature (synthetic illustration vs natural photograph vs physical pilot).
  - **Export Center**: Instant export to CSV/JSON for LaTeX table compilation in research papers.

---

## 5. Visual Design System Specification

### 5.1 Color Palette
```
Surface Canvas (Deep Charcoal):   #090b10  (neutral-950/deep)
Surface Panel (Card/Sidebar):     #11141d  (neutral-900/rich)
Surface Elevated (Inputs/Modals): #181d29  (neutral-800/soft)
Surface Active/Hover:             #222736

Border Subtle:                    #232a3b
Border Strong:                    #323b52
Border Accent Focus:              #06b6d4  (Cyan-500)

Accent Primary (Cyan):            #06b6d4  (hsl(189, 94%, 43%))
Accent Glow / Tint:               rgba(6, 182, 212, 0.12)
Accent Text Highlight:            #22d3ee  (Cyan-400)

Text Primary:                     #f8fafc  (Slate-50)
Text Secondary:                   #94a3b8  (Slate-400)
Text Monospace / Code:            #64748b  (Slate-500)

Semantic Epistemic States:
- CONFIRMED_PRESENT:             #10b981  (Emerald-500)
- CONFIRMED_MOVED:               #06b6d4  (Cyan-500)
- PENDING_REMOVAL / OCCLUDED:    #f59e0b  (Amber-500)
- CONFIRMED_REMOVED:             #f43f5e  (Rose-500)
- UNCERTAIN / DEGRADED:          #a855f7  (Purple-500)
```

### 5.2 Typography and Spacing
- **Interface Font**: System sans-serif (`Inter`, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto) with tight letter-spacing (`tracking-tight`).
- **Telemetry & Numbers**: Monospace font (`JetBrains Mono`, `SFMono-Regular`, Menlo, Consolas) with tabular numbers (`tabular-nums`) to prevent jitter during real-time value updates.
- **Hierarchy**:
  - View Titles: `text-xl font-bold tracking-tight text-slate-100`
  - Section Headers: `text-xs font-mono font-semibold uppercase tracking-wider text-slate-400`
  - Body Text: `text-sm text-slate-300 font-sans`
  - Data Values / Badges: `text-xs font-mono font-medium`

### 5.3 Motion and Accessibility
- Transitions: Maximum duration `150ms` with `ease-out`.
- Reduced-Motion Support: `@media (prefers-reduced-motion: reduce)` disables scale transforms and transitions.
- Interactive Affordance: Distinct focus rings with `focus-visible:ring-2 focus-visible:ring-cyan-500/50`.

---

## 6. Audit Conclusions & Roadmap

### A. Current Frontend Capabilities
- Can create and delete physical workspaces with isolated SQLite databases.
- Can upload visit photos (PNG/JPEG) with confidence threshold and detector selection (YOLOv8n or YOLO-World with text prompts).
- Displays raw detections and projected bounding boxes on full-resolution visit images.
- Performs side-by-side and swipe slider comparisons between any two visits.
- Displays 2D canonical workbench coordinates with displacement vectors for moved objects.
- Lists persistent entities and their observation history across visits.
- Displays change event audit logs with JSON evidence payloads.
- Displays detector metadata and registration health statistics.
- Executes benchmark evaluations on the 7-visit desk pilot and exports results to CSV/JSON.

### B. Missing Functionality
1. **Interactive Visit Timeline Scrubber**: No horizontal rail allowing an operator to step forward/backward across visits ($V_1 \dots V_n$) directly within the scene viewer.
2. **Synchronized Dual-Canvas Comparison**: Scene comparison currently renders separate canvases without synchronized zoom/pan.
3. **Entity Visual Crop Gallery**: Tracked entities display tabular coordinate numbers but do not display visual cropped image thumbnails across visits.
4. **Structured Physical Evidence Ledger**: Decision explanations are rendered as unstructured card text rather than a structured table contrasting displacement against adaptive thresholds and vacancy similarity against thresholds.
5. **Multi-Dataset Benchmarking in Evaluation Lab**: Evaluation lab is currently hardcoded to only run `desk_pilot_01`; the synthetic controlled benchmark (`controlled_temporal_v1`) and natural COCO subset are not selectable in the UI.
6. **Planar-Only Ground Plane Visualization**: Current 2.5D isometric view is an angled CSS tilt of an SVG; needs clear labeling as a 2D Planar Ground Plane with no 3D depth claims.

### C. Proposed Page Structure
```
ObjectMemory Studio
├── Top Navigation Bar (Logo, Active Workspace Dropdown, Health Status, View Tabs)
└── Main Viewport (Scene-First Layout)
    ├── View 1: Workspace Studio (SceneViewerHero + VisitTimelineScrubber + IngestDock)
    ├── View 2: Scene Comparison (DualCanvasSync / SwipeSlider + DisplacementVectors)
    ├── View 3: Spatial Memory (Canonical 2D Ground Plane + TrajectoryPaths + Coordinates)
    ├── View 4: Entities & Beliefs (EntityCardDeck + CropGallery + StateEvolutionRail)
    ├── View 5: Change Audit Log (StructuredEvidenceLedger + DecisionReasonMatrix)
    ├── View 6: System Diagnostics (RegistrationFidelityGauge + KeyframeTopologyTree)
    └── View 7: Research Lab (DatasetSelector + ProvenanceAlert + MetricsMatrix + Export)
```

### D. Component Implementation Order
1. **Design System & Shell Foundation**:
   - Purge `App.css` dead styles.
   - Establish charcoal and cyan Tailwind theme variables.
   - Update `Navbar.tsx` and `StatusBadge.tsx` with refined palette and typography.
2. **SceneViewerHero & BoundingBoxCanvas Upgrade**:
   - Upgrade canvas to support smooth pan/zoom HUD, normalized aspect ratio scaling, and cyan/slate overlay boxes.
3. **VisitTimelineScrubber**:
   - Implement horizontal stepped visit rail with step forward/backward keyboard navigation.
4. **Synchronized SceneComparisonView**:
   - Refactor dual-canvas and swipe slider with synchronized viewports and cyan vector displacement overlays.
5. **EntityHistoryView & Visual Crop Gallery**:
   - Implement entity card deck and client-side cropped thumbnail generation from visit images.
6. **Structured EvidenceLedger**:
   - Replace card list with structured physical criteria table (Displacement vs Threshold, Vacancy Similarity, Streak).
7. **Diagnostics & Registration Uncertainty Panel**:
   - Implement alignment fidelity gauge with explicit qualitative bands and keyframe topology tree.
8. **Research Evaluation Lab Overhaul**:
   - Add multi-dataset selector tabs, strict provenance badges, and per-class confusion matrices.

### E. API Dependencies
- **All 7 priorities are supported by existing backend endpoints**:
  - `GET /api/workspaces` & `POST /api/workspaces`
  - `POST /api/workspaces/{id}/observations`
  - `GET /api/workspaces/{id}/observations` & `.../observations/{session_id}`
  - `GET /api/workspaces/{id}/entities` & `.../entities/{entity_id}`
  - `GET /api/workspaces/{id}/events`
  - `GET /api/workspaces/{id}/compare`
  - `GET /api/workspaces/{id}/diagnostics`
  - `GET /api/workspaces/{id}/images/{filename}`
  - `GET /api/evaluation/datasets` & `POST /api/evaluation/run`
- **Optional Backend Enhancement (Future Phase)**:
  - Add `controlled_temporal_v1` and `coco_val2017_subset` to `/api/evaluation/datasets` response so they can be run from the UI without manual CLI invocation.

### F. Acceptance Criteria for the Finished Redesign
1. **Scene-First Priority**: The scene canvas occupies $\ge 65\%$ of viewport area in standard desktop view.
2. **Palette Conformity**: All orange elements replaced with dark charcoal surfaces (`#090b10`, `#11141d`) and restrained cyan accents (`#06b6d4`, `#22d3ee`).
3. **Visit Scrubbing**: An operator can step through all visits of a workspace with arrow keys or timeline clicks, updating the scene image and detections instantly.
4. **No Fabricated Uncertainty or 3D Claims**:
   - Canonical coordinate viewer clearly identifies space as "2D Planar Workbench Plane ($x, y$ in pixels)".
   - Inlier ratio and registration metrics explicitly labeled as geometric alignment residuals, not calibrated probabilities.
5. **Traceable Decision Ledger**: Every change event displays exact numerical evidence (pixel displacement vs threshold, patch similarity vs vacancy threshold).
6. **Zero Build or Test Regressions**:
   - `npm run build` succeeds cleanly (`tsc -b && vite build`).
   - Playwright E2E suite passes (`npx playwright test`).
   - All 288 Python unit tests continue to pass (`python -m unittest discover tests`).
