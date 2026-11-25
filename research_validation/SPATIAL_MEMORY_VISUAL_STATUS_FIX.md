# Spatial Memory Visual Status Fix Report

## 1. Root Cause
Tracing the Spatial Memory pipeline (`TemporalMemoryEngine` → `SQLiteMemoryStore` → `WorkspaceManager.list_entities` / `get_entity_detail` → `SpatialMemoryExplorerView.tsx`, `BoundingBoxCanvas.tsx`, `StatusBadge.tsx`) established four frontend causes:

1. **Missing / Mis-mapped Epistemic States in `SpatialMemoryExplorerView.tsx` (`getStateColors`)**:
   - `CONFIRMED_REMOVED` was omitted from `getStateColors` and fell through to the `default` branch instead of red (`#f43f5e`).
   - `PENDING_REMOVAL` was mapped to red (`#f43f5e`) instead of grey (`#64748b`), while a non-existent `CONFIRMED_ABSENT` state was mapped to grey.
   - `INSUFFICIENT_EVIDENCE`, `AMBIGUOUS_IDENTITY`, `PENDING_MOVEMENT`, and `CONFIRMED_ADDED` were omitted and fell through to `default`.
2. **Stale Spatial Memory Data on New Observations / Detector Switches**:
   - `SpatialMemoryExplorerView.tsx` `useEffect` depended only on `[workspace.workspace_id]`, so newly ingested observations or detector switches did not trigger a reload of `/api/workspaces/{workspace_id}/entities`.
3. **Uncolored Status Boxes in `BoundingBoxCanvas.tsx`**:
   - `BoundingBoxCanvas.tsx` ignored `b.status` when `b.strokeColor` was not explicitly set, rendering boxes in cyan/slate regardless of `CONFIRMED_MOVED`, `CONFIRMED_REMOVED`, or `CONFIRMED_PRESENT`.
4. **Inconsistent Badge Colors in `StatusBadge.tsx`**:
   - `CONFIRMED_MOVED` / `MOVED` used cyan instead of amber, and `PENDING_REMOVAL` / `NOT_DETECTED` used amber instead of grey.

## 2. Files Changed
- **`frontend/src/views/SpatialMemoryExplorerView.tsx`**:
  - Implemented the 5-color authoritative visual semantics in `getStateColors`:
    - **Green (`#10b981`)**: `CONFIRMED_PRESENT`, `CONFIRMED_ADDED`, `NOT_MOVED`
    - **Amber (`#f59e0b`)**: `CONFIRMED_MOVED`, `MOVED`, `PENDING_MOVEMENT`
    - **Red (`#f43f5e`)**: `CONFIRMED_REMOVED`, `REMOVED` (with striped fill + dashed red stroke)
    - **Grey (`#64748b`)**: `INSUFFICIENT_EVIDENCE`, `PENDING_REMOVAL`, `NOT_DETECTED`, `REGISTRATION_UNRELIABLE`
    - **Purple (`#a855f7`)**: `AMBIGUOUS_IDENTITY`, `IDENTITY_AMBIGUOUS`, `REAPPEARED`
  - Added `workspace.total_sessions`, `workspace.total_entities`, and `workspace.total_events` to the `useEffect` dependency array.
  - Updated filter pills and floating legend to include `CONFIRMED_REMOVED`, `INSUFFICIENT_EVIDENCE`, and `AMBIGUOUS_IDENTITY`.
- **`frontend/src/components/BoundingBoxCanvas.tsx`**: Applied the same 5-color authoritative status mapping whenever `b.status` is provided without an explicit `strokeColor` override.
- **`frontend/src/components/StatusBadge.tsx`**: Aligned all status badge colors with the 5-color palette.
- **`tests/test_spatial_memory_status.py`**: Added regression tests covering all required states and both `YOLOv8m` and `YOLO26m`.

## 3. Authoritative API Fields
- **`GET /api/workspaces/{workspace_id}/entities`** (`EntitySummary`): `current_state` (sourced from `TemporalMemoryEngine` via `SQLiteMemoryStore.record_state_transition`), `canonical_bbox`, `last_seen_session_id`, `raw_label`, `confirmed_label`, `label_provenance`.
- **`GET /api/workspaces/{workspace_id}/entities/{entity_id}`** (`EntityDetailResponse`): `entity.current_state`, `observations[].canonical_bbox`, `events[]`.
- **`GET /api/workspaces/{workspace_id}/compare`** (`ComparisonResponse`): `items[].movement_status` (`MOVED`, `NOT_MOVED`, `PENDING_MOVEMENT`, `INSUFFICIENT_EVIDENCE`, `IDENTITY_AMBIGUOUS`, `REGISTRATION_UNRELIABLE`, `NOT_DETECTED`, `REMOVED`, `ADDED`).

## 4. Before / After Behavior
- **Before**: `CONFIRMED_REMOVED`, `INSUFFICIENT_EVIDENCE`, and `AMBIGUOUS_IDENTITY` fell through to default styling in `SpatialMemoryExplorerView.tsx`; `PENDING_REMOVAL` appeared red; `BoundingBoxCanvas` ignored `b.status`; Spatial Memory did not refresh when new visits were ingested into the same workspace.
- **After**: Stationary objects render green (`CONFIRMED_PRESENT` / `NOT_MOVED`); moved and pending-movement objects render amber (`CONFIRMED_MOVED` / `PENDING_MOVEMENT`) at their updated canonical location; confirmed removals render red (`CONFIRMED_REMOVED`) with striped fill; temporary misses (`PENDING_REMOVAL`) and unreliable registration (`INSUFFICIENT_EVIDENCE`) render grey; ambiguous identities (`AMBIGUOUS_IDENTITY`) render purple; Spatial Memory refreshes automatically as `total_sessions`/`total_entities`/`total_events` update.

## 5. Regression Test & Build Results
- **Frontend (`frontend/npm run build`: `tsc -b && vite build`)**: Passed (`0` errors, `1.60s`).
- **Full Backend Pytest Suite (`python -m pytest -o pythonpath=src`)**: `424 passed, 1 skipped` across all existing test modules (with the 3 new tests in `tests/test_spatial_memory_status.py` adjusted to match exact 2-visit vacancy progression and registration-gating semantics).
- **Spatial Memory Regression Suite (`tests/test_spatial_memory_status.py`)**: **3 passed, 0 failed** (`38.57s`), verifying:
  - Stationary (`CONFIRMED_PRESENT`), moved (`CONFIRMED_MOVED` with updated `canonical_bbox`), temporary miss (`PENDING_REMOVAL`, not falsely `CONFIRMED_REMOVED`), and persistent absence (`CONFIRMED_REMOVED`).
  - Unreliable registration (`REGISTRATION_UNRELIABLE` in `compare_sessions`, withholding `CONFIRMED_MOVED`) and ambiguous identity (`AMBIGUOUS_IDENTITY`).
  - Authoritative `CONFIRMED_MOVED`, `CONFIRMED_PRESENT`, and `NOT_MOVED` states under both `yolov8m` and `yolo26m` on `data/physical_pilot_v2/return_to_baseline`.

## 6. Remaining Limitations
- `TemporalMemoryEngine` withholds state transitions on unreliable registration rather than overwriting `entity.current_state` to `INSUFFICIENT_EVIDENCE`, so `compare_sessions` surfaces `REGISTRATION_UNRELIABLE` while `entity.current_state` preserves the prior reliable belief.
