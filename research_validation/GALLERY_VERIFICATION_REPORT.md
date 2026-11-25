# ObjectMemory — Gallery & Deliverables Verification Report

**Document ID:** `OM-GALLERY-VERIFY-01`  
**Date:** November 25, 2026  
**Mode:** Strictly Read-Only Verification (zero files modified, zero automatic corrections applied)  
**Inspected Deliverables:**
1. `research_validation/dataset_visual_inspection/index.html`
2. `research_validation/dataset_visual_inspection/README.md`
3. `research_validation/dataset_visual_inspection/dataset_inventory.json`
4. `research_validation/dataset_visual_inspection/undisplayed_files.md`
5. `research_validation/DATASET_VISUAL_INSPECTION_REPORT.md`
6. `research_validation/DATASET_AUDIT_REPORT.md`
7. `research_validation/FINAL_EXPERIMENTAL_COMPLETION_REPORT.md`
8. `research_validation/FINAL_RESEARCH_GAP_AND_EXPERIMENT_PLAN.md`

---

## 1. Checks Passed

1. **All Local Image Paths Exist (`0` Missing):**
   - Section A (`synthetic_workbench_v2`): `25 / 25` image paths (`data/synthetic_workbench_v2/images/S01..S05*.png`) verified on disk and relative to `index.html` (`../../data/synthetic_workbench_v2/...`).
   - Section B (`desk_pilot_01`): `7 / 7` image paths (`data/real_workbench/visits/visit_01.png`–`visit_07.png`) verified on disk (`../../data/real_workbench/...`).
   - Section C (`OpenLORIS-Scene` Keyframes): Exact `10 / 10` evaluated keyframes (`openloris_obs_00`–`09`, frame indices `0, 89, 179, 269, 359, 448, 538, 628, 718, 808`) verified against `office1-1/manifest.json` (`../data/raw/openloris_scene/office/extracted/office1-1/color/...`).
   - Section C (`OpenLORIS-Scene` Pagination Pool): All `2,068 / 2,068` extracted RGB PNG paths (`office1-1`: `809`, `office1-2`: `899`, `office1-3`: `360`) verified on disk.
2. **Exact Ground-Truth Manifest Parity:**
   - `dataset_inventory.json` `objects` arrays for Section A (`synthetic_workbench_v2`) and Section B (`desk_pilot_01`) match `data/synthetic_workbench_v2/manifest_v2.json` and `data/real_workbench/manifest_v2.json` identically (`True` / `True`). No labels were invented.
3. **Empty Physical Dataset Reporting:**
   - Section F accurately reports `physical_pilot_v2` (`0` images, `TEMPLATE_AWAITING_PHYSICAL_DATA`) and `physical_workbench_v1` (`0` images, `AWAITING_PHYSICAL_DATA`) with no placeholder images.
4. **External Preview Labelling & HTTP Reachability:**
   - ETH ASL (`Office`, `Living Room`, `Lounge`) and ChangeSim previews are clearly labelled `OFFICIAL EXTERNAL PREVIEW — NOT LOCALLY DOWNLOADED DATA`.
   - All 4 external URLs return `HTTP 200`:
     - `ETH_ASL_Office`: `200`, `image/png`, `885,123` bytes
     - `ETH_ASL_Living Room`: `200`, `image/png`, `1,412,479` bytes
     - `ETH_ASL_Lounge`: `200`, `image/png`, `1,758,537` bytes
     - `ChangeSim_Industrial Indoor Overview`: `200`, `21,583,203` bytes (see UI/MIME caveat below).

---

## 2. Checks Failed

1. **Desk Pilot 01 Provenance Label Check — FAILED:**
   - **Requirement:** *"Confirm that Desk Pilot is labelled as AI-generated according to user-provided provenance, not as genuine physical camera captures."*
   - **Actual State:** None of the 8 inspected deliverables label `desk_pilot_01` (`data/real_workbench/`) as AI-generated.

---

## 3. Exact Files or Image Paths with Problems

| File / Path | Issue |
| :--- | :--- |
| `research_validation/dataset_visual_inspection/index.html`<br>`research_validation/dataset_visual_inspection/dataset_inventory.json`<br>`research_validation/DATASET_VISUAL_INSPECTION_REPORT.md`<br>`research_validation/DATASET_AUDIT_REPORT.md` | Label `desk_pilot_01` (`data/real_workbench/`) as `EXPLORATORY PILOT — PROVENANCE UNVERIFIED` / `UNKNOWN / Unverified Photographic Provenance` rather than `AI-generated`. |
| `research_validation/FINAL_RESEARCH_GAP_AND_EXPERIMENT_PLAN.md` | Line 11 & Section 5 describe `desk_pilot_01` as *"smartphone crops"* / *"7 smartphone crops"*. |
| `research_validation/dataset_visual_inspection/README.md` | Instructs running `cd research_validation/dataset_visual_inspection; python -m http.server 8080`, which breaks `../../data/...` relative image URLs over HTTP because `http.server` cannot serve above its working directory. |
| `https://raw.githubusercontent.com/SAMMiCA/ChangeSim/main/fig/github_main_gif.gif` (referenced in `index.html` & `dataset_inventory.json`) | Served by GitHub Raw with `Content-Type: application/octet-stream` and `X-Content-Type-Options: nosniff` (`21.58 MB`), which strict browsers may refuse to render inside an `<img>` tag. |

---

## 4. Dataset Provenance Discrepancies

- **Root Cause of Provenance Conflict:**
  - `data/real_workbench/manifest_v2.json` (`line 8`) states: *"Physical smartphone photographs cropped and saved as RGBA PNG without EXIF metadata."*
  - The Visual Gallery prompt instructed: *"Do not infer AI generation merely from absent EXIF metadata. If the original provenance cannot be established, preserve UNKNOWN/UNVERIFIED."*
  - This verification prompt now establishes **user-provided provenance for Desk Pilot (`desk_pilot_01`) as AI-generated**, conflicting with the `PROVENANCE UNVERIFIED` and `"smartphone crops"` labels across the existing deliverables.

---

## 5. UI Functionality Issues (`index.html`)

- **Local `file://` Opening:** Works cleanly because `dataset_inventory.json` is embedded in `<script id="inventory-data" type="application/json">`, avoiding `file://` `fetch()` CORS blocks. Sequence switching, Before/After pair selection, Original/Annotated `<canvas>` toggle, and 12-frame OpenLORIS pagination are all wired cleanly in DOM event handlers.
- **HTTP Serving Path Issue (`README.md`):** Running `python -m http.server 8080` inside `research_validation/dataset_visual_inspection/` prevents `../../data/...` and `../data/raw/...` from resolving. HTTP serving must be started from the **repository root**.
- **ChangeSim Preview Rendering Risk:** `github_main_gif.gif` (`21.58 MB`, `application/octet-stream` + `nosniff`) may trigger the `onerror` fallback banner in browsers enforcing `nosniff` on `<img>` elements.

---

## 6. Corrections Recommended (Not Applied)

1. **Update Desk Pilot 01 Provenance Across Deliverables:** Replace `EXPLORATORY PILOT — PROVENANCE UNVERIFIED` and `"smartphone crops"` with an explicit **`AI-GENERATED (USER-PROVIDED PROVENANCE)`** label in `index.html`, `dataset_inventory.json`, `DATASET_VISUAL_INSPECTION_REPORT.md`, `DATASET_AUDIT_REPORT.md`, `FINAL_EXPERIMENTAL_COMPLETION_REPORT.md`, and `FINAL_RESEARCH_GAP_AND_EXPERIMENT_PLAN.md` (while noting `data/real_workbench/manifest_v2.json` still records legacy `"smartphone"` text until/unless manifest updates are authorized).
2. **Fix HTTP Server Instruction in `research_validation/dataset_visual_inspection/README.md`:** Remove the `cd research_validation/dataset_visual_inspection` step so `python -m http.server 8080` is run strictly from the repository root (`http://localhost:8080/research_validation/dataset_visual_inspection/`).
3. **ChangeSim Preview:** Replace the `21.58 MB` `raw.githubusercontent.com` GIF `<img>` in Section E with a direct link to `https://github.com/SAMMiCA/ChangeSim` or a smaller static PNG preview served with `image/png`.

---

## 7. Final Readiness Assessment

- **Gallery & Data Integrity:** **READY** for local `file://` visual inspection of all 25 Synthetic v2, 7 Desk Pilot, and 10+2,068 OpenLORIS images, with 100% path and bounding-box fidelity.
- **Documentation & Decision Readiness:** **CONDITIONAL — 1 PROVENANCE CORRECTION REQUIRED** before final poster/report sign-off: apply the recommended **AI-generated** provenance label to `desk_pilot_01` across the inspection gallery and reports, and fix the HTTP server working-directory note in `README.md`.
