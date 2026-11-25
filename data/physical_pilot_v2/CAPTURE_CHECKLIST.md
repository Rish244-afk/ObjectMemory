# Physical Pilot v2 — 10-Visit Capture Checklist & Annotation Guide

**Status:** `AWAITING_PHYSICAL_DATA` (Do NOT claim physical evaluation is complete until all 10 genuine photographs are captured and ground-truth bounding boxes are annotated.)

## 1. Equipment & Setup
- **Surface:** Flat, textured tabletop or desk (at least 80 cm × 60 cm; avoid plain glossy white surfaces).
- **Camera:** Smartphone or webcam at a **single fixed resolution** across all 10 visits (e.g., `1280×720` or `1920×1080`, saved as `visits/visit_01.png` … `visits/visit_10.png`). Do **not** crop different visits to different resolutions.
- **Optical Lock:** Lock focus, exposure, and white balance if possible.
- **Objects (5 COCO-supported classes):**
  - `cup_01` (`cup`): Ceramic mug with handle
  - `bottle_01` (`bottle`): Water bottle
  - `scissors_01` (`scissors`): Scissors
  - `book_01` (`book`): Book
  - `plant_01` (`potted plant` or `cell phone` — update `class_name` in template if swapped)

## 2. 10-Visit Capture Checklist

| Visit | Filename | Action | Experimental Purpose | Expected Ground-Truth States |
| :---: | :--- | :--- | :--- | :--- |
| **V01** | `visits/visit_01.png` | Place all 5 objects separated by ≥10 cm; normal lighting. | Establish canonical reference frame & initial entity tracks. | All 5 `CONFIRMED_PRESENT` |
| **V02** | `visits/visit_02.png` | Touch nothing; capture second frame. | Stationary control (`FRR`, `UMR`). | All 5 `CONFIRMED_PRESENT` |
| **V03** | `visits/visit_03.png` | Dim room light (~50%) or add oblique desk-lamp shadow; objects stationary. | Lighting invariance of registration & 148-d HSV-Sobel signatures. | All 5 `CONFIRMED_PRESENT` |
| **V04** | `visits/visit_04.png` | Restore lighting; shift/tilt camera ~5° or ~5 cm laterally; objects stationary. | Camera misalignment & uncertainty margin ($\Delta U$) validation. | All 5 `CONFIRMED_PRESENT` |
| **V05** | `visits/visit_05.png` | Return camera to nominal view; move `cup_01` by >100 px (~15–20 cm). | Object movement precision/recall & displacement gating. | `cup_01`: `CONFIRMED_MOVED`; others: `CONFIRMED_PRESENT` |
| **V06** | `visits/visit_06.png` | Return `cup_01` to its exact V01 spot. | Historical identity recovery (`HRA`, `return_to_baseline`). | All 5 `CONFIRMED_PRESENT` (`cup_01` `event_id="return_to_baseline"`) |
| **V07** | `visits/visit_07.png` | Place a cardboard/folder over `scissors_01` (fully/mostly covered). | Temporary occlusion freeze (no false removal). | `scissors_01`: `CONFIRMED_PRESENT` (`PARTIALLY_OCCLUDED` / `HEAVILY_OCCLUDED`) |
| **V08** | `visits/visit_08.png` | Remove the occluding folder; `scissors_01` unchanged. | Occlusion recovery without identity fragmentation. | All 5 `CONFIRMED_PRESENT` |
| **V09** | `visits/visit_09.png` | Physically remove `bottle_01` from the desk. | Removal onset (streak 1, empty patch vacancy). | `bottle_01`: `PENDING_REMOVAL` (`UNOBSERVED`, `bbox: null`) |
| **V10** | `visits/visit_10.png` | Keep `bottle_01` removed; capture final frame. | Multi-visit removal confirmation (streak 2) & physical latency. | `bottle_01`: `CONFIRMED_REMOVED` (`UNOBSERVED`, `bbox: null`) |

## 3. Annotation Instructions
1. Copy `manifest_v2_template.json` to `manifest_v2.json`.
2. For each visit `V01`–`V10`, annotate `[x_min, y_min, x_max, y_max]` pixel coordinates in that visit's image frame for every visible object (replace the `0.0` placeholder boxes).
3. Verify schema compliance before evaluation:
   ```bash
   python -c "from pathlib import Path; from objectmemory.benchmarks.manifest import ManifestLoader; ManifestLoader.load(Path('data/physical_pilot_v2/manifest_v2.json'), check_files=True)"
   ```
