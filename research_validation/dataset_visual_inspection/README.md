# ObjectMemory — Visual Dataset Inspection Gallery

Standalone, read-only visual inspection gallery built on top of `research_validation/DATASET_AUDIT_REPORT.md`.

## How to Open

1. **Direct Local File:** Open `research_validation/dataset_visual_inspection/index.html` directly in any modern browser (the inventory data is embedded in the page so `file://` CORS restrictions do not block rendering, and all local image links use relative paths).
2. **Optional Local HTTP Server:**
   ```bash
   cd research_validation/dataset_visual_inspection
   python -m http.server 8080
   ```
   Then open `http://localhost:8080/` (or serve from repository root so `../../data/...` resolves cleanly: run `python -m http.server 8080` from repository root and visit `http://localhost:8080/research_validation/dataset_visual_inspection/`).

## Files
- `index.html` — Interactive gallery with Section A–F navigation, sequence selector, Original vs. Annotated bounding-box toggle, consecutive Before/After pair comparison, and paginated OpenLORIS frame browser.
- `dataset_inventory.json` — Machine-readable inventory of all inspected local images, dimensions, manifests, poses, and external preview metadata.
- `undisplayed_files.md` — Log of any unreadable/missing local files and non-downloaded external archives.
