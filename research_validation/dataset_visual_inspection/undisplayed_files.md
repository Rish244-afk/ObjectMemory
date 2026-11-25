# Undisplayed & Unavailable Files Log

**Inspection Date:** November 25, 2026

## 1. Local Image Verification (`dataset_inventory.json`)
- **Synthetic Workbench v2 (`data/synthetic_workbench_v2/images/`):** `25 / 25` files verified readable (`0` undisplayed).
- **Desk Pilot 01 (`data/real_workbench/visits/`):** `7 / 7` files verified readable (`0` undisplayed).
- **OpenLORIS-Scene Keyframes (`research_validation/data/raw/openloris_scene/office/extracted/office1-1/color/`):** `10 / 10` evaluated keyframes verified readable (`0` undisplayed). `2,068` total extracted PNG frames indexed across `office1-1` (`809`), `office1-2` (`899`), and `office1-3` (`360`).

## 2. Empty Local Datasets (0 Images on Disk)
- `data/physical_pilot_v2/visits/` — Directory does not exist; `0 / 10` planned photographs captured (`TEMPLATE_AWAITING_PHYSICAL_DATA`).
- `data/physical_workbench_v1/` — `0 / 13` planned photographs captured (`AWAITING_PHYSICAL_DATA`).

## 3. External Datasets Not Locally Downloaded
- **ETH ASL Change Detection (`office`, `living_room`, `lounge`):** Full ROS `.bag` archives and 3D TSDF meshes are not downloaded locally. Only official external preview images from `github.com/ethz-asl/change_detection_ds` are referenced (`OFFICIAL EXTERNAL PREVIEW — NOT LOCALLY DOWNLOADED DATA`). If viewed offline, external preview URLs will not load.
- **ChangeSim (Optional Candidate):** Full Unreal Engine 4 simulation dataset is not downloaded locally. Only the official README preview animation from `github.com/SAMMiCA/ChangeSim` is referenced (`OFFICIAL EXTERNAL PREVIEW — NOT LOCALLY DOWNLOADED DATA`).
