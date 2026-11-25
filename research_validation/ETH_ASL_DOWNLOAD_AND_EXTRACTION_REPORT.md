# ETH ASL Change Detection (Lounge) — Download & Extraction Report

**Document ID:** `OM-ETH-ASL-EXTRACT-01`  
**Date:** November 25, 2026  
**Official Sources:**
- Project Page: `https://projects.asl.ethz.ch/datasets/change-detection/`
- GitHub Repo: `https://github.com/ethz-asl/change_detection_ds`
- ETH Research Collection (`lounge.zip`): `https://www.research-collection.ethz.ch/server/api/core/bitstreams/7492cccc-c182-488a-a21b-8b5102cc0348/content`

---

## 1. Pre-Download Disk & Archive Plan

- **Available Disk Space (`C:\`):** `38.34 GB` free prior to download.
- **`lounge.zip` Archive Size:** `30,950,780,627` bytes (`28.83 GB` compressed) containing `10` ROS 1 `.bag` files (`67.02 GB` uncompressed) + `10` `.ply` aligned TSDF meshes (`190.76 MB` uncompressed).
- **Disk-Safe Resumable Strategy:** Retaining `lounge.zip` (`28.83 GB`) alongside all `10` extracted `.bag` files (`67.02 GB`) requires `~96 GB`, exceeding `38.34 GB` free space. Because the ETH Research Collection server supports `HTTP 206 Partial Content` (`Accept-Ranges: bytes`), `scripts/prepare_eth_asl_lounge.py` streamed each `.ply` and `.bag` entry individually via `256 MB` Range chunks (with `HTTP 429` backoff), verified exact **Zip64 CRC32** and **SHA-256** checksums, extracted all `color_image` (`1280×720` `bgr8`) frames as lossless PNGs + `T_G_C`/`T_G_D` transforms, and purged the temporary `.bag` cache after each observation.

---

## 2. Downloaded & Extracted Inventory (`10 / 10` Observations Verified)

- **Total Recordings Processed:** `10` (`observation_01` – `observation_10`)
- **Total Extracted RGB Frames:** **`21,084`** lossless PNGs (`1280×720`, original resolution & color preserved)
- **Corrupt / Missing Frames:** **`0`**
- **Total `.ply` Meshes Preserved:** `10` (`200,024,463` bytes / `190.76 MB` under `data/eth_asl_lounge/raw/complete_mesh/`)
- **Total Extracted RGB Disk Usage:** `20,346,271,957` bytes (`18.95 GB`)

| Obs ID | Source Bag | Bag Compressed | Bag Uncompressed | Bag CRC32 | Extracted RGB PNGs | Corrupt | RGB Disk (MB) | Mesh `.ply` (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `observation_01` | `observation_0.bag` | 2,439.24 MB | 5,973.88 MB | `22b4d3df` | **1,888** | 0 | 1,708.90 | 21.11 |
| `observation_02` | `observation_1.bag` | 2,047.66 MB | 5,078.49 MB | `129d6a51` | **1,560** | 0 | 1,356.11 | 16.70 |
| `observation_03` | `observation_2.bag` | 3,424.77 MB | 8,057.55 MB | `214cdd3c` | **2,533** | 0 | 2,338.85 | 17.13 |
| `observation_04` | `observation_3.bag` | 2,978.28 MB | 7,037.39 MB | `c804cad3` | **2,171** | 0 | 1,983.74 | 17.70 |
| `observation_05` | `observation_4.bag` | 3,104.81 MB | 7,284.26 MB | `e94b8a97` | **2,185** | 0 | 2,002.00 | 16.42 |
| `observation_06` | `observation_5.bag` | 2,375.61 MB | 5,837.47 MB | `58e2a5ea` | **1,784** | 0 | 1,565.13 | 13.15 |
| `observation_07` | `observation_6.bag` | 3,283.38 MB | 7,263.85 MB | `28e307f5` | **2,186** | 0 | 2,040.50 | 19.20 |
| `observation_08` | `observation_7.bag` | 2,986.71 MB | 6,971.72 MB | `1d93e7ed` | **2,149** | 0 | 1,981.09 | 20.87 |
| `observation_09` | `observation_8.bag` | 3,814.99 MB | 8,600.33 MB | `f5c2b7f3` | **2,627** | 0 | 2,483.10 | 22.25 |
| `observation_10` | `observation_9.bag` | 2,958.89 MB | 6,528.51 MB | `44b260a2` | **2,001** | 0 | 1,944.28 | 26.23 |
| **Total** | **10 bags** | **28.72 GB** | **67.02 GB** | — | **21,084** | **0** | **18.95 GB** | **190.76 MB** |

*(Full SHA-256 hashes, topic counts, and per-frame timestamps/poses are recorded in `research_validation/ETH_ASL_DATASET_MANIFEST.json` and `data/eth_asl_lounge/extracted/observation_XX/metadata/frames_and_poses.json`.)*
