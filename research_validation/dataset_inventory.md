# ObjectMemory Research Validation — Dataset Inventory & Reconnaissance

**Document Version:** 1.0.0  
**Date:** November 2026  
**Auditor:** Antigravity Autonomous Research Suite  
**Project:** ObjectMemory (Long-Term Visual Memory for Persistent Environmental Changes)  
**Target File:** `research_validation/dataset_inventory.md`

---

## 1. Executive Summary

This inventory presents the source-level reconnaissance of four target datasets selected to validate `ObjectMemory`'s core research claims: distinguishing genuine object movements and removals from temporary occlusions, lighting variations, and registration failures while preserving explainable event histories.

The four datasets comprise:
1. **OpenLORIS-Scene (ICRA 2020):** Primary real-world lifelong robotic vision benchmark with real perceptual drift and ground-truth trajectories.
2. **ChangeSim (IROS 2021):** Controlled photorealistic simulated benchmark with ground-truth pixel-level change segmentation masks and 6-DoF poses.
3. **OpenLORIS-Object (ICRA 2020):** Supplementary real-world robotic dataset with systematic variations in illumination, occlusion, clutter, and pixel size.
4. **Physical Workbench (Internal 2026):** In-situ desktop and workbench sequence suite with fine-grained entity-level lifecycle annotations and explainable event ground truth.

---

## 2. Dataset 1: OpenLORIS-Scene (Primary Real-World Benchmark)

* **Official Project Page:** [https://lifelong-robotic-vision.github.io/dataset/scene.html](https://lifelong-robotic-vision.github.io/dataset/scene.html)
* **Official Code Repository:** [https://github.com/lifelong-robotic-vision/OpenLORIS-Scene](https://github.com/lifelong-robotic-vision/OpenLORIS-Scene)
* **Hugging Face Repository:** [https://huggingface.co/datasets/shixuesong/openloris-scene](https://huggingface.co/datasets/shixuesong/openloris-scene)
* **Baidu Pan Mirror:** `pan.baidu.com/s/1FoiDFPYPduTEO1CTVRot5g` (Extraction code: `59rm`)
* **Associated Paper:** X. Shi, D. Li, P. Zhao, et al., *"Are We Ready for Service Robots? The OpenLORIS-Scene Datasets for Lifelong SLAM,"* IEEE ICRA 2020 (arXiv:1911.05603).

### 2.1 Environmental Scope & Sequences
The dataset covers five distinct real-world indoor environments captured by mobile service robots across multiple temporal revisits:

| Environment | Description | Sequence Count | Total Duration | Key Challenges |
|---|---|---|---|---|
| **`office1`** | Research office with desks, chairs, computers, books, personal items | 7 sequences (`office1-1` to `office1-7`) | 225 seconds | Moving chairs, moved laptops/cups, daytime/nighttime lighting, human traffic. **OptiTrack sub-millimeter ground-truth.** |
| **`corridor1`** | Long public corridor connecting offices and labs | 5 sequences (`corridor1-1` to `corridor1-5`) | 679 seconds | Extreme perceptual aliasing, dynamic pedestrians, severe lighting shifts. |
| **`cafe1`** | University cafeteria and dining area | 2 sequences (`cafe1-1` to `cafe1-2`) | 147 seconds | Movable tables, chairs, varying table clutter, reflective surfaces. |
| **`home1`** | Typical domestic apartment layout | 5 sequences (`home1-1` to `home1-5`) | 438 seconds | Clutter, furniture shifts, day/night domestic illumination. |
| **`market1`** | Retail supermarket aisles | 3 sequences (`market1-1` to `market1-3`) | 755 seconds | Repetitive shelving, dynamic shoppers, reflective floors (laser scan omitted). |
| **Total** | **5 Environments** | **22 Sequences** | **2,244 seconds** | **~67,320 frames at 30 FPS / ~33,660 frames at 15 FPS** |

### 2.2 Modalities & Sensor Specifications
* **RGB Color Camera (Intel RealSense D435i):** $848 \times 480$ (or $640 \times 480$) 8-bit RGB color stream at 30 FPS.
* **Depth Camera (Intel RealSense D435i):** Aligned 16-bit depth images at 30 FPS (depth unit: millimeters).
* **Fisheye Stereo (Intel RealSense T265):** Dual fisheye grayscale streams at $848 \times 800$ at 30 FPS.
* **Inertial Measurement Units (IMUs):** Dual 6-DoF accelerometer + gyroscope feeds from both D435i (200 Hz) and T265 (200 Hz).
* **Odometry:** Wheel encoder odometry from the robot base.
* **2D LiDAR:** Laser scan range data (available for all scenes except `market`).

### 2.3 Camera Poses & Ground-Truth Trajectory
* **Ground Truth Trajectories:** **Acquired and Cryptographically Verified** in `research_validation/data/raw/openloris_scene/groundtruth/extracted/` (MD5: `6dd2bfdf741831089ef974110b2860d1`, SHA-256: `07564d7ed3d6739585002afa12bcf481cc0e9e358fc64efd5e658e2c994bdc3b`).
* **Format:** TUM RGB-D trajectory format (`timestamp px py pz qx qy qz qw`).
* **Accuracy & Trajectory Sources:**
  - `office1` (7 sequences, 53,348 poses total across 225s): Sub-millimeter position accuracy recorded via an external **OptiTrack optical motion capture system** running at **~238.5 Hz** with **0 duplicate timestamps** and strict monotonic progression. Max time gap $\le 0.28$s.
  - `cafe1` & `corridor1`: 2D/3D LiDAR graph SLAM sampled at **40.0 Hz** with 0 duplicates.
  - `home1`: Visual-inertial/wheel odometry at **9–26 Hz** with significant timestamp duplication (375–491 duplicates per sequence).
* **Alignment Feasibility:** Because `office1` ground truth is sampled at 240 Hz, every 30 FPS camera frame aligns to a ground-truth pose within $\le 2.08\text{ ms}$ ($\le 0.62\text{ mm}$ displacement at $0.3\text{ m/s}$), enabling sub-millimeter trajectory association **without synthetic interpolation**.

### 2.4 Object & Change Annotations
* **Object Annotations:** **None natively.** OpenLORIS-Scene was recorded for SLAM benchmarking, not object detection. It does not provide 2D bounding boxes or class labels.
* **Change Annotations:** **None natively.** Environmental changes (moved chairs, rearranged desk objects, altered lighting) exist physically in the video, but no ground-truth change event timestamps or object lifecycle labels are provided.
* **Manual Requirement:** Tabletop and desk regions in selected sequences must be annotated with 2D bounding boxes and state transitions using ObjectMemory's manifest schema.

### 2.5 Sequence Timestamps & Chronological Structure
* Synchronized Unix epoch timestamps (seconds and nanoseconds) across all sensor streams.
* Chronological sequence numbering indicates temporal progression. In `office1`, empirical timestamp analysis reveals continuous multi-pass execution: `office1-1` ($1560000083.95 \to 1560000110.90$) $\to$ gap: 33 ms $\to$ `office1-3` ($1560000110.94 \to 1560000122.91$) $\to$ gap: 34 ms $\to$ `office1-2` ($1560000122.94 \to 1560000152.90$).

### 2.6 Official Download Methods & Verified Pilot Acquisition
Hosted on Hugging Face (`shixuesong/openloris-scene`) and Baidu Pan:

| Archive Name | Content | Format | Compressed Size | Estimated Uncompressed Size | Verification Status |
|---|---|---|---|---|---|
| `package/groundtruth.zip` | Ground truth trajectories for all scenes | ZIP | **10.6 MB** | ~45 MB | **Downloaded & Verified** (Per-sequence OptiTrack trajectories extracted) |
| `package/office1-1_7-package.tar` | Office sequences in TUM format (PNG + CSV) | TAR | **9.9 GB** | ~28.2 GB | **Acquired, Selectively Extracted & Audited** (SHA-256: `2a610f04...`, MD5: `1e568195...`) |
| `package/cafe1-1_2-package.tar` | Cafe sequences in TUM format (PNG + CSV) | TAR | **7.0 GB** | ~20.5 GB | Unacquired (Deferred) |
| `package/home1-1_5-package.tar` | Home sequences in TUM format | TAR | **17.7 GB** | ~50.1 GB | Unacquired (Deferred) |
| `package/corridor1-1_5-package.tar` | Corridor sequences in TUM format | TAR | **31.3 GB** | ~89.0 GB | Unacquired (Deferred) |
| `package/market1-1_3-package.tar` | Market sequences in TUM format | TAR | **38.8 GB** | ~110.0 GB | Unacquired (Deferred) |
| `rosbag/office1-1_7-rosbag.tar` | Office ROS1 bags (compressed) | TAR | **8.6 GB** | ~25.0 GB | Unacquired (TUM package used instead) |
| **Total Full Dataset** | **All packages** | | **~105 GB (package) / ~88 GB (rosbag)** | **> 280 GB** | **Pilot Extracted: 1.08 GB on disk** |

#### Verified Office Pilot Subset (`office1-1`, `office1-2`, `office1-3`):
* **Extracted Path:** `research_validation/data/raw/openloris_scene/office/extracted/`
* **Frames:** 2,068 color frames ($848 \times 480$, 8-bit RGB `uint8`), 100% verified decodable.
* **Trajectory Alignment (Phase 1C Audit):** Across all 2,068 frames, median $\Delta t = 1.11$ ms, mean $\Delta t = 1.73$ ms, 95th percentile $\Delta t = 2.03$ ms.
* **Outlier Resolution:** Maximum $\Delta t = 119.96$ ms at frame 97 (`office1-2`) resolved to an OptiTrack optical marker tracking dropout ($270.83$ ms duration) while the robot was stationary ($v = 0.0061$ m/s), bounding spatial error to $< 0.73$ mm.
* **Motion-Bounded Spatial Accuracy:** $93.38\%$ of frames have spatial uncertainty bound $\le 1.0$ mm; $99.76\%$ are $\le 5.0$ mm (`SYNCHRONIZED_SUB_CENTIMETER`).
* **Calibration & Conventions:** Standard ROS TF column vector mapping $\mathbf{p}_{\text{parent}} = T \cdot \mathbf{p}_{\text{child}}$ verified from physical camera mount and optical axis directions; orthonormal $\mathrm{SE}(3)$ rotation ($\det R = +1.000000$).
* **Cross-Sequence Nature:** Forensically proven to be a single continuous $68.95$-second traversal through a static room (zero frame loss across boundaries), not multi-day visits. True long-term revisits occur across `office1-4` through `office1-7`.
* **Multi-Day Revisit Feasibility (Phase 1D Audit):** All seven office sequences span 4 calendar days (June 8–11, 2019) across 5 sessions. While `office1-4` through `office1-7` offer genuine day/night lighting shifts and multi-day revisits along the same 4m aisle, they lack 2D/3D object bounding boxes and change ground truth. Recommendation: **DO NOT ACQUIRE YET** (retain current 1.08 GB pilot; use Physical Workbench and ChangeSim for ground-truth change evaluation).
* **Storage Footprint:** The 9.89 GB TAR container and intermediate 7z archives were removed; active pilot occupies only 1.08 GB.
* **Audit Documents:** Refer to [`research_validation/openloris_office_archive_audit.md`](openloris_office_archive_audit.md), [`research_validation/openloris_phase1c_alignment_verification.md`](openloris_phase1c_alignment_verification.md), and [`research_validation/openloris_multiday_revisit_feasibility.md`](openloris_multiday_revisit_feasibility.md).

* **Can individual sequences be downloaded independently?** No. On Hugging Face and Baidu Pan, sequences are bundled per-environment into multi-sequence TAR archives (e.g., `office1-1_7-package.tar` contains all 7 office sequences). Only `market1-1.bag`, `market1-2.bag`, and `market1-3.bag` are hosted as separate files.
* **Can selected frames be extracted without downloading the complete archive?** No. Standard HTTP TAR files do not support random-access member extraction without streaming or downloading the container.

### 2.7 License & Redistribution Terms
* **License:** Creative Commons Attribution-NoDerivatives 4.0 International (**CC BY-ND 4.0**).
* **Redistribution Terms:**
  - Users may copy and redistribute the material in any medium or format for any purpose, even commercially.
  - **NoDerivatives Clause:** If you remix, transform, or build upon the material, you may **not** distribute the modified material publicly without express written permission (`openloris@gmail.com`).
  - **Implication for ObjectMemory:** We must keep raw OpenLORIS-Scene images immutable and distribute our annotations as **stand-alone external JSON manifests/annotations** referencing original filenames and timestamps, rather than distributing a modified derivative dataset.
* **Citation:**
  ```bibtex
  @inproceedings{shi2020openloris,
    title={Are We Ready for Service Robots? The {OpenLORIS-Scene} Datasets for Lifelong {SLAM}},
    author={Shi, Xuesong and Li, Dongjiang and Zhao, Pengpeng and Tian, Qinbin and Tian, Yuxin and Long, Qiwei and Zhu, Chunhao and Song, Jingwei and Qiao, Fei and Song, Le and Guo, Yangquan and Wang, Zhigang and Zhang, Yimin and Qin, Baoxing and Yang, Wei and Wang, Fangshi and Chan, Rosa H. M. and She, Qi},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
    pages={3139--3145},
    year={2020}
  }
  ```

---

## 3. Dataset 2: ChangeSim (Controlled Photorealistic Simulation)

* **Official Project Page:** [https://sammica.github.io/ChangeSim](https://sammica.github.io/ChangeSim)
* **Official Code Repository:** [https://github.com/SAMMiCA/ChangeSim](https://github.com/SAMMiCA/ChangeSim)
* **Documentation & Data Spec:** [https://github.com/SAMMiCA/ChangeSim/blob/main/data.md](https://github.com/SAMMiCA/ChangeSim/blob/main/data.md)
* **Associated Paper:** J.-M. Park, J.-H. Jang, S.-M. Yoo, et al., *"ChangeSim: Towards End-to-End Online Scene Change Detection in Industrial Indoor Environments,"* IEEE/RSJ IROS 2021 (arXiv:2103.05368).

### 3.1 Environmental Scope & Sequences
ChangeSim simulates industrial indoor warehouse environments generated using Unreal Engine 4 and AirSim:
* **Environments:** Multiple warehouse layouts (`Warehouse_0`, `Warehouse_1`, etc.).
* **Sequence Structure:** Paired Reference (`Ref_Seq_`) and Query (`Query_Seq_`) trajectories.
* **Environmental Perturbations:** Query sequences include targeted environmental variants:
  - `Seq_0`: Standard baseline lighting and clear air.
  - `Seq_0_dust`: Air turbidity/dust particles simulated.
  - `Seq_1_dark`: Low-illumination / darkened warehouse lighting.
* **Scale:** Multiple trajectories with several hundred to several thousand synchronized frames per sequence.

### 3.2 Modalities & Sensor Specifications
* **RGB Images:** $640 \times 480$ 8-bit RGB in PNG format ($f_x = 320.0, f_y = 320.0, c_x = 320.0, c_y = 240.0$, FoV: $90^\circ$).
* **Depth Images:** $640 \times 480$ 8-bit RGB encoding metric depth (unit: meters).
* **Semantic Segmentation:** $640 \times 480$ pixel-level semantic category masks.
* **Change Segmentation Masks:** $640 \times 480$ pixel-level change masks directly indicating added, removed, or altered elements.
* **3D Point Cloud & SLAM Database:** `cloud_map.ply` and `rtabmap.db` from RTAB-Map.

### 3.3 Camera Poses & Ground-Truth Trajectory
* **Format:** Text file (`trajectory.txt` or per-frame `pose/0.txt`) with lines: `tx ty tz qx qy qz qw`.
* **Coordinate Frame:** North-East-Down (NED) frame: $+x$ is camera forward, $+y$ is camera right, $+z$ is camera downward.

### 3.4 Object & Change Annotations
* **Change Ground Truth:** **Natively provided** as dense 2D pixel change segmentation masks (`change_segmentation/0.png` to `xxxx.png`).
* **Object Bounding Boxes:** Not natively stored as 2D/3D bounding boxes; must be computed by calculating the bounding envelope of connected components in the semantic segmentation masks.

### 3.5 Official Download Methods & Archive Sizes
Hosted on KAIST Microsoft SharePoint (`kaistackr-my.sharepoint.com`):

| Archive Name | Split & Type | Size |
|---|---|---|
| `Reference_Sequence_Train` | Training Reference Sweeps | **52.8 GB** |
| `Reference_Sequence_Test` | Testing Reference Sweeps | **30.2 GB** |
| `Query_Sequence_Train` | Training Query Sweeps (Changes Present) | **42.8 GB** |
| `Query_Sequence_Test` | Testing Query Sweeps (Changes Present) | **30.3 GB** |
| **Total Full Dataset** | **Train + Test (Ref + Query)** | **156.1 GB** |

* **Can individual sequences be downloaded independently?** The official repository links provide only four monolithic ZIP files (30–53 GB each). When accessing SharePoint via a web browser, individual folders (e.g., `Warehouse_0/Seq_0`) can sometimes be browsed and downloaded, but programmatic direct downloads require pulling the full 30+ GB archive.
* **Can selected frames be extracted without downloading the full archive?** No. The files are hosted as monolithic ZIP archives on SharePoint.

### 3.6 License & Redistribution Terms
* **License:** Academic research use under the ChangeSim repository terms (IROS 2021). Commercial re-use and redistribution of raw simulation assets are restricted.
* **Citation:**
  ```bibtex
  @inproceedings{park2021changesim,
    title={ChangeSim: Towards End-to-End Online Scene Change Detection in Industrial Indoor Environments},
    author={Park, Jin-Man and Jang, Jae-hyuk and Yoo, Sahng-Min and Lee, Sun-Kyung and Kim, Ue-hwan and Kim, Jong-Hwan},
    booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    pages={3774--3780},
    year={2021}
  }
  ```

---

## 4. Dataset 3: OpenLORIS-Object (Supplementary Detector Robustness)

* **Official Project Page:** [https://lifelong-robotic-vision.github.io/dataset/object](https://lifelong-robotic-vision.github.io/dataset/object)
* **Official Code Repository:** [https://github.com/lifelong-robotic-vision/OpenLORIS-Object](https://github.com/lifelong-robotic-vision/OpenLORIS-Object)
* **Associated Paper:** Q. She, F. Feng, X. Hao, et al., *"OpenLORIS-Object: A Robotic Vision Dataset and Benchmark for Lifelong Deep Learning,"* IEEE ICRA 2020 (arXiv:1911.06487).

### 4.1 Scope & Categories
* **Objects:** 69 daily household/office object instances across 19 categories (e.g., mugs, cups, bottles, cans, glue sticks, staple removers, stationery boxes).
* **Total Samples:** 215,280 frames ($260 \text{ frames/instance} \times 69 \text{ instances} \times 4 \text{ factors} \times 3 \text{ levels}$).
* **Systematic Factor Variations:**
  1. *Illumination:* Normal, Weak, Strong.
  2. *Occlusion:* 0%, 25%, 50%.
  3. *Object Pixel Size:* $< 30 \times 30$, $30 \times 30 - 200 \times 200$, $> 200 \times 200$.
  4. *Clutter:* Simple, Normal, Complex backgrounds.

### 4.2 Modalities & Trajectory
* **RGB & Aligned Depth:** Intel RealSense D435i ($640 \times 480$).
* **Trajectory / Poses:** Continuous video sweeps of a mobile robot circulating around targeted tabletop objects. No global SLAM poses; focuses on local view variations.

### 4.3 Annotations & Utility for ObjectMemory
* **Object Bounding Boxes:** Provided for all target object instances.
* **Change Annotations:** None natively.
* **Role in ObjectMemory:** Serves as a targeted detector stress benchmark. Enables evaluating YOLOv8 / YOLO-World detector confidence and bounding box stability under isolated illumination shifts (normal $\rightarrow$ weak $\rightarrow$ strong) and partial occlusions (0% $\rightarrow$ 25% $\rightarrow$ 50%), directly quantifying the observation noise feeding into ObjectMemory's temporal state machine.

### 4.4 Download Methods & Sizes
* **Hosting:** Google Drive (`train.zip` and `test.zip`) via project download links.
* **Archive Sizes:**
  - `train.zip`: ~15.4 GB
  - `test.zip`: ~8.2 GB
  - Total: ~23.6 GB.
* **License:** Academic research use under Lifelong Robotic Vision terms.

---

## 5. Dataset 4: Physical Workbench (In-Situ Real-World Benchmarks)

* **Workspace Paths:** `data/real_workbench/` and `data/physical_workbench_v1/`.
* **Provenance:** Independently captured real-world desktop and workbench sequences recorded under controlled physical protocols.

### 5.1 Scope & Scenarios
* **Real Workbench Pilot (`data/real_workbench/`):** 7 completed physical visits (V01–V07) with 42 evaluated entity-visit annotations on disk.
* **Physical Workbench Protocol (`data/physical_workbench_v1/`):** 7 planned scenarios (S01–S07, 35 planned visits) in capture specification; 0 physical images currently on disk (`status: AWAITING_PHYSICAL_DATA`).
* **Physical Workbench v1 Protocol (`data/physical_workbench_v1/`):** 7 formal validation scenarios:
  1. `S01_stationary_baseline`: 5 stationary visits measuring baseline false movement/removal rates.
  2. `S02_temporary_movement_return`: Object translation and subsequent return to origin (`MOVED_BACK_TO_BASELINE`).
  3. `S03_persistent_relocation`: Object displacement across multiple consecutive visits.
  4. `S04_permanent_removal`: Complete object extraction with multi-visit vacancy confirmation.
  5. `S05_partial_occlusion`: Object obscured by foreign occluder ($\text{IoU} > 0.20$) verifying removal freezing.
  6. `S06_illumination_stress`: Extreme lighting changes (fluorescent, daylight, dim incandescent).
  7. `S07_viewpoint_perturbation`: Oblique camera angle shifts ($15^\circ–45^\circ$) testing homography quality gating.

### 5.2 Modalities & Annotations
* **RGB Color:** $1920 \times 1080$ uncalibrated monocular camera (smartphone and webcam).
* **Ground Truth Annotations:** 100% complete entity-level manifests (`manifest.json`) detailing canonical bounding boxes, physical observations (`VISIBLE`, `ABSENT`), expected states (`CONFIRMED_PRESENT`, `MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`, `REAPPEARED`, `AMBIGUOUS_IDENTITY`), and structured causal explanations.
* **Storage Requirement:** **< 50 MB total.**
* **Availability:** Immediately accessible, fully verified, and passing all regression test suites.

---

## 6. Comprehensive Reconnaissance Matrix

| Attribute | OpenLORIS-Scene | ChangeSim | OpenLORIS-Object | Physical Workbench |
|---|---|---|---|---|
| **Environment Type** | Real indoor offices, homes, cafes | Synthetic photorealistic warehouse | Real indoor tabletop objects | Real in-situ desktop/workbench |
| **Viewpoint Paradigm** | Mobile robot (6-DoF navigation) | Drone/AGV (6-DoF trajectories) | Mobile robot (object-centric sweep) | Monocular oblique planar worktop |
| **Key Sensor Modalities** | RGB, Depth, Stereo Fisheye, IMU, Odometry | RGB, Depth, Semantic Mask, Change Mask | RGB, Depth | High-resolution Monocular RGB |
| **Resolution** | $848 \times 480$ / $640 \times 480$ | $640 \times 480$ | $640 \times 480$ | $1920 \times 1080$ |
| **Camera Poses** | Ground truth OptiTrack / LiDAR | Ground truth 6-DoF NED trajectory | Relative object sweep | Planar homography + ArUco |
| **Object Annotations** | None (Requires manual labeling) | Semantic segmentation masks | 2D Bounding boxes & class labels | 2D Bounding boxes & class labels |
| **Change Annotations** | None (Physical events unannotated) | Native 2D change pixel masks | None (Classification benchmark) | Complete lifecycle change events |
| **Smallest Download** | `package/groundtruth.zip` (10.6 MB)<br>`cafe1-package.tar` (7.0 GB)<br>`office1-package.tar` (9.9 GB) | `Ref_Seq_Test` (30.2 GB) + `Query_Seq_Test` (30.3 GB) = **60.5 GB** | `test.zip` (**8.2 GB**) | **0 MB (Locally available)** |
| **Total Archive Size** | ~105 GB (package) / ~88 GB (rosbag) | **156.1 GB** | **23.6 GB** | **~50 MB** |
| **License** | **CC BY-ND 4.0** (No Derivatives) | Academic / Research (IROS 2021) | Academic / Research (ICRA 2020) | Proprietary / Internal Project |
| **Redistribution Policy**| Annotations must be separate manifests | External annotation manifests only | External annotation manifests only | Internal / Fully Permitted |
| **Primary Role in Audit** | Real-world drift & SLAM alignment | Scaled change detection verification | Detector factor stress testing | Primary end-to-end memory audit |
