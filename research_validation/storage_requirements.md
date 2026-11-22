# ObjectMemory Research Validation — Storage Requirements & Directory Architecture (Task 3)

**Document Version:** 1.0.0  
**Date:** November 2026  
**Auditor:** Antigravity Autonomous Research Suite  
**Project:** ObjectMemory (Long-Term Visual Memory for Persistent Environmental Changes)  
**Target File:** `research_validation/storage_requirements.md`

---

## 1. Storage Environment Audit

* **Workstation Drive:** `C:\`
* **File System:** NTFS
* **Total Capacity:** 254.03 GB
* **Used Space:** 194.62 GB
* **Available Free Space:** **59.41 GB** (as of November 7, 2026)
* **Minimum Safety Buffer Threshold:** **25.0 GB** (Must never be breached to protect OS paging, Python caching, and active service operation).
* **Usable Operational Storage Budget:** **34.41 GB maximum.**

---

## 2. Storage Comparison: Full Datasets vs. Proposed Pilot Subsets

```
+---------------------------------------------------------------------------------------------------+
|                                  STORAGE FOOTPRINT COMPARISON                                     |
+----------------------+-----------------------+-----------------------+----------------------------+
| Dataset              | Full Dataset Size     | Pilot Subset Size     | Storage Reduction Factor   |
+----------------------+-----------------------+-----------------------+----------------------------+
| **OpenLORIS-Scene**  | ~105 GB (compressed)  | **~1.5 GB** (processed| **~70x reduction**         |
|                      | > 280 GB (extracted)  |  extracted frames)    |                            |
+----------------------+-----------------------+-----------------------+----------------------------+
| **ChangeSim**        | 156.1 GB (compressed) | **~1.2 GB** (processed| **~130x reduction**        |
|                      | > 350 GB (extracted)  |  extracted frames)    |                            |
+----------------------+-----------------------+-----------------------+----------------------------+
| **OpenLORIS-Object** | 23.6 GB (compressed)  | **~250 MB** (extracted| **~95x reduction**         |
|                      | ~65 GB (extracted)    |  target classes)      |                            |
+----------------------+-----------------------+-----------------------+----------------------------+
| **Physical**         | ~50 MB (local)        | **~50 MB** (local)    | Already stored on disk     |
| **Workbench**        |                       |                       |                            |
+----------------------+-----------------------+-----------------------+----------------------------+
| **Total**            | **> 700 GB**          | **~3.0 GB**           | **Enables execution on**   |
|                      | *(Cannot fit on C:)*  | *(Easily fits on C:)* | **current 59.4 GB drive!** |
+----------------------+-----------------------+-----------------------+----------------------------+
```

---

## 3. Directory Architecture Layout

The directory layout isolates all research validation artifacts within `research_validation/` while leaving existing production code, datasets, and tests completely undisturbed:

```text
research_validation/
├── dataset_inventory.md               # Reconnaissance and source catalog
├── dataset_download_plan.md           # Step-by-step acquisition protocol
├── dataset_manifest_schema.json       # JSON-Schema Draft-07 specification
├── evaluation_split_protocol.md       # Sequence-level split and evaluation rules
├── storage_requirements.md            # This document: storage budget and rules
├── data/
│   ├── raw/                           # Immutable raw downloads and archives
│   │   ├── openloris_scene/
│   │   │   └── groundtruth/           # Unzipped 6-DoF groundtruth trajectories (10.6 MB)
│   │   ├── changesim/                 # Temporary landing zone for sequence archives
│   │   └── openloris_object/          # Temporary landing zone for object zip
│   ├── processed/                     # Chronologically sampled and normalized frames
│   │   ├── openloris_scene/
│   │   │   └── office1/               # Sampled 2Hz keyframes from office1-1, 1-2, 1-3
│   │   ├── changesim/
│   │   │   └── warehouse_0/           # Extracted Seq_0 frames (RGB, depth, change masks)
│   │   └── openloris_object/
│   │       └── target_crops/          # Extracted cup, bottle, and box factor samples
│   ├── annotations/                   # External JSON manifests conforming to schema
│   │   ├── openloris_scene_manifest.json
│   │   ├── changesim_manifest.json
│   │   └── physical_workbench_manifest.json
│   └── splits/                        # Split definition files (train/val/test sequence maps)
│       ├── openloris_splits.json
│       ├── changesim_splits.json
│       └── physical_workbench_splits.json
├── scripts/
│   ├── validate_manifest.py           # Manifest validation and file check utility
│   ├── extract_openloris_subset.py    # Streaming subset extractor (avoids full extraction)
│   └── extract_changesim_subset.py    # Sequence-level extractor for ChangeSim
└── logs/                              # Audit and extraction logs
```

---

## 4. Immutability & Safety Rules

1. **Raw Data Immutability:**  
   Once downloaded, raw images, depth maps, and ground-truth trajectory files in `data/raw/` are marked read-only. No processing script may modify or write into raw directories.
2. **Zero In-Place Overwriting:**  
   Existing production datasets in `data/real_workbench/` and `data/physical_workbench_v1/` are strictly read-only and will never be overwritten.
3. **Reference-Based Processing:**  
   Processed datasets maintain relative path references back to raw files wherever possible, avoiding unnecessary image duplication.
4. **Intermediate Archive Purging:**  
   When downloading a multi-gigabyte tar/zip archive:
   - Stream or extract *only* the specific target sequence.
   - Verify SHA-256 / MD5 checksums of extracted items.
   - Immediately delete the compressed tar/zip archive before proceeding to the next dataset.
   - This ensures peak disk usage never exceeds **~12 GB** during any single operation.
