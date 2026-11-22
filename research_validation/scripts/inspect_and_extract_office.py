#!/usr/bin/env python3
"""
TAR Member Inspection and Selective Extraction Utility for OpenLORIS-Scene Office Archive.
"""

import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Set

TARGET_SEQUENCES = {"office1-1", "office1-2", "office1-3"}

def inspect_tar(tar_path: Path, output_json: Path) -> Dict:
    print(f"Opening TAR archive for inspection: {tar_path}")
    inventory = {
        "tar_path": str(tar_path),
        "total_members": 0,
        "total_uncompressed_bytes": 0,
        "sequences_found": {},
        "root_files": [],
        "file_types": {},
    }

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            inventory["total_members"] += 1
            inventory["total_uncompressed_bytes"] += member.size
            
            # Classify extension
            ext = Path(member.name).suffix.lower() or "(dir)" if member.isdir() else "(none)"
            inventory["file_types"][ext] = inventory["file_types"].get(ext, 0) + 1

            # Check sequence assignment
            parts = Path(member.name).parts
            if len(parts) > 0:
                top = parts[0]
                if any(seq in top for seq in ["office1-1", "office1-2", "office1-3", "office1-4", "office1-5", "office1-6", "office1-7"]):
                    if top not in inventory["sequences_found"]:
                        inventory["sequences_found"][top] = {
                            "member_count": 0,
                            "total_bytes": 0,
                            "sample_files": []
                        }
                    seq_info = inventory["sequences_found"][top]
                    seq_info["member_count"] += 1
                    seq_info["total_bytes"] += member.size
                    if len(seq_info["sample_files"]) < 5 and not member.isdir():
                        seq_info["sample_files"].append(member.name)
                else:
                    inventory["root_files"].append({
                        "name": member.name,
                        "size": member.size,
                        "isdir": member.isdir()
                    })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"Inspection complete. Recorded {inventory['total_members']} members to {output_json}")
    return inventory

def selective_extract(tar_path: Path, dest_dir: Path, target_seqs: Set[str]) -> List[str]:
    print(f"Extracting target sequences {target_seqs} to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []

    with tarfile.open(tar_path, "r") as tar:
        selected_members = []
        for member in tar.getmembers():
            # Security check
            if member.name.startswith("/") or ".." in member.name:
                print(f"[SECURITY WARNING] Skipping unsafe member: {member.name}")
                continue
            
            parts = Path(member.name).parts
            if len(parts) == 0:
                continue
            top = parts[0]
            # Match target sequence or root calibration/metadata
            if top in target_seqs or len(parts) == 1 or "calib" in member.name.lower() or "yaml" in member.name.lower():
                selected_members.append(member)
        
        print(f"Found {len(selected_members)} matching members for extraction.")
        for m in selected_members:
            tar.extract(m, path=dest_dir)
            if not m.isdir():
                extracted_files.append(m.name)

    print(f"Extracted {len(extracted_files)} files successfully.")
    return extracted_files

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_and_extract_office.py <tar_path> <dest_dir>")
        sys.exit(1)
    tar_p = Path(sys.argv[1])
    dest_p = Path(sys.argv[2])
    out_json = dest_p / "office_tar_inventory.json"
    dest_p.mkdir(parents=True, exist_ok=True)
    inspect_tar(tar_p, out_json)
    selective_extract(tar_p, dest_p, TARGET_SEQUENCES)
