#!/usr/bin/env python3
"""
Manifest Validation Utility for ObjectMemory Research Validation Suite.

Validates observation manifests against the official JSON Schema
(research_validation/dataset_manifest_schema.json) and verifies file integrity
and spatial coordinate consistency.

Usage:
    python validate_manifest.py --manifest <path_to_manifest.json> [--check-files] [--root-dir <path>]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "dataset_manifest_schema.json"


def load_json(filepath: Path) -> Dict[str, Any]:
    """Loads and parses a JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(data: Dict[str, Any], schema_path: Path) -> List[str]:
    """Validates data against JSON Schema."""
    errors = []
    if not schema_path.exists():
        return [f"Schema file not found at: {schema_path}"]

    if HAS_JSONSCHEMA:
        schema = load_json(schema_path)
        validator = jsonschema.Draft7Validator(schema)
        for err in validator.iter_errors(data):
            path_str = " -> ".join(str(p) for p in err.absolute_path)
            errors.append(f"Schema Error at '{path_str}': {err.message}")
    else:
        # Fallback basic structure check if jsonschema is not installed
        for field in ["schema_version", "dataset_name", "environment_id", "sequence_id", "observations"]:
            if field not in data:
                errors.append(f"Missing required top-level field: '{field}'")
    return errors


def validate_semantics_and_files(
    data: Dict[str, Any],
    root_dir: Path,
    check_files: bool = True,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validates coordinate geometry, state logic, and on-disk file existence.
    Returns: (errors, warnings, summary_stats)
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    observations = data.get("observations", [])
    total_obs = len(observations)
    unique_entities = set()
    state_distribution: Dict[str, int] = {}
    missing_files_count = 0

    valid_visibility = {"CLEARLY_VISIBLE", "PARTIALLY_OCCLUDED", "OUT_OF_FRAME", "UNOBSERVED", "UNCERTAIN"}
    valid_states = {
        "CONFIRMED_PRESENT",
        "MOVED",
        "MOVED_BACK_TO_BASELINE",
        "PENDING_REMOVAL",
        "CONFIRMED_REMOVED",
        "AMBIGUOUS_IDENTITY",
    }

    for idx, obs in enumerate(observations):
        obs_id = obs.get("observation_id", f"obs_idx_{idx}")
        img_rel = obs.get("image_path")
        
        # Check image file existence
        if check_files and img_rel:
            img_full = root_dir / img_rel
            if not img_full.exists():
                errors.append(f"[{obs_id}] Image file not found: {img_full}")
                missing_files_count += 1

        # Check depth file existence if specified
        depth_rel = obs.get("depth_path")
        if check_files and depth_rel:
            depth_full = root_dir / depth_rel
            if not depth_full.exists():
                warnings.append(f"[{obs_id}] Depth file not found: {depth_full}")

        # Check change mask existence if specified
        mask_rel = obs.get("change_mask_path")
        if check_files and mask_rel:
            mask_full = root_dir / mask_rel
            if not mask_full.exists():
                warnings.append(f"[{obs_id}] Change mask file not found: {mask_full}")

        # Check object annotations
        for obj in obs.get("objects", []):
            obj_id = obj.get("object_id")
            if obj_id:
                unique_entities.add(obj_id)
            
            state = obj.get("object_state")
            if state:
                state_distribution[state] = state_distribution.get(state, 0) + 1
                if state not in valid_states:
                    errors.append(f"[{obs_id}:{obj_id}] Invalid object_state '{state}'")

            vis = obj.get("visibility_status")
            if vis and vis not in valid_visibility:
                errors.append(f"[{obs_id}:{obj_id}] Invalid visibility_status '{vis}'")

            # Validate bounding box coordinates [x_min, y_min, x_max, y_max]
            bbox = obj.get("bounding_box")
            if bbox is not None:
                if len(bbox) != 4:
                    errors.append(f"[{obs_id}:{obj_id}] Bounding box must have 4 coordinates, got {bbox}")
                else:
                    x_min, y_min, x_max, y_max = bbox
                    if x_min < 0 or y_min < 0:
                        warnings.append(f"[{obs_id}:{obj_id}] Negative bounding box coordinates: {bbox}")
                    if x_max <= x_min:
                        errors.append(f"[{obs_id}:{obj_id}] x_max ({x_max}) must be strictly greater than x_min ({x_min})")
                    if y_max <= y_min:
                        errors.append(f"[{obs_id}:{obj_id}] y_max ({y_max}) must be strictly greater than y_min ({y_min})")

    stats = {
        "dataset_name": data.get("dataset_name"),
        "environment_id": data.get("environment_id"),
        "sequence_id": data.get("sequence_id"),
        "total_observations": total_obs,
        "unique_entities_count": len(unique_entities),
        "unique_entities": sorted(list(unique_entities)),
        "state_distribution": state_distribution,
        "missing_files_count": missing_files_count,
    }
    return errors, warnings, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ObjectMemory Dataset Manifests")
    parser.add_argument("--manifest", required=True, help="Path to the JSON manifest file")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Path to JSON Schema file")
    parser.add_argument("--root-dir", default=".", help="Project root directory for resolving relative file paths")
    parser.add_argument("--check-files", action="store_true", help="Verify that referenced image and depth files exist on disk")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    schema_path = Path(args.schema).resolve()
    root_dir = Path(args.root_dir).resolve()

    print(f"=== ObjectMemory Manifest Validator ===")
    print(f"Manifest:   {manifest_path}")
    print(f"Schema:     {schema_path}")
    print(f"Root Dir:   {root_dir}")
    print(f"File Check: {'ENABLED' if args.check_files else 'DISABLED'}")
    print("=" * 40)

    try:
        manifest_data = load_json(manifest_path)
    except Exception as exc:
        print(f"FATAL: Failed to read manifest JSON: {exc}", file=sys.stderr)
        return 1

    # 1. Schema Validation
    schema_errors = validate_schema(manifest_data, schema_path)
    if schema_errors:
        print(f"\n[FAIL] Schema validation failed with {len(schema_errors)} error(s):")
        for err in schema_errors[:10]:
            print(f"  - {err}")
        if len(schema_errors) > 10:
            print(f"  ... and {len(schema_errors) - 10} more errors.")
        return 1
    else:
        print("[PASS] JSON Schema validation passed successfully.")

    # 2. Semantic & File Integrity Validation
    sem_errors, warnings, stats = validate_semantics_and_files(
        manifest_data, root_dir=root_dir, check_files=args.check_files
    )

    if warnings:
        print(f"\n[WARN] {len(warnings)} warning(s) detected:")
        for w in warnings[:10]:
            print(f"  - {w}")

    if sem_errors:
        print(f"\n[FAIL] Semantic / File validation failed with {len(sem_errors)} error(s):")
        for err in sem_errors[:10]:
            print(f"  - {err}")
        return 1

    print("\n[PASS] Semantic validation passed successfully.")
    print("\n=== Manifest Summary Statistics ===")
    print(f"Dataset:            {stats['dataset_name']}")
    print(f"Environment:        {stats['environment_id']}")
    print(f"Sequence:           {stats['sequence_id']}")
    print(f"Total Observations: {stats['total_observations']}")
    print(f"Unique Entities:    {stats['unique_entities_count']} ({', '.join(stats['unique_entities'][:6])})")
    print(f"State Breakdown:    {json.dumps(stats['state_distribution'], indent=2)}")
    print("====================================")
    print("ALL VALIDATION CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
