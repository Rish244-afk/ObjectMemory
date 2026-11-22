"""
Command-line utility and validation tool for real-world multi-visit datasets (Phase 7).

Provides:
1. `validate`: Validates dataset integrity, monotonicity, schema compliance,
   and semantic consistency according to the Phase 7 protocol.
2. `summary`: Displays statistical breakdown of object identities, visibility
   states, physical movements, removals, and viewpoint/lighting conditions.
3. `template`: Initializes a clean multi-visit template dataset directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from objectmemory.datasets.real_dataset import (
    RealWorkbenchDataset,
    RealVisitAnnotation,
    PhysicalObservation,
    AnnotatedEvent,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dataset_tool")


def cmd_validate(manifest_path: Path) -> int:
    """Validates the specified dataset manifest against Phase 7 protocol rules."""
    logger.info("Validating dataset manifest: %s", manifest_path)
    if not manifest_path.exists():
        logger.error("Manifest file not found: %s", manifest_path)
        return 1

    dataset_dir = manifest_path.parent
    try:
        dataset = RealWorkbenchDataset.load_from_directory(dataset_dir)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        return 1

    errors = dataset.validate_dataset_integrity()
    if errors:
        logger.error("Dataset validation failed with %d error(s):", len(errors))
        for err in errors:
            print(f"  [ERROR] {err}")
        return 1

    logger.info("SUCCESS: Dataset '%s' passed all integrity checks (%d visits).", dataset.dataset_name, len(dataset.visits))
    return 0


def cmd_summary(manifest_path: Path) -> int:
    """Prints a structured summary of visits, entities, movements, and environmental conditions."""
    if not manifest_path.exists():
        logger.error("Manifest file not found: %s", manifest_path)
        return 1

    dataset = RealWorkbenchDataset.load_from_directory(manifest_path.parent)

    print("\n" + "=" * 70)
    print(f"DATASET SUMMARY: {dataset.dataset_name}")
    print("=" * 70)
    print(f"Description:  {dataset.description}")
    print(f"Total Visits: {len(dataset.visits)}")

    # Aggregate metadata & entity statistics
    all_entity_ids = set()
    event_counts = {}
    obs_counts = {}
    class_counts = {}

    for v in dataset.visits:
        for a in v.annotations:
            all_entity_ids.add(a.entity_id)
            ev = a.annotated_event.value
            event_counts[ev] = event_counts.get(ev, 0) + 1
            ob = a.physical_observation.value
            obs_counts[ob] = obs_counts.get(ob, 0) + 1
            cls_name = a.entity_label
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    print(f"Unique Tracked Physical Entities: {len(all_entity_ids)}")
    print(f"Object Classes: {', '.join(sorted(class_counts.keys()))}")
    print("\nPhysical Observation Distribution:")
    for k, cnt in sorted(obs_counts.items()):
        print(f"  {k:20s}: {cnt:3d}")

    print("\nAnnotated Ground-Truth Events:")
    for k, cnt in sorted(event_counts.items()):
        print(f"  {k:20s}: {cnt:3d}")

    print("\nVisit Sequence:")
    for v in dataset.visits:
        ann_cnt = len(v.annotations)
        env_notes = (v.metadata or {}).get("notes", "No notes")
        print(f"  [{v.visit_id}] {v.timestamp} | {v.session_name:40s} | {ann_cnt:2d} annotations")
    print("=" * 70 + "\n")
    return 0


def cmd_template(target_dir: Path, name: str) -> int:
    """Generates an empty, schema-compliant dataset directory and manifest template."""
    logger.info("Initializing dataset template at %s with name '%s'", target_dir, name)
    out_manifest = RealWorkbenchDataset.create_empty_template(target_dir)
    logger.info("Template manifest created at: %s", out_manifest)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ObjectMemory Real-World Dataset Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate a dataset manifest")
    p_val.add_argument("manifest", type=Path, help="Path to manifest.json")

    # summary
    p_sum = subparsers.add_parser("summary", help="Print dataset summary statistics")
    p_sum.add_argument("manifest", type=Path, help="Path to manifest.json")

    # template
    p_tpl = subparsers.add_parser("template", help="Create empty dataset template")
    p_tpl.add_argument("target_dir", type=Path, help="Target directory")
    p_tpl.add_argument("--name", type=str, default="real_workbench_sample", help="Dataset slug name")

    args = parser.parse_args()

    if args.command == "validate":
        sys.exit(cmd_validate(args.manifest))
    elif args.command == "summary":
        sys.exit(cmd_summary(args.manifest))
    elif args.command == "template":
        sys.exit(cmd_template(args.target_dir, args.name))


if __name__ == "__main__":
    main()
