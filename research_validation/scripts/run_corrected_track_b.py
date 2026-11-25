#!/usr/bin/env python3
"""
Runs the corrected Track B Oracle benchmark on physical pilot v2.
Saves results to research_validation/physical_pilot_v2_track_b_oracle_corrected_results.json.
Does NOT overwrite historical results.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root and src/scripts
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from objectmemory.benchmarks.manifest import ManifestLoader
from objectmemory.datasets.physical_workbench import (
    AbsenceReason,
    PhysicalVisibility,
    PhysicalWorkbenchAnnotation,
    PhysicalWorkbenchDataset,
    PhysicalWorkbenchSequence,
    PhysicalWorkbenchVisit,
)
from objectmemory.domain.states import EpistemicState
from evaluate_physical_workbench import evaluate_physical_track_b

def main():
    manifest_path = REPO_ROOT / "data" / "physical_pilot_v2" / "manifest_v2.json"
    m = ManifestLoader.load(manifest_path, check_files=True)

    vis_map = {
        "CLEARLY_VISIBLE": PhysicalVisibility.VISIBLE,
        "PARTIALLY_OCCLUDED": PhysicalVisibility.PARTIALLY_OCCLUDED,
        "HEAVILY_OCCLUDED": PhysicalVisibility.FULLY_OCCLUDED,
        "UNOBSERVED": PhysicalVisibility.ABSENT,
    }

    pw_seqs = {}
    for seq in m.sequences:
        visits = []
        for idx, obs in enumerate(seq.observations, start=1):
            anns = []
            for obj in obs.objects:
                vis = vis_map.get(obj.visibility_status, PhysicalVisibility.VISIBLE)
                anns.append(
                    PhysicalWorkbenchAnnotation(
                        sequence_id=seq.sequence_id,
                        visit_id=obs.observation_id,
                        object_id=obj.object_id,
                        class_label=obj.class_name,
                        bbox=obj.bounding_box,
                        visibility=vis,
                        expected_state=EpistemicState(obj.object_state),
                        expected_event=obj.event_id,
                    )
                )
            w, h = obs.metadata.get("image_dimensions", [1600, 1200])
            visits.append(
                PhysicalWorkbenchVisit(
                    visit_id=obs.observation_id,
                    visit_index=idx,
                    image_relative_path=obs.raw_image_path,
                    image_absolute_path=str(obs.image_path),
                    expected_dimensions=(int(w), int(h)),
                    timestamp=obs.timestamp,
                    annotations=anns,
                )
            )
        pw_seqs[seq.sequence_id] = PhysicalWorkbenchSequence(
            sequence_id=seq.sequence_id,
            scenario_name=seq.challenge_mode or seq.sequence_id,
            description="",
            visits=visits,
        )

    pw_ds = PhysicalWorkbenchDataset(
        dataset_dir=str(m.dataset_root),
        manifest_data={},
        sequences=pw_seqs,
    )

    print("Running corrected Track B Oracle Benchmark...")
    results = evaluate_physical_track_b(pw_ds)

    out_file = REPO_ROOT / "research_validation" / "physical_pilot_v2_track_b_oracle_corrected_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Saved corrected Track B results to: {out_file}")
    print("Primary Metric:", results["primary_metric"])
    print("Identity Switches:", results["total_identity_switches"])
    print("Duplicate Tracks:", results["total_duplicate_tracks"])
    print("Event Accuracy:", results.get("event_accuracy"))

    # Print per sequence breakdown
    for sid, s in results["sequences"].items():
        el = s["entity_level_evaluation"]
        c = el["correct_entity_visits"]
        t = el["total_entity_visits_evaluated"]
        acc = el["entity_state_accuracy"]
        print(f"  {sid}: {c}/{t} ({acc:.2%})")

if __name__ == "__main__":
    main()
