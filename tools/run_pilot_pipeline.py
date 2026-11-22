"""
Pilot pipeline runner for ObjectMemory desk_pilot_01 dataset.

Differences from run_real_pipeline.py:
- Uses a permanent SQLite database at data/real_workbench/pilot_memory.db
  (not a tempfile — survives process restart).
- Skips is_evaluation_ready() gate (is_template blocks evaluator but NOT
  the pipeline runner; this script runs the pipeline regardless).
- Prints full per-visit detail: registration, detections, entity associations,
  state transitions, and change events.
- Does NOT inject ground-truth annotations into the prediction path.
- Clearly labels all results as synthetic pilot data.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from objectmemory.datasets.real_dataset import RealWorkbenchDataset
from objectmemory.detection.detector import OpenCVDNNDetector
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner

DATA_DIR   = Path("data/real_workbench")
DEFAULT_DB_PATH = DATA_DIR / "pilot_memory.db"
MODEL_PATH = Path("models/yolov8n.onnx")
CONF_THRESH = 0.35

SEPARATOR = "=" * 80


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="ObjectMemory Pilot Pipeline Runner")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to SQLite database")
    args = parser.parse_args()
    DB_PATH = Path(args.db)

    print(SEPARATOR)
    print(" ObjectMemory — Synthetic Pilot Pipeline Run")
    print(" Dataset provenance: synthetic_ai_generated_crops (NOT physical photographs)")
    print(" This run does NOT constitute real-world validation.")
    print(SEPARATOR)

    # --- Model check ---
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model not found: {MODEL_PATH}")
        return 1

    import hashlib
    with open(MODEL_PATH, "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    expected = "68f864475d06e2ec4037181052739f268eeac38d"
    sha1_ok = sha1 == expected
    print(f"\n[Step 1] Model verification")
    print(f"  Path:     {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"  SHA1:     {sha1}")
    print(f"  Match:    {sha1_ok}")
    if not sha1_ok:
        print("[ERROR] SHA1 mismatch — model may be corrupt or wrong version.")
        return 1

    detector = OpenCVDNNDetector(
        weights_path=str(MODEL_PATH), framework="onnx", architecture="yolov8"
    )
    info = detector.model_info()
    print(f"  Backend:  {info['backend']}")
    print(f"  Classes:  {info['classes_count']}")

    # --- Dataset ---
    print(f"\n[Step 2] Dataset validation")
    ds = RealWorkbenchDataset.load_from_directory(DATA_DIR)
    print(f"  Name:     {ds.dataset_name}")
    print(f"  Visits:   {len(ds.visits)}")
    print(f"  Template: {ds.metadata.get('is_template', False)}")
    errors = ds.validate_dataset_integrity()
    print(f"  Structural errors: {len(errors)}")
    for e in errors:
        print(f"    ERROR: {e}")

    # --- Permanent DB ---
    print(f"\n[Step 3] Persistent database")
    if DB_PATH.exists():
        print(f"  Existing DB found: {DB_PATH} — will reuse (idempotent)")
    else:
        print(f"  Creating new DB: {DB_PATH}")

    with SQLiteMemoryStore(DB_PATH) as store:
        runner = RealImagePipelineRunner(store=store, detector=detector)

        print(f"\n[Step 4] Pipeline execution ({len(ds.visits)} visits)\n")
        all_results = []
        for v in ds.visits:
            t0 = time.perf_counter()
            result = runner.process_visit(v, confidence_threshold=CONF_THRESH)
            elapsed = (time.perf_counter() - t0) * 1000

            reg = result.registration
            report = result.temporal_report
            batch = result.detection_batch

            print(f"  {'-'*76}")
            print(f"  {v.visit_id}  |  {v.session_name}")
            print(f"  {'-'*76}")
            print(f"  Image:         {v.image_path.name}  ({v.image_path.stat().st_size//1024} KB)")

            # Registration
            print(f"  Registration:  status={reg.status.value}  method={reg.method_used}")
            if reg.inlier_ratio is not None:
                print(f"                 inlier_ratio={reg.inlier_ratio:.3f}  success={reg.success}")

            # Detections
            print(f"  Detections:    {len(batch.detections)} objects  [{batch.latency_ms:.0f}ms]")
            for d in sorted(batch.detections, key=lambda x: -x.confidence):
                print(f"    {d.label:<18} conf={d.confidence:.3f}  "
                      f"bbox=({d.raw_bbox.x_min:.0f},{d.raw_bbox.y_min:.0f},"
                      f"{d.raw_bbox.x_max:.0f},{d.raw_bbox.y_max:.0f})")

            # Temporal report
            print(f"  Temporal:      new_entities={report.new_entities_created}  "
                  f"transitions={report.state_transitions_count}  "
                  f"skipped={getattr(report,'skipped_reason','—')}")

            # Entity belief updates
            if hasattr(report, 'belief_updates') and report.belief_updates:
                for bu in report.belief_updates:
                    arrow = f"{bu.prior_state.value} -> {bu.new_state.value}" if bu.transition_occurred else f"[{bu.new_state.value}]"
                    print(f"    entity={bu.entity_id:<20} {arrow}  vis={bu.visibility_status.value}")

            print(f"  Total latency: {elapsed:.0f}ms")
            print()
            all_results.append(result)
            v.release_cache()

        # --- Post-run: query stored data ---
        print(f"\n[Step 5] Stored state summary")
        entities = store.list_entities()
        print(f"  Total entities in DB: {len(entities)}")
        for ent in sorted(entities, key=lambda e: e.entity_id):
            bbox_str = f"({ent.canonical_bbox.x_min:.0f},{ent.canonical_bbox.y_min:.0f},{ent.canonical_bbox.x_max:.0f},{ent.canonical_bbox.y_max:.0f})" if ent.canonical_bbox else "null"
            print(f"    {ent.entity_id:<30} state={ent.current_state.value:<22} bbox={bbox_str}")

        print(f"\n  Change events:")
        for ent in sorted(entities, key=lambda e: e.entity_id):
            evts = store.list_change_events_for_entity(ent.entity_id)
            for evt in evts:
                print(f"    [{evt.entity_id}] {evt.prior_state.value} -> {evt.new_state.value}  "
                      f"session={evt.to_session_id}  reason={evt.explanation[:80] if evt.explanation else '—'}")

        print(f"\n  Sessions:")
        sessions = store.list_sessions()
        for s in sessions:
            print(f"    {s.session_id:<12} ts={s.timestamp}  img={s.image_path}")

        print(f"\n[Step 6] DB file size: {DB_PATH.stat().st_size} bytes")

    print(f"\n{SEPARATOR}")
    print(" Pilot run complete.")
    print(" These results are from synthetic AI-generated image crops.")
    print(" Do not interpret as real-world accuracy.")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
