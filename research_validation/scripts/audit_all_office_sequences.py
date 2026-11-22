#!/usr/bin/env python3
"""
Audit all seven per-sequence trajectory files from OpenLORIS-Scene office benchmark.
"""

import datetime
from pathlib import Path
import numpy as np

gt_base = Path("C:/Users/Rishh/.gemini/antigravity/scratch/objectmemory/research_validation/data/raw/openloris_scene/groundtruth/extracted/per-sequence")
seqs = ["office1-1", "office1-2", "office1-3", "office1-4", "office1-5", "office1-6", "office1-7"]

print(f"{'Sequence':<11} | {'Start Time (UTC)':<20} | {'Duration':<9} | {'Poses':<6} | {'Rate':<8} | {'Gaps>10ms':<9} | {'Max Gap':<8} | {'Path Len':<9} | {'Net Disp':<8}")
print("-" * 110)

all_data = []

for s in seqs:
    p = gt_base / s / "groundtruth.txt"
    if not p.exists():
        print(f"{s:<11} | NOT FOUND")
        continue

    ts = []
    poses = []
    with open(p, "r", encoding="utf-8") as f:
        for l in f:
            if not l.strip() or l.startswith("#"):
                continue
            parts = l.split()
            ts.append(float(parts[0]))
            poses.append([float(x) for x in parts[1:8]])

    ts = np.array(ts)
    poses = np.array(poses)

    t0 = ts[0]
    t1 = ts[-1]
    dur = t1 - t0
    d_t = np.diff(ts)
    d_pos = np.diff(poses[:, :3], axis=0)

    path_len = np.sum(np.linalg.norm(d_pos, axis=1))
    net_disp = np.linalg.norm(poses[-1, :3] - poses[0, :3])

    mins = np.min(poses[:, :3], axis=0)
    maxs = np.max(poses[:, :3], axis=0)

    gaps_10 = int(np.sum(d_t > 0.010))
    max_gap = float(np.max(d_t))

    med_dt = float(np.median(d_t))
    freq = 1.0 / med_dt if med_dt > 0 else 0

    dt0 = datetime.datetime.fromtimestamp(t0, datetime.timezone.utc)
    dt1 = datetime.datetime.fromtimestamp(t1, datetime.timezone.utc)

    all_data.append({
        "seq": s,
        "t0": t0,
        "t1": t1,
        "dt0_str": dt0.strftime("%Y-%m-%d %H:%M:%S"),
        "dt1_str": dt1.strftime("%Y-%m-%d %H:%M:%S"),
        "dur": dur,
        "poses": len(ts),
        "freq": freq,
        "gaps_10": gaps_10,
        "max_gap": max_gap,
        "path_len": path_len,
        "net_disp": net_disp,
        "bbox": [mins, maxs],
        "start_pos": poses[0, :3],
        "end_pos": poses[-1, :3]
    })

    print(f"{s:<11} | {dt0.strftime('%Y-%m-%d %H:%M:%S'):<20} | {dur:7.2f} s | {len(ts):<6} | {freq:5.1f} Hz | {gaps_10:<9} | {max_gap*1000:6.1f}ms | {path_len:7.2f} m | {net_disp:6.2f} m")

print("-" * 110)

# Print bounding boxes and positions
print("\n=== Spatial Bounding Boxes & Trajectory Boundaries ===")
for d in all_data:
    mins, maxs = d["bbox"]
    print(f"{d['seq']}:")
    print(f"  Start: [{d['start_pos'][0]:.3f}, {d['start_pos'][1]:.3f}, {d['start_pos'][2]:.3f}] -> End: [{d['end_pos'][0]:.3f}, {d['end_pos'][1]:.3f}, {d['end_pos'][2]:.3f}]")
    print(f"  BBox:  X [{mins[0]:.3f}, {maxs[0]:.3f}] (span {maxs[0]-mins[0]:.2f}m), Y [{mins[1]:.3f}, {maxs[1]:.2f}] (span {maxs[1]-mins[1]:.2f}m), Z [{mins[2]:.3f}, {maxs[2]:.3f}]")
