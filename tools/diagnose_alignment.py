"""
Diagnostic script for inspecting image alignment, feature matches, and registration overlays.

Usage:
    python tools/diagnose_alignment.py --ref path/to/ref.jpg --query path/to/query.jpg [--out output.png] [--method SIFT]
    python tools/diagnose_alignment.py --demo [--out demo_overlay.png]
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from objectmemory.registration.models import AlignmentConfig, AlignmentStatus
from objectmemory.registration.registrar import PlanarRegistrar


def create_diagnostic_overlay(
    ref_image: np.ndarray,
    aligned_image: np.ndarray,
    overlap_mask: np.ndarray,
) -> np.ndarray:
    """
    Creates a visual diagnostic composite:
    Top-Left: Reference Image
    Top-Right: Query Image Warped into Reference Frame
    Bottom-Left: Alpha blend (50/50) in valid overlap region
    Bottom-Right: Difference Heatmap in valid overlap region
    """
    h, w = ref_image.shape[:2]
    # Ensure 3-channel BGR
    ref_bgr = ref_image if ref_image.ndim == 3 else cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
    alg_bgr = aligned_image if aligned_image.ndim == 3 else cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2BGR)

    # 1. Alpha blend
    mask_3ch = cv2.merge([overlap_mask, overlap_mask, overlap_mask]) > 0
    blend = ref_bgr.copy()
    overlap_blend = cv2.addWeighted(ref_bgr, 0.5, alg_bgr, 0.5, 0)
    blend[mask_3ch] = overlap_blend[mask_3ch]

    # 2. Difference Heatmap
    diff = cv2.absdiff(ref_bgr, alg_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    # Mask out non-overlapping areas
    heatmap[overlap_mask == 0] = [30, 30, 30]

    # Annotate panels
    cv2.putText(ref_bgr, "1. Reference Visit", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(alg_bgr, "2. Aligned Query", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(blend, "3. Alpha Overlay (50/50)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(heatmap, "4. Absolute Diff Heatmap", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    top_row = np.hstack([ref_bgr, alg_bgr])
    bot_row = np.hstack([blend, heatmap])
    return np.vstack([top_row, bot_row])


def run_demo(out_path: Optional[str] = None) -> None:
    """Generates a synthetic workbench test pair and runs alignment diagnostics."""
    print("=" * 70)
    print(" ObjectMemory: Diagnostic Alignment Demo (Synthetic Workbench) ")
    print("=" * 70)

    # Build synthetic workbench image
    from tests.test_registration import _generate_synthetic_workbench
    ref_img = _generate_synthetic_workbench(640, 480, seed=42)

    # Synthesize query with translation, rotation, and slight perspective tilt
    h, w = ref_img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[20, 15], [w - 15, 25], [w - 30, h - 20], [15, h - 10]])
    H_sim = cv2.getPerspectiveTransform(src_pts, dst_pts)
    query_img = cv2.warpPerspective(ref_img, H_sim, (w, h))

    print("Simulated test scenario: 640x480 workbench with perspective tilt.")
    run_diagnosis(ref_img, query_img, out_path=out_path)


def run_diagnosis(
    ref_img: np.ndarray,
    query_img: np.ndarray,
    method: str = "SIFT",
    out_path: Optional[str] = None,
) -> None:
    config = AlignmentConfig(feature_method=method)
    registrar = PlanarRegistrar(config)

    print(f"\nRunning alignment using method: {method} ...")
    result = registrar.register(ref_img, query_img)

    print("\n" + "-" * 40)
    print(" ALIGNMENT DIAGNOSTIC REPORT ")
    print("-" * 40)
    print(f"Outcome Status:           {result.status.value}")
    print(f"Alignment Succeeded:      {result.success}")
    print(f"Feature Method Used:      {result.method_used}")
    print(f"Execution Latency:        {result.latency_ms:.2f} ms")
    print(f"Keypoints Extracted:      Reference={result.num_keypoints_ref}, Query={result.num_keypoints_query}")
    print(f"Tentative Matches:        {result.num_tentative_matches}")
    print(f"RANSAC Inliers:           {result.num_inliers}")
    print(f"Inlier Ratio:             {result.inlier_ratio:.2%}")

    if result.reprojection_stats:
        stats = result.reprojection_stats
        print(f"Reprojection Error:       Mean={stats.mean:.2f}px, Median={stats.median:.2f}px, Max={stats.max_error:.2f}px")
    if result.overlap_area_fraction is not None:
        print(f"Overlap Area Fraction:    {result.overlap_area_fraction:.2%}")

    if not result.success:
        print(f"Failure Reason:           {result.failure_reason}")
    else:
        print("\nEstimated Homography Matrix H (Query -> Ref):")
        print(np.array2string(result.homography, precision=4, suppress_small=True))

        if out_path:
            overlay = create_diagnostic_overlay(ref_img, result.aligned_image, result.overlap_mask)
            cv2.imwrite(out_path, overlay)
            print(f"\n[Saved] Visual diagnostic overlay written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ObjectMemory Planar Alignment Diagnostics")
    parser.add_argument("--ref", type=str, help="Path to reference image")
    parser.add_argument("--query", type=str, help="Path to query image")
    parser.add_argument("--out", type=str, default=None, help="Output path for diagnostic overlay image")
    parser.add_argument("--method", type=str, default="SIFT", choices=["SIFT", "ORB"], help="Feature extractor")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demonstration")

    args = parser.parse_args()

    if args.demo or (not args.ref and not args.query):
        run_demo(out_path=args.out or "demo_alignment_overlay.png")
        return

    if not args.ref or not args.query:
        print("Error: Both --ref and --query must be provided when not running --demo.")
        sys.exit(1)

    ref_path = Path(args.ref)
    query_path = Path(args.query)

    if not ref_path.exists():
        print(f"Error: Reference image not found at {ref_path}")
        sys.exit(1)
    if not query_path.exists():
        print(f"Error: Query image not found at {query_path}")
        sys.exit(1)

    ref_img = cv2.imread(str(ref_path))
    query_img = cv2.imread(str(query_path))

    if ref_img is None:
        print(f"Error: Could not decode reference image: {ref_path}")
        sys.exit(1)
    if query_img is None:
        print(f"Error: Could not decode query image: {query_path}")
        sys.exit(1)

    run_diagnosis(ref_img, query_img, method=args.method, out_path=args.out)


if __name__ == "__main__":
    main()
