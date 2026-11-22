"""
Spatial feature coverage calculation for image co-registration.

Computes spatial distribution and coverage of inlier feature matches
across an image canvas to detect localized feature clustering and extrapolate
geometric uncertainty across the field of view.
"""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


class SpatialCoverageCalculator:
    """
    Computes spatial distribution and coverage of inlier feature matches
    across an image canvas using grid bin occupancy and convex hull fraction.
    """

    @staticmethod
    def calculate_coverage(
        points: np.ndarray,
        canvas_size: Tuple[int, int],
        grid_bins: Tuple[int, int] = (4, 4),
    ) -> float:
        """
        Calculates spatial feature coverage [0.0, 1.0].

        Args:
            points: (N, 2) array of (x, y) coordinates.
            canvas_size: (width, height) of the image canvas.
            grid_bins: (cols, rows) for spatial partitioning.
        """
        if points is None or len(points) < 4:
            return 0.0

        w, h = canvas_size
        if w <= 0 or h <= 0:
            return 0.0

        cols, rows = grid_bins
        cell_w = w / cols
        cell_h = h / rows

        occupied_cells = set()
        for pt in points:
            x, y = float(pt[0]), float(pt[1])
            c = min(cols - 1, max(0, int(x / cell_w)))
            r = min(rows - 1, max(0, int(y / cell_h)))
            occupied_cells.add((c, r))

        grid_coverage = len(occupied_cells) / (cols * rows)

        # Compute normalized convex hull area
        try:
            pts_f = points.astype(np.float32)
            hull = cv2.convexHull(pts_f)
            hull_area = cv2.contourArea(hull)
            total_area = float(w * h)
            hull_fraction = min(1.0, hull_area / total_area) if total_area > 0 else 0.0
        except Exception:
            hull_fraction = grid_coverage

        # Weighted composite score
        return float(np.clip(0.6 * grid_coverage + 0.4 * hull_fraction, 0.0, 1.0))
