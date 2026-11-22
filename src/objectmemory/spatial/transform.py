"""
Spatial coordinate transformations and workbench scene projections.
"""

from __future__ import annotations
import math
from typing import Tuple, Union
from objectmemory.domain.models import BoundingBox


def normalize_bbox(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
    clamp: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Normalizes a canonical or raw BoundingBox to [0.0, 1.0] coordinates relative to image width and height.

    Returns:
        (norm_x_min, norm_y_min, norm_x_max, norm_y_max)
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({image_width}, {image_height})")

    x1 = bbox.x_min / float(image_width)
    y1 = bbox.y_min / float(image_height)
    x2 = bbox.x_max / float(image_width)
    y2 = bbox.y_max / float(image_height)

    if clamp:
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

    return (x1, y1, x2, y2)


def denormalize_bbox(
    norm_bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """
    Denormalizes [0, 1] relative coordinates to pixel coordinates.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({image_width}, {image_height})")

    x1, y1, x2, y2 = norm_bbox
    px1 = max(0.0, x1 * image_width)
    py1 = max(0.0, y1 * image_height)
    px2 = max(px1, x2 * image_width)
    py2 = max(py1, y2 * image_height)
    return BoundingBox(x_min=px1, y_min=py1, x_max=px2, y_max=py2)


def compute_centroid(bbox: BoundingBox) -> Tuple[float, float]:
    """
    Computes (center_x, center_y) for a bounding box.
    """
    return ((bbox.x_min + bbox.x_max) / 2.0, (bbox.y_min + bbox.y_max) / 2.0)


def compute_displacement_vector(
    bbox_a: BoundingBox,
    bbox_b: BoundingBox,
) -> Tuple[float, float, float]:
    """
    Computes displacement vector between two bounding box centroids.

    Returns:
        (dx, dy, euclidean_distance) where dx = x_b - x_a, dy = y_b - y_a.
    """
    c_a = compute_centroid(bbox_a)
    c_b = compute_centroid(bbox_b)
    dx = c_b[0] - c_a[0]
    dy = c_b[1] - c_a[1]
    dist = math.hypot(dx, dy)
    return (dx, dy, dist)


def project_to_isometric(
    x: float,
    y: float,
    z: float = 0.0,
    angle_deg: float = 30.0,
) -> Tuple[float, float]:
    """
    Projects 3D/2.5D coordinates (x, y, z) to 2D isometric screen space.
    Angle in degrees (default 30° standard isometric projection).
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    iso_x = (x - y) * cos_a
    iso_y = (x + y) * sin_a - z
    return (iso_x, iso_y)


def is_point_in_bbox(x: float, y: float, bbox: BoundingBox) -> bool:
    """
    Checks if point (x, y) lies inside bounding box [x_min, y_min, x_max, y_max].
    """
    return (bbox.x_min <= x <= bbox.x_max) and (bbox.y_min <= y <= bbox.y_max)
