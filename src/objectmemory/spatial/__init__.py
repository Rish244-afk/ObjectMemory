"""
Spatial transformations and coordinate normalization for ObjectMemory.
"""

from objectmemory.spatial.transform import (
    normalize_bbox,
    denormalize_bbox,
    compute_centroid,
    compute_displacement_vector,
    project_to_isometric,
    is_point_in_bbox,
)

__all__ = [
    "normalize_bbox",
    "denormalize_bbox",
    "compute_centroid",
    "compute_displacement_vector",
    "project_to_isometric",
    "is_point_in_bbox",
]
