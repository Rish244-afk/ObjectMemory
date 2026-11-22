"""
Aspect-ratio-preserving letterboxing and inverse coordinate transformations.

Provides explicit, documented geometric scaling, padding, and inverse mapping
for neural network detector inputs (e.g. YOLOv8, YOLO-World).

Equations:
Given original image dimensions (h_orig, w_orig) and model input size (in_w, in_h):
    r = min(in_w / w_orig, in_h / h_orig)
    new_unpad_w = round(w_orig * r)
    new_unpad_h = round(h_orig * r)
    pad_w = in_w - new_unpad_w
    pad_h = in_h - new_unpad_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

Inverse mapping from model coordinates (x_m, y_m) to original coordinates (x_orig, y_orig):
    x_orig = clamp((x_m - pad_left) / r, 0, w_orig)
    y_orig = clamp((y_m - pad_top) / r, 0, h_orig)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    """Metadata recording exact scaling and padding applied to an image."""
    orig_width: int
    orig_height: int
    target_width: int
    target_height: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    new_unpad_width: int
    new_unpad_height: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orig_width": self.orig_width,
            "orig_height": self.orig_height,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "scale": self.scale,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
            "new_unpad_width": self.new_unpad_width,
            "new_unpad_height": self.new_unpad_height,
        }


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_value: int = 114,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, LetterboxInfo]:
    """
    Resizes an image preserving aspect ratio and pads to target_size (width, height).

    Standard YOLOv8 / YOLO-World padding value is 114 (neutral grey).
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot letterbox an empty or None image.")

    h_orig, w_orig = image.shape[:2]
    in_w, in_h = target_size

    # 1. Compute scale ratio
    r = min(in_w / float(w_orig), in_h / float(h_orig))

    # 2. Compute unpadded resized dimensions
    new_unpad_w = int(round(w_orig * r))
    new_unpad_h = int(round(h_orig * r))

    # 3. Compute padding
    pad_w = in_w - new_unpad_w
    pad_h = in_h - new_unpad_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # 4. Resize
    if (w_orig, h_orig) != (new_unpad_w, new_unpad_h):
        resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=interpolation)
    else:
        resized = image.copy()

    # 5. Pad with constant value
    color = (pad_value, pad_value, pad_value) if resized.ndim == 3 else pad_value
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    info = LetterboxInfo(
        orig_width=w_orig,
        orig_height=h_orig,
        target_width=in_w,
        target_height=in_h,
        scale=r,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        new_unpad_width=new_unpad_w,
        new_unpad_height=new_unpad_h,
    )
    return padded, info


def inverse_transform_coordinates(
    x_min_model: float,
    y_min_model: float,
    x_max_model: float,
    y_max_model: float,
    info: LetterboxInfo,
    min_box_size: float = 1.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Transforms bounding box coordinates from model input space back to original image space.

    Applies:
    1. Inverse translation (subtract padding)
    2. Inverse scaling (divide by scale factor)
    3. Clamping to valid image boundaries [0, orig_width] and [0, orig_height]
    4. Rejection of degenerate or collapsed boxes
    """
    r = info.scale
    if r <= 0:
        return None

    # Un-pad and un-scale
    x1 = (x_min_model - float(info.pad_left)) / r
    y1 = (y_min_model - float(info.pad_top)) / r
    x2 = (x_max_model - float(info.pad_left)) / r
    y2 = (y_max_model - float(info.pad_top)) / r

    # Clamp to original image boundaries
    w_orig = float(info.orig_width)
    h_orig = float(info.orig_height)

    x_min_clamped = max(0.0, min(w_orig, min(x1, x2)))
    y_min_clamped = max(0.0, min(h_orig, min(y1, y2)))
    x_max_clamped = max(0.0, min(w_orig, max(x1, x2)))
    y_max_clamped = max(0.0, min(h_orig, max(y1, y2)))

    # Reject degenerate boxes
    box_w = x_max_clamped - x_min_clamped
    box_h = y_max_clamped - y_min_clamped

    if box_w < min_box_size or box_h < min_box_size:
        return None

    return (x_min_clamped, y_min_clamped, x_max_clamped, y_max_clamped)


def inverse_transform_cxcywh(
    cx_model: float,
    cy_model: float,
    w_model: float,
    h_model: float,
    info: LetterboxInfo,
    min_box_size: float = 1.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Transforms center-x, center-y, width, height from model space to original image space.
    """
    x_min = cx_model - 0.5 * w_model
    y_min = cy_model - 0.5 * h_model
    x_max = cx_model + 0.5 * w_model
    y_max = cy_model + 0.5 * h_model
    return inverse_transform_coordinates(x_min, y_min, x_max, y_max, info, min_box_size=min_box_size)
