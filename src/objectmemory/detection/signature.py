"""
Visual signature extraction combining normalized color histograms and texture descriptors.

Safeguards:
1. Invariant to patch dimensions via standard resizing.
2. Robust to illumination shifts via HSV chromaticity weighting and gradient angles.
3. Feature vectors are L2-normalized; cosine similarity is bounded in [0.0, 1.0].
"""

from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np

from objectmemory.detection.models import VisualSignature
from objectmemory.domain.models import BoundingBox


class VisualSignatureExtractor:
    """
    Extracts invariant multi-modal appearance signatures from image crops.
    """

    def __init__(
        self,
        hsv_bins: Tuple[int, int, int] = (8, 4, 4),
        gradient_bins: int = 16,
        standard_size: Tuple[int, int] = (64, 64),
    ) -> None:
        self.hsv_bins = hsv_bins
        self.gradient_bins = gradient_bins
        self.standard_size = standard_size

    def extract(self, image: np.ndarray, bbox: BoundingBox) -> VisualSignature:
        """
        Extracts appearance signature for the region defined by bbox in image.
        Image is not modified in-place.
        """
        h, w = image.shape[:2]
        # Coordinates clamp
        x0 = max(0, min(w - 1, int(round(bbox.x_min))))
        x1 = max(x0 + 1, min(w, int(round(bbox.x_max))))
        y0 = max(0, min(h - 1, int(round(bbox.y_min))))
        y1 = max(y0 + 1, min(h, int(round(bbox.y_max))))

        crop = image[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            # Fallback uniform vector for degenerate boxes, explicitly flagged as invalid
            total_dim = (self.hsv_bins[0] * self.hsv_bins[1] * self.hsv_bins[2]) + self.gradient_bins + 4
            vec = np.ones(total_dim, dtype=np.float32)
            vec /= np.linalg.norm(vec)
            return VisualSignature(
                color_histogram=vec[:128].copy(),
                texture_descriptor=vec[128:].copy(),
                combined_vector=vec,
                is_valid=False,
                failure_reason="Degenerate or empty crop (< 2x2 pixels or 0 area)",
            )

        # Ensure BGR 3-channel
        crop_bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        # 1. Color Histogram (HSV)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            list(self.hsv_bins),
            [0, 180, 0, 256, 0, 256],
        )
        color_hist = hist.flatten().astype(np.float32)
        color_sum = float(np.sum(color_hist))
        if color_sum > 1e-7:
            color_hist /= color_sum

        # 2. Texture Descriptor (Gradient orientations & statistics)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(gray, self.standard_size, interpolation=cv2.INTER_AREA)

        gx = cv2.Sobel(resized_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angles = (np.arctan2(gy, gx) * 180.0 / np.pi) % 360.0

        # Weighted angle histogram
        bin_width = 360.0 / self.gradient_bins
        angle_bins = (angles / bin_width).astype(int) % self.gradient_bins
        grad_hist = np.zeros(self.gradient_bins, dtype=np.float32)
        for b in range(self.gradient_bins):
            grad_hist[b] = float(np.sum(mag[angle_bins == b]))

        grad_sum = float(np.sum(grad_hist))
        if grad_sum > 1e-7:
            grad_hist /= grad_sum

        # Texture statistics (mean, variance, contrast)
        stats = np.array(
            [
                float(np.mean(mag)),
                float(np.std(mag)),
                float(np.mean(resized_gray)),
                float(np.std(resized_gray)),
            ],
            dtype=np.float32,
        )
        stats_norm = stats / (np.linalg.norm(stats) + 1e-7)
        texture_desc = np.concatenate([grad_hist, stats_norm])

        # Balance subvectors with L2 normalization before weighted combination
        c_norm = float(np.linalg.norm(color_hist))
        if c_norm > 1e-7:
            color_hist /= c_norm

        t_norm = float(np.linalg.norm(texture_desc))
        if t_norm > 1e-7:
            texture_desc /= t_norm

        # 3. Concatenate with 65% color and 35% texture weighting, then normalize
        combined = np.concatenate([0.65 * color_hist, 0.35 * texture_desc]).astype(np.float32)
        norm = float(np.linalg.norm(combined))
        if norm > 1e-7:
            combined /= norm

        return VisualSignature(
            color_histogram=color_hist,
            texture_descriptor=texture_desc,
            combined_vector=combined,
        )
