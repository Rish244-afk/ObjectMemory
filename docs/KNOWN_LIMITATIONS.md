# ObjectMemory Known Limitations & Technical Constraints

This document provides a rigorous technical breakdown of the architectural assumptions, operational boundary conditions, and current limitations in ObjectMemory.

---

## 1. Planar Homography & Non-Coplanar Parallax

### Current Implementation
ObjectMemory relies on planar homography ($3 \times 3$ matrix $H$) estimated via SIFT feature matching and RANSAC to register subsequent visits to the canonical reference coordinate space.

### Technical Limitation
- Planar homography assumes that the scene is dominated by a single 2D plane (e.g. tabletop surface) or that camera motion is purely rotational with zero baseline displacement.
- When tall 3D objects (e.g. tall bottles, stacked boxes) are viewed with moderate baseline translation, parallax induces non-linear perspective distortions that cannot be parameterized by a single homography.
- **Consequence**: The base/footprint of the object aligns accurately with the workbench surface, but the top portion appears displaced, potentially causing bounding box expansion or coordinate jitter.

---

## 2. Field-of-View (FOV) Shifts & Resolution Discontinuities

### Observed in Pilot Dataset
In the 7-visit desk pilot dataset:
- Visits 1–3 were captured at approximately $780 \times 636$ resolution.
- Visits 4–7 were captured at approximately $584 \times 576$ resolution with a closer zoom/crop.

### Technical Limitation
- Fixed pixel movement thresholds (e.g. $\Delta_{thresh} = 35\text{ px}$) are scale-dependent. A 35-pixel translation in a $780 \times 636$ frame corresponds to a larger physical movement than in a higher-resolution close-up frame.
- While homography transforms bounding boxes into canonical reference coordinates, severe FOV shifts reduce feature overlap across visit boundaries, stressing SIFT match density.

---

## 3. Image Boundary Truncation & Apparent Movement

### Technical Limitation
When an object rests near the edge of the camera frame, small camera pan or tilt differences cause the bounding box to be partially clipped by the image canvas.
- When clipped, the bounding box centroid shifts inward toward the image interior.
- If unmitigated, this centroid shift can exceed movement thresholds and trigger false `CONFIRMED_MOVED` events.

### Mitigations Implemented
- The engine computes image boundary proximity ($< 15\text{ px}$ from frame edge).
- When clipped, displacement is classified as unreliable, and state transitions are held in provisional status unless overwhelming visual evidence is present.

---

## 4. Temporary Detector Class Confusion

### Observed Behavior
Visual detectors (such as MobileNet-SSD or lightweight CNNs) can temporarily oscillate between visually similar classes:
- Example: Navy hardcover book predicted as `book` in Visits 1–5 and 7, but predicted as `laptop` (confidence ~0.553) in Visit 6 due to color, aspect ratio, and keyboard-like texture proximity.

### Mitigations Implemented
- Conservative cross-class object association allows candidate matching across compatible classes when canonical spatial IoU is high ($\ge 0.40$), appearance signatures match, and no competing candidate exists.
- The persistent `WorkbenchEntity` preserves its established canonical identity rather than spawning spurious duplicate entities.

---

## 5. Lighting Extremes & Specular Glare

### Technical Limitation
- Intense direct sunlight or directional desk lamps cause localized specular blooming on metallic or ceramic objects (e.g. stainless steel scissors or ceramic mugs).
- High specular reflections eliminate local gradient texture, reducing SIFT keypoint yield and altering HSV color histograms.
- **Safety Safeguard**: When SIFT inliers drop below $20\%$, registration gating halts state transitions, preventing erroneous removal or displacement classifications.
