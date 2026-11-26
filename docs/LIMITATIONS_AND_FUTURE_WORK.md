# ObjectMemory — Limitations and Future Work

**Document ID:** `OM-DOC-LIM-01`  
**Date:** November 21, 2026  
**Audience:** Engineers, Researchers, System Architects  
**Subject:** Technical Boundaries, Assumptions, and Research Directions  

---

## 1. Planar Homography and Scene Geometry Assumptions

### 1.1 The Dominant Plane Requirement
ObjectMemory currently relies on **2D planar homography** ($H \in \mathbb{R}^{3 \times 3}$) estimated via RANSAC on SIFT/ORB feature matches. This establishes a mathematically exact coordinate transformation if and only if:
1. The physical surface (tabletop, electronics workbench, desk) is approximately planar.
2. The observed objects lie on or directly adjacent to this dominant plane.

### 1.2 Boundary Failure Modes Observed
- **3D Parallax in Multi-Room Traversal**: In supplementary evaluations on public multi-session room benchmarks (e.g. ETH ASL Lounge), camera translation across 3D rooms with out-of-plane perspective shifts causes registration degradation due to depth parallax. Because objects and architectural surfaces exist across multiple depth planes ($0.5\text{m}$ to $5.0\text{m}$), 2D homography cannot model unconstrained non-planar perspective projections. ETH ASL Lounge is used strictly for supplementary planar registration evaluation, not as a full object-memory lifecycle benchmark.
- **Extreme Camera Angles**: Pitch angles exceeding $45^\circ$ or oblique grazing views distort bounding box envelopes, degrading area ratio and spatial overlap metrics.
- **Untextured Tabletops**: Plain monochrome surfaces (e.g. pure white plastic desks) produce sparse feature points ($<12$ inliers), requiring artificial markers (e.g. ArUco fiducials) to anchor registration.

---

## 2. Upstream Detector Dependencies and Failure Propagation

### 2.1 Detector False Negatives
The temporal memory engine decouples per-frame detections from persistent entity beliefs through an unobserved streak counter ($\text{unobserved\_streak} \ge 2$) and visual patch vacancy checks. However:
- If an object is missed by the detector for more consecutive visits than the threshold (e.g. 3 consecutive visits), the engine will transition the object to `PENDING_REMOVAL` and subsequently `CONFIRMED_REMOVED`, falsely declaring it absent.
- The system is inherently bounded by the open-vocabulary or closed-set detector's sensitivity.

### 2.2 Semantic Class Confusion
While the spatial-temporal associator implements conservative cross-class association:
- Severe category misclassifications with high bounding box drift or low appearance similarity can lead to unassigned tracks or fragmented entity lifecycles.

---

## 3. Appearance Extraction and Visual Signature Limits

### 3.1 148-Dimensional Appearance Descriptors
The visual signature extractor combines a 128-bin Hue-Saturation color histogram (65% weight) and a 20-bin texture gradient energy descriptor (35% weight).
- **Illumination Shifts**: Extreme colored lighting (e.g. monochromatic red or blue LEDs in laboratory settings) shifts the HS histogram and reduces cosine similarity below the association threshold ($0.65$).
- **Rotational Texture Invariance**: While the HS histogram is rotation-invariant, the Sobel texture subvector has limited orientation invariance under in-plane 2D rotations exceeding $45^\circ$.
- **Deep Feature Representation**: Deep foundation model embeddings (e.g. DINOv2 or CLIP visual features) were evaluated in research phases but are not integrated into the CPU-only inference pipeline due to latency and parameter overhead.

---

## 4. Evaluation and Dataset Limitations

### 4.1 Synthetic Dataset Nature
- The 25-observation benchmark `synthetic_workbench_v2` utilizes procedurally composited segmented photographic cutouts over procedural wooden tabletop textures.
- While realistic for evaluating detector triggers, shadow physics, and spatial association, it is **not a full 3D physics-based simulation** (e.g. Blender/Unreal Engine ray-tracing) and cannot simulate complex non-rigid deformability, liquid volume changes, or fine-grained subsurface scattering.

### 4.2 Genuine Physical Pilot Scope
- The genuine physical desk pilot (`desk_pilot_01`) comprises 7 observations. While sufficient to verify that the end-to-end pipeline operates on real smartphone camera captures without code modifications, it represents an exploratory pilot rather than a large-scale statistical validation across diverse multi-operator environments.

---

## 5. Future Work & Research Directions

### 5.1 SE(3) Rigid Pose and 3D Volumetric Mapping
- Transitioning from 2D planar homography to 3D point cloud registration (e.g. ICP or Truncated Signed Distance Field TSDF reconstruction) using RGB-D sensors (such as Intel RealSense or Azure Kinect).
- Estimating full 6-DOF camera poses ($[R | t]$) to decouple object bounding boxes from planar table constraints.

### 5.2 Deep Metric Feature Extractors for CPU
- Deploying quantized, distilled visual foundation models (e.g. MobileNetV4 or Tiny-DINO quantized to INT8) to replace handcrafted color/texture histograms with semantic visual embeddings that execute in under $20\text{ms}$ on CPU.

### 5.3 Active Visual Verification and Robotic Manipulation
- Integrating ObjectMemory into autonomous mobile manipulators (e.g. Boston Dynamics Spot or Franka Emika Panda).
- When an object enters `PENDING_REMOVAL`, commanding the robotic arm to move the camera for an active, close-range multi-angle inspection before confirming removal.

### 5.4 Multi-Surface Hierarchical Memory
- Supporting multi-level workspaces comprising multiple desks, shelves, and storage racks within a single unified facility map.
