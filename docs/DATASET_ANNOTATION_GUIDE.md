# Real-Image Workbench Dataset Annotation Guide

This guide defines the standardized methodology for capturing, labeling, and verifying multi-visit image sequences for the ObjectMemory long-term visual memory framework.

---

## 1. Directory Structure

Datasets must follow the standardized physical workbench repository layout:

```text
data/real_workbench/
├── manifest.json            # Ground-truth annotations and visit metadata
├── reference_frame.png      # Baseline visit 1 photograph
├── pilot_memory.db          # Immutable evaluation SQLite database
└── visits/
    ├── visit_01.png         # Reference frame visit
    ├── visit_02.png         # Visit 2 (viewpoint jitter, lighting variation)
    ├── visit_03.png         # Visit 3 (ambient lighting shift)
    ├── visit_04.png         # Visit 4 (relocation / addition)
    ├── visit_05.png         # Visit 5 (temporary absence / occlusion)
    ├── visit_06.png         # Visit 6 (relocation / state confirmation)
    └── visit_07.png         # Visit 7 (reappearance / stabilization)
```

---

## 2. Coordinate System & Bounding Box Convention

- **Format**: `[x_min, y_min, x_max, y_max]` in raw pixel coordinates.
- **Origin**: $(0, 0)$ is the top-left corner of the image canvas.
- **Bounds**: Coordinates must satisfy $0 \le x_{min} < x_{max} \le W$ and $0 \le y_{min} < y_{max} \le H$.
- **Boundary Margin**: Objects within $15\text{ px}$ of an image edge must have boundary proximity noted, as camera framing truncation can induce apparent centroid shifts.

---

## 3. Human Observation Vocabulary

To preserve epistemic decoupling, annotators record **independent physical facts**, not internal system state predictions.

### 3.1 Physical Observation (`physical_observation`)
- `VISIBLE`: Entity is clearly observable and unobstructed.
- `PARTIALLY_OCCLUDED`: Entity is partially covered by another physical object or hand, but recognizable features remain.
- `FULLY_OCCLUDED`: Entity is physically present in the scene but fully hidden behind another object.
- `OUT_OF_FRAME`: Entity was moved or camera viewpoint cropped it out of the camera field of view.
- `ABSENT`: Entity has been removed from the workbench environment entirely.

### 3.2 Annotated Event (`annotated_event`)
- `UNCHANGED`: Entity remained stationary in the same physical location as the reference frame.
- `RELOCATED`: Entity was moved to a new physical location on the workbench.
- `NEWLY_APPEARED`: Entity was placed onto the workbench for the first time in this sequence.
- `ABSENT`: Entity is no longer present on the workbench.

---

## 4. Ambiguity Resolution Policy

1. **Occlusion vs Removal Ambiguity**:
   If an object disappears and another object is placed where it used to be (e.g. spiral notebook over scissors), annotators must NOT invent occluded status unless physical evidence proves the object is underneath. Mark as `ABSENT` with a clarifying note in `notes`.
2. **Detector Misclassification**:
   If an object is visually ambiguous (e.g. closed navy hardcover book appearing similar to a closed laptop), retain the persistent `entity_id` and annotate physical truth in `entity_label`.
3. **Template Protection**:
   Template datasets containing placeholder images or AI-generated synthetic crops must have `"is_template": true` in `manifest.json`. Do not set `is_template: false` until real physical photographs are substituted.
