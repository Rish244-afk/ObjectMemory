# Tripartite Architecture: Raw Detection Evidence, Registered Observations, and Persistent Temporal Entities

## 1. Executive Summary

In long-term physical visual memory systems like **ObjectMemory**, perceptual sensor evidence must be strictly decoupled from epistemic belief states. Previously, when image co-registration failed (e.g., due to viewpoint shifts, scale changes, or extreme camera rotation in live webcam monitoring), unmatched object detections were dropped by the temporal engine. This resulted in "zero detections" displayed in the user interface, despite the detector (e.g. YOLOv8n) successfully recognizing objects in the frame.

The **Live Camera Detection Persistence Fix** resolves this by introducing a strict three-tier separation between:
1. **Raw Visual Evidence (`RawDetection`)**
2. **Registered Observations (`Observation`)**
3. **Persistent Temporal Entities (`WorkbenchEntity`)**

---

## 2. Tripartite Data Separation

```
+---------------------------------------------------------------------------------------+
| 1. RAW DETECTION EVIDENCE (Perceptual Layer)                                          |
| Table: raw_detections                                                                 |
| Model: RawDetection                                                                   |
| Coordinates: Native Sensor Pixels [x_min, y_min, x_max, y_max]                        |
| Invariant: Persisted for EVERY valid detector output regardless of registration.      |
| Registration Status: "REGISTERED" or "UNREGISTERED"                                   |
+---------------------------------------------------------------------------------------+
                                           │
                    Is Planar Homography Verified & Reliable?
                                 /                  \
                            YES /                    \ NO
                               v                      v
+--------------------------------------------+  +---------------------------------------+
| 2. REGISTERED OBSERVATIONS (Spatial Layer) |  | Unregistered Detection Isolation     |
| Table: observations                        |  | - Canonical projection disabled       |
| Model: Observation                         |  | - Zero false entity associations      |
| Coordinates: Canonical Workspace Frame     |  | - Zero false removals/relocations     |
| - Feature Signatures Extracted             |  | - Frontend fallback: Dashed Amber Box |
| - Bipartite Kuhn-Munkres Assignment        |  +---------------------------------------+
+--------------------------------------------+
                       │
                       v
+---------------------------------------------------------------------------------------+
| 3. PERSISTENT TEMPORAL ENTITIES (Epistemic Belief Layer)                              |
| Table: entities, change_events                                                        |
| Models: WorkbenchEntity, ChangeEvent                                                  |
| States: CONFIRMED_PRESENT, CONFIRMED_MOVED, PENDING_REMOVAL, CONFIRMED_REMOVED, etc.  |
| Invariant: Only modified when registration and spatial alignment are verified.        |
+---------------------------------------------------------------------------------------+
```

### Tier 1: Raw Detection Evidence (`RawDetection`)
- **Nature**: Objective sensor fact. Represents what the visual detector detected at time $t$ in native camera sensor pixel coordinates $[x_{min}, y_{min}, x_{max}, y_{max}]$.
- **Storage**: Table `raw_detections` (schema version 3).
- **Properties**: `detection_id`, `session_id`, `timestamp`, `label`, `confidence`, `raw_bbox`, optional `canonical_bbox`, `registration_status`, and `metadata`.
- **Integrity**: Zero dependency on homography, feature matches, or baseline overlap. It is never discarded due to downstream alignment failure.

### Tier 2: Registered Observations (`Observation`)
- **Nature**: Spatial hypothesis. Represents an observation projected into the canonical workspace coordinate frame via a verified planar homography $H$.
- **Storage**: Table `observations`.
- **Properties**: `observation_id`, `session_id`, `entity_id`, `raw_bbox`, `canonical_bbox`, `visibility_status`, `identity_match_score`, `alignment_quality`.
- **Integrity**: Only generated when registration is verified ($R_{inliers} \ge 0.20$, plausible warp geometry, and verified overlap mask).

### Tier 3: Persistent Temporal Entities (`WorkbenchEntity` & `ChangeEvent`)
- **Nature**: Epistemic beliefs regarding physical objects over time.
- **Storage**: Tables `entities` and `change_events`.
- **Properties**: Tracks multi-visit identity, movement displacement distance $\Delta d$, baseline return, occlusion streaks, and absence confirmation.
- **Integrity**: State transitions are strictly governed by domain policy. When registration fails, epistemic transitions are frozen (`VisibilityStatus.UNOBSERVED`, streak does not increment), preventing hallucinated deletions or relocations.

---

## 3. Registration Failure Safeguards

When registration between a query frame and the reference baseline fails:
1. **Mathematical Indeterminacy**: Without a valid homography $H$, the spatial transformation between pixel coordinates and canonical coordinates is unconstrained.
2. **No False Entity Creation**: Unregistered detections are not added to `entities`. They do not create spurious persistent objects.
3. **No False Identity Associations**: Bipartite matching across sessions is disabled for unregistered frames to prevent false cross-object mergers.
4. **No False Removals**: Existing entities cannot be confirmed absent or moved when the viewpoint cannot be verified. They remain in their established state with `VisibilityStatus.UNOBSERVED`.
5. **Auditability**: Unregistered visits record `RegistrationResult.failure_reason` and maintain raw detections for forensic verification.

---

## 4. Frontend Visualization Strategy

In the ObjectMemory UI (`WorkspaceOverviewView.tsx` & `BoundingBoxCanvas.tsx`):
- **Hero Canvas Fallback**: When viewing a visit where registered observations are empty but raw detections exist (`isUnregisteredSession`):
  - Canvas maps bounding boxes directly from `raw_detections`.
  - Boxes are rendered with a distinct dashed amber style (`strokeColor="#F59E0B"`, `fillColor="rgba(245, 158, 11, 0.12)"`, `strokeDasharray="6 3"`).
  - Status banner displays: *"Objects detected, but scene registration failed. Cross-visit change analysis is unavailable."*
- **Inspector Strip**: Displays `Raw Detections in Frame (X) [Unregistered Viewpoint]` with amber `UNREGISTERED` badges.
- **History Table**: Displays `{raw_detections_count} (unreg)` in amber, informing the user that detections occurred even though cross-visit registration failed.
