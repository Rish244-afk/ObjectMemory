"""
Procedural Photorealistic Workbench v2 Dataset Generator for ObjectMemory.

Generates 5 controlled sequences of 5 observations each (25 total observations)
at 1080x720 resolution using segmented photographic assets, realistic contact/cast
shadows, textured wooden tabletop, fiducial markers, and precise ground truth
annotations adhering to Manifest v2.0 format.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def generate_wood_tabletop(
    width: int = 1080, height: int = 720, seed: int = 42
) -> np.ndarray:
    """
    Generates a high-frequency, feature-rich wooden tabletop background
    suitable for feature-based planar homography registration (SIFT/ORB).
    """
    np.random.seed(seed)
    x = np.linspace(0, 14.0, width)
    y = np.linspace(0, 9.0, height)
    xx, yy = np.meshgrid(x, y)

    # Multi-octave sinusoidal wood grain
    grain1 = np.sin(xx * 2.8 + np.sin(yy * 1.5) * 1.8) * 35.0
    grain2 = np.sin(xx * 5.6 + np.sin(yy * 3.0) * 0.9) * 18.0
    grain3 = np.sin(xx * 14.0) * 8.0
    wood = 165.0 + grain1 + grain2 + grain3

    # Warm oak tone in BGR
    b = np.clip(wood * 0.72, 0, 255).astype(np.uint8)
    g = np.clip(wood * 0.88, 0, 255).astype(np.uint8)
    r = np.clip(wood * 1.05, 0, 255).astype(np.uint8)
    table = np.stack([b, g, r], axis=-1)

    # Add realistic tabletop micro-texture noise
    noise = (np.random.randn(height, width, 3) * 7.0).astype(np.int16)
    table = np.clip(table.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw tabletop plank seams (subtle vertical lines every ~270px)
    for px in [270, 540, 810]:
        cv2.line(
            table,
            (px, 0),
            (px, height),
            (int(b.mean() * 0.75), int(g.mean() * 0.75), int(r.mean() * 0.75)),
            1,
        )

    # Add ArUco fiducials at corners for reliable physical workbench alignment
    # 4 markers: corners [40, 40], [1000, 40], [40, 640], [1000, 640]
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for idx, (mx, my) in enumerate(
        [(40, 40), (1000, 40), (40, 640), (1000, 640)]
    ):
        marker = cv2.aruco.generateImageMarker(aruco_dict, idx, 40)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        table[my : my + 40, mx : mx + 40] = marker_bgr

    return table


def composite_asset(
    canvas: np.ndarray,
    rgba_asset: np.ndarray,
    pos_x: int,
    pos_y: int,
    light_offset: Tuple[int, int] = (12, 16),
    shadow_opacity: float = 0.42,
    brightness_factor: float = 1.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Composites an RGBA asset onto canvas with realistic directional cast shadow
    and contact ambient occlusion. Returns updated canvas and bounding box [x1, y1, x2, y2].
    """
    ch, cw = canvas.shape[:2]
    ah, aw = rgba_asset.shape[:2]

    # Ensure bounds
    if pos_x < 0 or pos_y < 0 or pos_x + aw > cw or pos_y + ah > ch:
        raise ValueError(
            f"Asset at ({pos_x}, {pos_y}) of size ({aw}, {ah}) exceeds canvas ({cw}, {ch})"
        )

    alpha = rgba_asset[:, :, 3].astype(np.float32) / 255.0

    # 1. Cast shadow
    sox, soy = light_offset
    sh_x1 = max(0, pos_x + sox)
    sh_y1 = max(0, pos_y + soy)
    sh_x2 = min(cw, pos_x + aw + sox)
    sh_y2 = min(ch, pos_y + ah + soy)

    shadow_mask = np.zeros((ch, cw), dtype=np.float32)
    src_x1 = max(0, -pos_x - sox)
    src_y1 = max(0, -pos_y - soy)
    src_x2 = src_x1 + (sh_x2 - sh_x1)
    src_y2 = src_y1 + (sh_y2 - sh_y1)

    shadow_mask[sh_y1:sh_y2, sh_x1:sh_x2] = (
        alpha[src_y1:src_y2, src_x1:src_x2] * shadow_opacity
    )
    # Gaussian blur for soft shadow
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 7.0)

    for c in range(3):
        canvas[:, :, c] = np.clip(
            canvas[:, :, c].astype(np.float32) * (1.0 - shadow_mask), 0, 255
        ).astype(np.uint8)

    # 2. Contact shadow (darkening at bottom 15% of asset)
    contact_mask = np.zeros((ch, cw), dtype=np.float32)
    contact_h = max(4, int(ah * 0.15))
    contact_strip = (
        alpha[ah - contact_h : ah, :]
        * np.linspace(0.1, 0.6, contact_h)[:, None]
    )
    contact_mask[
        pos_y + ah - contact_h : pos_y + ah, pos_x : pos_x + aw
    ] = contact_strip
    contact_mask = cv2.GaussianBlur(contact_mask, (9, 9), 3.0)

    for c in range(3):
        canvas[:, :, c] = np.clip(
            canvas[:, :, c].astype(np.float32) * (1.0 - contact_mask * 0.5),
            0,
            255,
        ).astype(np.uint8)

    # 3. Paste asset with alpha
    bgr_asset = rgba_asset[:, :, :3].astype(np.float32)
    if brightness_factor != 1.0:
        bgr_asset = np.clip(bgr_asset * brightness_factor, 0, 255)

    alpha_3d = alpha[:, :, None]
    target_roi = canvas[pos_y : pos_y + ah, pos_x : pos_x + aw].astype(
        np.float32
    )
    blended = target_roi * (1.0 - alpha_3d) + bgr_asset * alpha_3d
    canvas[pos_y : pos_y + ah, pos_x : pos_x + aw] = np.clip(
        blended, 0, 255
    ).astype(np.uint8)

    # Compute tight bounding box
    ys, xs = np.where(alpha > 0.1)
    if len(xs) > 0 and len(ys) > 0:
        bbox = [
            float(pos_x + xs.min()),
            float(pos_y + ys.min()),
            float(pos_x + xs.max()),
            float(pos_y + ys.max()),
        ]
    else:
        bbox = [
            float(pos_x),
            float(pos_y),
            float(pos_x + aw),
            float(pos_y + ah),
        ]

    return canvas, bbox


def apply_occluder(
    canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """
    Renders a physical occluder (manila cardboard folder) covering the specified region.
    """
    folder = np.full(
        (y2 - y1, x2 - x1, 3), (120, 160, 205), dtype=np.uint8
    )  # manila color BGR
    # Add subtle border and texture
    cv2.rectangle(
        folder, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), (90, 130, 175), 2
    )
    # Cast shadow under folder
    shadow = np.zeros_like(canvas, dtype=np.float32)
    shadow[y1 + 10 : y2 + 10, x1 + 10 : x2 + 10] = 0.4
    shadow = cv2.GaussianBlur(shadow, (15, 15), 5.0)
    canvas = np.clip(
        canvas.astype(np.float32) * (1.0 - shadow), 0, 255
    ).astype(np.uint8)
    canvas[y1:y2, x1:x2] = folder
    return canvas


def warp_viewpoint(
    image: np.ndarray,
    angle_deg: float = 2.5,
    tx: float = 15.0,
    ty: float = 10.0,
    scale: float = 1.02,
) -> np.ndarray:
    """
    Simulates a slight camera perspective tilt and translation.
    """
    h, w = image.shape[:2]
    # Source points
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Perspective shift
    rad = math.radians(angle_deg)
    d_x = math.tan(rad) * 15.0
    d_y = math.sin(rad) * 12.0

    dst_pts = np.float32(
        [
            [tx, ty],
            [w * scale + tx + d_x, ty + d_y],
            [w * scale + tx, h * scale + ty],
            [tx - d_x, h * scale + ty - d_y],
        ]
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped, M


def build_synthetic_workbench_dataset(output_dir: Path) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"

    # Load high-resolution segmented assets
    assets = {
        "cup": cv2.imread(
            str(assets_dir / "cup.png"), cv2.IMREAD_UNCHANGED
        ),
        "bottle": cv2.imread(
            str(assets_dir / "bottle.png"), cv2.IMREAD_UNCHANGED
        ),
        "scissors": cv2.imread(
            str(assets_dir / "scissors.png"), cv2.IMREAD_UNCHANGED
        ),
        "plant": cv2.imread(
            str(assets_dir / "plant.png"), cv2.IMREAD_UNCHANGED
        ),
        "book": cv2.imread(
            str(assets_dir / "book.png"), cv2.IMREAD_UNCHANGED
        ),
    }

    # Canonical base coordinates on 1080x720 canvas
    base_positions = {
        "cup_01": ("cup", "cup", 120, 260),
        "bottle_01": ("bottle", "bottle", 360, 180),
        "book_01": ("book", "book", 520, 360),
        "scissors_01": ("scissors", "scissors", 680, 180),
        "plant_01": ("plant", "potted plant", 870, 360),
    }

    sequences_data: List[Dict[str, Any]] = []

    # Sequence definitions:
    # S01: Stationary
    # S02: Relocation & return (cup moved v2, v3, bottle moved v4, cup returns v5)
    # S03: Removal & confirmation (bottle removed v2-v5, scissors removed v4-v5)
    # S04: Occlusion & recovery (folder occludes book v2, occludes scissors v3, removed v4-v5)
    # S05: Lighting & viewpoint perturbations (dim v2, warp v3, bright v4, shift v5)

    seq_configs = [
        ("S01_stationary", "Stationary baseline across 5 visits"),
        ("S02_relocation_return", "Object relocation and return to canonical position"),
        ("S03_removal_confirmation", "Multi-visit object removal with vacancy confirmation"),
        ("S04_occlusion_recovery", "Transient occlusion, persistence and reappearance"),
        ("S05_lighting_viewpoint", "Illumination changes and camera viewpoint perturbations"),
    ]

    for s_idx, (seq_id, seq_desc) in enumerate(seq_configs, start=1):
        obs_list: List[Dict[str, Any]] = []

        for v_idx in range(1, 6):
            obs_id = f"{seq_id}_visit_{v_idx:02d}"
            img_rel_path = f"images/{obs_id}.png"
            img_full_path = output_dir / img_rel_path

            # Generate fresh base tabletop with slight variation
            canvas = generate_wood_tabletop(width=1080, height=720, seed=100 * s_idx + v_idx)

            gt_objects: List[Dict[str, Any]] = []
            timestamp = f"2026-11-21T10:{s_idx:02d}:{v_idx*10:02d}Z"

            # Determine object placements and states for each sequence
            if seq_id == "S01_stationary":
                # All stationary
                for obj_id, (asset_key, coco_class, bx, by) in base_positions.items():
                    canvas, bbox = composite_asset(canvas, assets[asset_key], bx, by)
                    gt_objects.append({
                        "object_id": obj_id,
                        "class_name": coco_class,
                        "bounding_box": bbox,
                        "visibility_status": "CLEARLY_VISIBLE",
                        "object_state": "CONFIRMED_PRESENT",
                        "timestamp": timestamp,
                    })

            elif seq_id == "S02_relocation_return":
                # S02: Cup relocated in v2-v4, returns in v5. Bottle moves in v4-v5.
                for obj_id, (asset_key, coco_class, bx, by) in base_positions.items():
                    cur_x, cur_y = bx, by
                    state = "CONFIRMED_PRESENT"

                    if obj_id == "cup_01":
                        if v_idx in (2, 3, 4):
                            cur_x, cur_y = 220, 430  # Moved ~198px
                            state = "MOVED" if v_idx == 2 else "CONFIRMED_PRESENT"
                        elif v_idx == 5:
                            cur_x, cur_y = bx, by  # Returned!
                            state = "CONFIRMED_PRESENT"

                    if obj_id == "bottle_01":
                        if v_idx in (4, 5):
                            cur_x, cur_y = 130, 180  # Moved 230px
                            state = "MOVED" if v_idx == 4 else "CONFIRMED_PRESENT"

                    canvas, bbox = composite_asset(canvas, assets[asset_key], cur_x, cur_y)
                    gt_objects.append({
                        "object_id": obj_id,
                        "class_name": coco_class,
                        "bounding_box": bbox,
                        "visibility_status": "CLEARLY_VISIBLE",
                        "object_state": state,
                        "timestamp": timestamp,
                    })

            elif seq_id == "S03_removal_confirmation":
                # S03: Bottle removed in v2-v5. Scissors removed in v4-v5.
                for obj_id, (asset_key, coco_class, bx, by) in base_positions.items():
                    if obj_id == "bottle_01" and v_idx >= 2:
                        # Removed!
                        state = "PENDING_REMOVAL" if v_idx == 2 else "CONFIRMED_REMOVED"
                        gt_objects.append({
                            "object_id": obj_id,
                            "class_name": coco_class,
                            "bounding_box": None,
                            "visibility_status": "ABSENT",
                            "object_state": state,
                            "timestamp": timestamp,
                        })
                        continue

                    if obj_id == "scissors_01" and v_idx >= 4:
                        # Removed!
                        state = "PENDING_REMOVAL" if v_idx == 4 else "CONFIRMED_REMOVED"
                        gt_objects.append({
                            "object_id": obj_id,
                            "class_name": coco_class,
                            "bounding_box": None,
                            "visibility_status": "ABSENT",
                            "object_state": state,
                            "timestamp": timestamp,
                        })
                        continue

                    canvas, bbox = composite_asset(canvas, assets[asset_key], bx, by)
                    gt_objects.append({
                        "object_id": obj_id,
                        "class_name": coco_class,
                        "bounding_box": bbox,
                        "visibility_status": "CLEARLY_VISIBLE",
                        "object_state": "CONFIRMED_PRESENT",
                        "timestamp": timestamp,
                    })

            elif seq_id == "S04_occlusion_recovery":
                # S04: Cardboard folder occludes book in v2, scissors in v3, removed in v4-v5
                # First paste all objects
                drawn_bboxes = {}
                for obj_id, (asset_key, coco_class, bx, by) in base_positions.items():
                    canvas, bbox = composite_asset(canvas, assets[asset_key], bx, by)
                    drawn_bboxes[obj_id] = bbox

                if v_idx == 2:
                    # Folder partially occludes book
                    canvas = apply_occluder(canvas, 500, 340, 720, 560)
                elif v_idx == 3:
                    # Folder covers scissors completely
                    canvas = apply_occluder(canvas, 660, 160, 980, 310)

                for obj_id, (asset_key, coco_class, bx, by) in base_positions.items():
                    bbox = drawn_bboxes[obj_id]
                    vis = "CLEARLY_VISIBLE"
                    state = "CONFIRMED_PRESENT"
                    if v_idx == 2 and obj_id == "book_01":
                        vis = "PARTIALLY_OCCLUDED"
                    elif v_idx == 3 and obj_id == "scissors_01":
                        vis = "FULLY_OCCLUDED"
                        bbox = None  # fully occluded

                    gt_objects.append({
                        "object_id": obj_id,
                        "class_name": coco_class,
                        "bounding_box": bbox,
                        "visibility_status": vis,
                        "object_state": state,
                        "timestamp": timestamp,
                    })

            elif seq_id == "S05_lighting_viewpoint":
                # S05: Lighting & viewpoint perturbations
                brightness = 1.0
                warp = False
                warp_angle = 0.0
                tx, ty = 0.0, 0.0

                if v_idx == 2:
                    brightness = 0.65  # Dimmed
                elif v_idx == 3:
                    warp = True
                    warp_angle = 2.5
                    tx, ty = 14.0, 10.0
                elif v_idx == 4:
                    brightness = 1.25  # Bright directional
                elif v_idx == 5:
                    warp = True
                    warp_angle = -1.8
                    tx, ty = -10.0, -8.0

                drawn_bboxes = {}
                for obj_id, (asset_key, coco_class, bx, by) in base_positions.items():
                    canvas, bbox = composite_asset(
                        canvas, assets[asset_key], bx, by, brightness_factor=brightness
                    )
                    drawn_bboxes[obj_id] = bbox

                if warp:
                    canvas, M = warp_viewpoint(canvas, angle_deg=warp_angle, tx=tx, ty=ty)
                    # Transform ground truth bboxes
                    for obj_id, bbox in drawn_bboxes.items():
                        pts = np.float32([
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]],
                        ]).reshape(-1, 1, 2)
                        warped_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
                        warped_bbox = [
                            float(warped_pts[:, 0].min()),
                            float(warped_pts[:, 1].min()),
                            float(warped_pts[:, 0].max()),
                            float(warped_pts[:, 1].max()),
                        ]
                        drawn_bboxes[obj_id] = warped_bbox

                for obj_id, (asset_key, coco_class, bx, by) in base_positions.items():
                    gt_objects.append({
                        "object_id": obj_id,
                        "class_name": coco_class,
                        "bounding_box": drawn_bboxes[obj_id],
                        "visibility_status": "CLEARLY_VISIBLE",
                        "object_state": "CONFIRMED_PRESENT",
                        "timestamp": timestamp,
                    })

            # Save generated image
            cv2.imwrite(str(img_full_path), canvas)

            obs_list.append({
                "observation_id": obs_id,
                "timestamp": timestamp,
                "image_path": img_rel_path,
                "objects": gt_objects,
                "metadata": {
                    "sequence_id": seq_id,
                    "visit_index": v_idx,
                    "resolution": [1080, 720],
                    "render_type": "photorealistic_procedural_compositor",
                },
            })

        sequences_data.append({
            "sequence_id": seq_id,
            "environment_id": "synthetic_workbench_env",
            "dataset_name": "synthetic_workbench_v2",
            "coordinate_convention": "pixel_xyxy",
            "split": "test",
            "metadata": {
                "description": seq_desc,
                "num_visits": 5,
            },
            "observations": obs_list,
        })

    # Construct v2.0 manifest
    manifest_v2 = {
        "manifest_version": "2.0",
        "dataset_name": "synthetic_workbench_v2",
        "provenance": "SYNTHETIC_GENERATED",
        "metadata": {
            "title": "ObjectMemory Photorealistic Synthetic Workbench Benchmark v2",
            "version": "2.0",
            "author": "Antigravity Research Evaluation Harness",
            "total_sequences": len(sequences_data),
            "total_observations": sum(len(s["observations"]) for s in sequences_data),
            "resolution": [1080, 720],
            "asset_source": "Segmented physical photographs with contact shadow physics",
            "license": "Research evaluation only",
        },
        "sequences": sequences_data,
    }

    manifest_path = output_dir / "manifest_v2.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_v2, f, indent=2)

    # Legacy copy for compatibility
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_v2, f, indent=2)

    print(f"Generated {len(sequences_data)} sequences with 25 total images in {output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    out = Path("data/synthetic_workbench_v2")
    build_synthetic_workbench_dataset(out)
