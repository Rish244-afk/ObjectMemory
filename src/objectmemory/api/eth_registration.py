"""
Supplementary ETH ASL Change Detection (Lounge) Registration Evaluation API.

Scientific Safeguards:
- Strictly scoped as a supplementary 2D planar registration evaluation.
- Does NOT compute object-lifecycle metrics (no 2D bounding boxes or object IDs exist).
- Uses the frozen production PlanarRegistrar (AlignmentConfig defaults) unchanged.
- Never fabricates camera movement or missing registration metrics.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from objectmemory.registration.registrar import PlanarRegistrar

logger = logging.getLogger(__name__)

_DEFAULT_ETH_ROOT = Path("data/eth_asl_lounge")
_DEFAULT_HISTORICAL_RESULTS = Path("research_validation/results/eth_asl_lounge/eth_asl_lounge_evaluation_results.json")

_SESSION_RE = re.compile(r"^observation_\d{2}$")
_FRAME_RE = re.compile(r"^frame_\d+\.(png|jpg|jpeg)$", re.IGNORECASE)


class EthRegisterRequest(BaseModel):
    ref_session_id: str = Field(..., description="Reference ETH session ID (e.g., observation_01)")
    ref_frame: str = Field(..., description="Reference frame filename (e.g., frame_00000.png)")
    query_session_id: str = Field(..., description="Query ETH session ID (e.g., observation_01)")
    query_frame: str = Field(..., description="Query frame filename (e.g., frame_00001.png)")


def _validate_session_id(session_id: str) -> str:
    cleaned = session_id.strip()
    if not _SESSION_RE.match(cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ETH session ID '{session_id}'. Expected format 'observation_XX'.",
        )
    return cleaned


def _validate_frame_name(filename: str) -> str:
    cleaned = filename.strip()
    if not _FRAME_RE.match(cleaned) or ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid frame filename '{filename}'.",
        )
    return cleaned


def _encode_image_data_url(img_bgr: Any) -> Optional[str]:
    if img_bgr is None:
        return None
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def create_eth_router(
    dataset_root: Optional[Path] = None,
    historical_results_path: Optional[Path] = None,
) -> APIRouter:
    """Creates the ETH ASL Lounge supplementary registration router."""
    root = (dataset_root or _DEFAULT_ETH_ROOT).resolve()
    hist_path = (historical_results_path or _DEFAULT_HISTORICAL_RESULTS).resolve()
    router = APIRouter(prefix="/api/eth-lounge", tags=["ETH ASL Supplementary"])
    registrar = PlanarRegistrar()

    @router.get("/sessions")
    def list_eth_sessions() -> Dict[str, Any]:
        extracted_dir = root / "extracted"
        mesh_dir = root / "raw" / "complete_mesh"
        sessions: List[Dict[str, Any]] = []

        if extracted_dir.exists() and extracted_dir.is_dir():
            for entry in sorted(extracted_dir.iterdir()):
                if not entry.is_dir() or not _SESSION_RE.match(entry.name):
                    continue
                rgb_dir = entry / "rgb"
                frame_count = 0
                if rgb_dir.exists():
                    frame_count = sum(1 for f in rgb_dir.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"))
                # observation_01 -> observation_0.ply
                obs_idx = int(entry.name.split("_")[1]) - 1
                mesh_file = mesh_dir / f"observation_{obs_idx}.ply"
                has_poses = (entry / "metadata" / "frames_and_poses.json").exists()
                sessions.append({
                    "session_id": entry.name,
                    "frame_count": frame_count,
                    "has_mesh": mesh_file.exists(),
                    "mesh_filename": mesh_file.name if mesh_file.exists() else None,
                    "has_camera_poses": has_poses,
                })

        return {
            "dataset_name": "ETH ASL Change Detection (Lounge)",
            "evaluation_role": "SUPPLEMENTARY_REGISTRATION_ONLY",
            "available": len(sessions) > 0,
            "scope_notice": (
                "Supplementary 2D planar registration evaluation only. ETH ASL Lounge provides 3D meshes and "
                "camera poses without 2D object bounding boxes or object-lifecycle ground truth. Object-lifecycle "
                "metrics (state accuracy, removal recall, identity consistency) are NOT evaluated on this dataset."
            ),
            "sessions": sessions,
        }

    @router.get("/sessions/{session_id}/frames")
    def list_session_frames(
        session_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
        stride: int = Query(1, ge=1, le=500),
    ) -> Dict[str, Any]:
        sid = _validate_session_id(session_id)
        rgb_dir = root / "extracted" / sid / "rgb"
        if not rgb_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ETH session '{sid}' RGB directory not found.",
            )

        all_files = sorted(
            f.name for f in rgb_dir.iterdir()
            if f.is_file() and _FRAME_RE.match(f.name)
        )
        strided = [(idx, name) for idx, name in enumerate(all_files) if idx % stride == 0]
        sliced = strided[offset: offset + limit]

        return {
            "session_id": sid,
            "total_frames": len(all_files),
            "strided_total": len(strided),
            "offset": offset,
            "limit": limit,
            "stride": stride,
            "frames": [{"index": idx, "filename": name} for idx, name in sliced],
        }

    @router.get("/sessions/{session_id}/frames/{filename}")
    def get_session_frame(session_id: str, filename: str) -> FileResponse:
        sid = _validate_session_id(session_id)
        fname = _validate_frame_name(filename)
        fpath = (root / "extracted" / sid / "rgb" / fname).resolve()
        expected_parent = (root / "extracted" / sid / "rgb").resolve()
        if fpath.parent != expected_parent or not fpath.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Frame '{fname}' not found in session '{sid}'.",
            )
        media_type = "image/png" if fpath.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(str(fpath), media_type=media_type)

    @router.post("/register")
    def register_eth_pair(req: EthRegisterRequest) -> Dict[str, Any]:
        ref_sid = _validate_session_id(req.ref_session_id)
        ref_fname = _validate_frame_name(req.ref_frame)
        q_sid = _validate_session_id(req.query_session_id)
        q_fname = _validate_frame_name(req.query_frame)

        ref_path = (root / "extracted" / ref_sid / "rgb" / ref_fname).resolve()
        q_path = (root / "extracted" / q_sid / "rgb" / q_fname).resolve()
        if not ref_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Reference frame not found: {ref_sid}/{ref_fname}")
        if not q_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Query frame not found: {q_sid}/{q_fname}")

        ref_img = cv2.imread(str(ref_path))
        q_img = cv2.imread(str(q_path))
        if ref_img is None or q_img is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Failed to decode one or both ETH RGB frames.")

        reg = registrar.register(ref_img, q_img)

        unreliable_warning: Optional[str] = None
        if not reg.success:
            unreliable_warning = (
                f"Registration failed ({reg.status.value}): "
                f"{reg.failure_reason or 'Planar homography could not be reliably verified for this 3D viewpoint pair.'}"
            )
        elif reg.inlier_ratio is not None and reg.inlier_ratio < registrar.config.min_inlier_ratio:
            unreliable_warning = (
                f"Inlier ratio ({reg.inlier_ratio:.3f}) is below threshold ({registrar.config.min_inlier_ratio:.2f})."
            )

        reproj = reg.reprojection_stats
        return {
            "ref_session_id": ref_sid,
            "ref_frame": ref_fname,
            "query_session_id": q_sid,
            "query_frame": q_fname,
            "success": bool(reg.success),
            "status": reg.status.value,
            "method_used": reg.method_used,
            "ref_size": list(reg.ref_size),
            "query_size": list(reg.query_size),
            "num_keypoints_ref": reg.num_keypoints_ref,
            "num_keypoints_query": reg.num_keypoints_query,
            "num_tentative_matches": reg.num_tentative_matches,
            "num_inliers": reg.num_inliers,
            "inlier_ratio": round(float(reg.inlier_ratio), 4) if reg.inlier_ratio is not None else None,
            "mean_reprojection_error": round(float(reproj.mean), 4) if reproj is not None else None,
            "median_reprojection_error": round(float(reproj.median), 4) if reproj is not None else None,
            "max_reprojection_error": round(float(reproj.max_error), 4) if reproj is not None else None,
            "overlap_area_fraction": round(float(reg.overlap_area_fraction), 4) if reg.overlap_area_fraction is not None else None,
            "latency_ms": round(float(reg.latency_ms), 2),
            "failure_reason": reg.failure_reason,
            "unreliable_warning": unreliable_warning,
            "aligned_image_data_url": _encode_image_data_url(reg.aligned_image),
            "ref_image_url": f"/api/eth-lounge/sessions/{ref_sid}/frames/{ref_fname}",
            "query_image_url": f"/api/eth-lounge/sessions/{q_sid}/frames/{q_fname}",
        }

    @router.get("/historical-summary")
    def get_historical_summary() -> Dict[str, Any]:
        if not hist_path.exists():
            return {"available": False, "experiments": {}}
        try:
            raw = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to read historical summary: {exc}")

        experiments: Dict[str, Any] = {}
        for key in ("smoke_test", "intra_observation_01", "cross_observation_first_frames", "cross_observation_pose_matched"):
            if key in raw and isinstance(raw[key], dict):
                v = raw[key]
                experiments[key] = {
                    "frames_evaluated": v.get("frames_evaluated"),
                    "transitions_attempted": v.get("transitions_attempted"),
                    "transitions_succeeded": v.get("transitions_succeeded"),
                    "registration_success_rate": v.get("registration_success_rate"),
                    "registration_status_counts": v.get("registration_status_counts", {}),
                    "mean_inlier_ratio_when_success": v.get("mean_inlier_ratio_when_success"),
                    "mean_reproj_error_px_when_success": v.get("mean_reproj_error_px_when_success"),
                    "unavailable_metrics": v.get("unavailable_metrics", []),
                    "unavailable_reason": v.get("unavailable_reason"),
                }
        return {"available": True, "experiments": experiments}

    return router


router = create_eth_router()
