"""
FastAPI application router and service endpoints for ObjectMemory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from objectmemory.api.schemas import (
    ChangeEventSummary,
    ComparisonResponse,
    ConfirmLocationRequest,
    ConfirmLocationResponse,
    DetectorInfoSchema,
    DetectorListResponse,
    DiagnosticsResponse,
    EntityDetailResponse,
    EntityLabelUpdate,
    EntitySummary,
    ObservationSubmissionResult,
    SessionDetailResponse,
    SessionSummary,
    WorkspaceCreate,
    WorkspaceExportResponse,
    WorkspaceResponse,
    WorkspaceSummary,
)
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.api.workspace_manager import WorkspaceManager
from objectmemory.api.evaluation import router as evaluation_router
from objectmemory.api.eth_registration import router as eth_router

logger = logging.getLogger(__name__)

# Max upload limit: 25 MB
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def create_app(manager: Optional[WorkspaceManager] = None) -> FastAPI:
    """
    Factory creating the FastAPI instance. Allows dependency injection of WorkspaceManager for testing.
    """
    if manager is None:
        manager = WorkspaceManager(base_dir=Path("data/workspaces"))

    app = FastAPI(
        title="ObjectMemory API",
        description=(
            "Long-Term Visual Memory for Detecting Persistent Environmental Changes. "
            "Provides workspace isolation, multi-visit temporal memory, conservative "
            "cross-class object association, and auditable change events."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Evaluation Router & Supplementary ETH Registration Router
    app.include_router(evaluation_router)
    app.include_router(eth_router)

    # Attach manager to app state
    app.state.manager = manager

    # -----------------------------------------------------------------------
    # System Health
    # -----------------------------------------------------------------------
    @app.get("/api/health", tags=["System"])
    def get_health() -> Dict[str, Any]:
        mgr: WorkspaceManager = app.state.manager
        detector_ok = False
        detector_info = {}
        try:
            det = mgr._get_detector()
            detector_ok = det.is_available()
            detector_info = det.model_info()
        except Exception as exc:
            detector_info = {"error": str(exc)}

        workspaces = mgr.list_workspaces()
        return {
            "status": "HEALTHY",
            "detector_available": detector_ok,
            "detector_info": detector_info,
            "workspaces_count": len(workspaces),
            "storage_root": str(mgr.base_dir),
        }

    # -----------------------------------------------------------------------
    # Detectors & Models
    # -----------------------------------------------------------------------
    @app.get("/api/detectors", response_model=DetectorListResponse, tags=["Detectors"])
    def get_detectors() -> DetectorListResponse:
        """Returns the list of registered object detectors, their availability, and open-vocabulary capabilities."""
        items = DetectorRegistry.list_available()
        return DetectorListResponse(
            default_detector=DetectorRegistry.get_default_name(),
            detectors=[DetectorInfoSchema(**item) for item in items],
        )

    # -----------------------------------------------------------------------
    # Workspaces
    # -----------------------------------------------------------------------
    @app.post(
        "/api/workspaces",
        response_model=WorkspaceResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Workspaces"],
    )
    def create_workspace(payload: WorkspaceCreate) -> WorkspaceResponse:
        mgr: WorkspaceManager = app.state.manager
        try:
            meta = mgr.create_workspace(
                workspace_id=payload.workspace_id,
                name=payload.name,
                description=payload.description,
            )
            summary_dict = mgr.get_workspace_summary(payload.workspace_id)
            if not summary_dict:
                raise HTTPException(status_code=500, detail="Failed to retrieve created workspace.")
            return WorkspaceResponse(
                workspace=WorkspaceSummary(**summary_dict),
                entity_state_counts=summary_dict.get("entity_state_counts", {}),
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @app.get("/api/workspaces", response_model=List[WorkspaceSummary], tags=["Workspaces"])
    def list_workspaces() -> List[WorkspaceSummary]:
        mgr: WorkspaceManager = app.state.manager
        summaries = mgr.list_workspaces()
        return [WorkspaceSummary(**s) for s in summaries]

    @app.get("/api/workspaces/{workspace_id}", response_model=WorkspaceResponse, tags=["Workspaces"])
    def get_workspace(workspace_id: str) -> WorkspaceResponse:
        mgr: WorkspaceManager = app.state.manager
        summary_dict = mgr.get_workspace_summary(workspace_id)
        if not summary_dict:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' not found.",
            )
        return WorkspaceResponse(
            workspace=WorkspaceSummary(**summary_dict),
            entity_state_counts=summary_dict.get("entity_state_counts", {}),
        )

    @app.delete("/api/workspaces/{workspace_id}", tags=["Workspaces"])
    def delete_workspace(workspace_id: str) -> Dict[str, Any]:
        mgr: WorkspaceManager = app.state.manager
        deleted = mgr.delete_workspace(workspace_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' not found.",
            )
        return {"status": "DELETED", "workspace_id": workspace_id}

    @app.get(
        "/api/workspaces/{workspace_id}/export",
        response_model=WorkspaceExportResponse,
        tags=["Workspaces"],
    )
    def export_workspace(workspace_id: str) -> WorkspaceExportResponse:
        """
        Exports the entire workspace experimental timeline, including all visits,
        entity life cycles, change events, and system diagnostics as structured JSON.
        """
        mgr: WorkspaceManager = app.state.manager
        try:
            res = mgr.export_workspace(workspace_id)
            return WorkspaceExportResponse(**res)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        except Exception as exc:
            logger.exception("Workspace export error: %s", exc)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # -----------------------------------------------------------------------
    # Observation Submission & Processing
    # -----------------------------------------------------------------------
    @app.post(
        "/api/workspaces/{workspace_id}/observations",
        response_model=ObservationSubmissionResult,
        status_code=status.HTTP_201_CREATED,
        tags=["Observations"],
    )
    async def submit_observation(
        workspace_id: str,
        image: UploadFile = File(..., description="Image file (PNG or JPEG)"),
        session_name: Optional[str] = Form(None),
        timestamp: Optional[str] = Form(None),
        notes: Optional[str] = Form(None),
        confidence_threshold: float = Form(0.25),
        detector_name: Optional[str] = Form(None),
        prompts: Optional[str] = Form(None, description="Optional comma-separated open-vocabulary category prompts"),
    ) -> ObservationSubmissionResult:
        mgr: WorkspaceManager = app.state.manager

        # Check existence of workspace first
        if not mgr.get_workspace_meta(workspace_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' not found.",
            )

        # Validate file size
        contents = await image.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        parsed_classes: Optional[List[str]] = None
        if prompts:
            parsed_classes = [c.strip() for c in prompts.split(",") if c.strip()]

        try:
            res_dict = mgr.process_image_observation(
                workspace_id=workspace_id,
                image_bytes=contents,
                session_name=session_name,
                timestamp=timestamp,
                notes=notes,
                confidence_threshold=confidence_threshold,
                detector_name=detector_name,
                target_classes=parsed_classes,
            )
            return ObservationSubmissionResult(**res_dict)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except Exception as exc:
            logger.exception("Observation processing error: %s", exc)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @app.post(
        "/api/workspaces/{workspace_id}/observations/batch",
        response_model=List[ObservationSubmissionResult],
        status_code=status.HTTP_201_CREATED,
        tags=["Observations"],
    )
    async def submit_observation_batch(
        workspace_id: str,
        files: List[UploadFile] = File(..., description="Chronological sequence of images"),
        confidence_threshold: float = Form(0.25),
        detector_name: Optional[str] = Form(None),
        prompts: Optional[str] = Form(None),
    ) -> List[ObservationSubmissionResult]:
        """
        Submits and processes a sequence of image visits chronologically in natural filename order.
        Maintains temporal continuity across sequential physical observations.
        """
        mgr: WorkspaceManager = app.state.manager
        if not mgr.get_workspace_meta(workspace_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' not found.",
            )

        parsed_classes: Optional[List[str]] = None
        if prompts:
            parsed_classes = [c.strip() for c in prompts.split(",") if c.strip()]

        file_payloads: List[Tuple[str, bytes]] = []
        for f in files:
            contents = await f.read()
            if len(contents) > 0 and len(contents) <= MAX_UPLOAD_BYTES:
                file_payloads.append((f.filename or "visit", contents))

        if not file_payloads:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid non-empty images provided in batch.",
            )

        try:
            results = mgr.process_image_sequence(
                workspace_id=workspace_id,
                files=file_payloads,
                confidence_threshold=confidence_threshold,
                detector_name=detector_name,
                prompts=parsed_classes,
            )
            return [ObservationSubmissionResult(**r) for r in results]
        except Exception as exc:
            logger.exception("Batch observation processing error: %s", exc)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # -----------------------------------------------------------------------
    # Sessions & History
    # -----------------------------------------------------------------------
    @app.get(
        "/api/workspaces/{workspace_id}/observations",
        response_model=List[SessionSummary],
        tags=["Observations"],
    )
    def list_observations(workspace_id: str) -> List[SessionSummary]:
        mgr: WorkspaceManager = app.state.manager
        try:
            sessions = mgr.list_sessions(workspace_id)
            return [SessionSummary(**s) for s in sessions]
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    @app.get(
        "/api/workspaces/{workspace_id}/observations/{session_id}",
        response_model=SessionDetailResponse,
        tags=["Observations"],
    )
    def get_observation_detail(workspace_id: str, session_id: str) -> SessionDetailResponse:
        mgr: WorkspaceManager = app.state.manager
        res = mgr.get_session_detail(workspace_id, session_id)
        if not res:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{session_id}' not found in workspace '{workspace_id}'.",
            )
        return SessionDetailResponse(**res)

    # -----------------------------------------------------------------------
    # Entities
    # -----------------------------------------------------------------------
    @app.get(
        "/api/workspaces/{workspace_id}/entities",
        response_model=List[EntitySummary],
        tags=["Entities"],
    )
    def list_entities(
        workspace_id: str,
        state: Optional[str] = Query(None, description="Filter by EpistemicState (e.g. CONFIRMED_PRESENT, CONFIRMED_MOVED)"),
    ) -> List[EntitySummary]:
        mgr: WorkspaceManager = app.state.manager
        try:
            entities = mgr.list_entities(workspace_id, state=state)
            return [EntitySummary(**e) for e in entities]
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    @app.patch(
        "/api/workspaces/{workspace_id}/entities/{entity_id}/label",
        response_model=EntitySummary,
        tags=["Entities"],
    )
    def confirm_entity_label(
        workspace_id: str,
        entity_id: str,
        payload: EntityLabelUpdate,
    ) -> EntitySummary:
        mgr: WorkspaceManager = app.state.manager
        try:
            res = mgr.update_entity_label(workspace_id, entity_id, payload.label)
            return EntitySummary(**res)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @app.delete(
        "/api/workspaces/{workspace_id}/entities/{entity_id}/label",
        response_model=EntitySummary,
        tags=["Entities"],
    )
    def clear_entity_label(
        workspace_id: str,
        entity_id: str,
    ) -> EntitySummary:
        mgr: WorkspaceManager = app.state.manager
        try:
            res = mgr.clear_entity_label(workspace_id, entity_id)
            return EntitySummary(**res)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    @app.post(
        "/api/workspaces/{workspace_id}/observations/{session_id}/confirm-location",
        response_model=ConfirmLocationResponse,
        tags=["Observations"],
    )
    def confirm_entity_location(
        workspace_id: str,
        session_id: str,
        payload: ConfirmLocationRequest,
    ) -> ConfirmLocationResponse:
        mgr: WorkspaceManager = app.state.manager
        try:
            res = mgr.confirm_entity_location(
                workspace_id=workspace_id,
                session_id=session_id,
                entity_id=payload.entity_id,
                bbox_dict=payload.bbox.model_dump(),
                notes=payload.notes,
            )
            return ConfirmLocationResponse(**res)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @app.get(
        "/api/workspaces/{workspace_id}/entities/{entity_id}",
        response_model=EntityDetailResponse,
        tags=["Entities"],
    )
    def get_entity(workspace_id: str, entity_id: str) -> EntityDetailResponse:
        mgr: WorkspaceManager = app.state.manager
        res = mgr.get_entity_detail(workspace_id, entity_id)
        if not res:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity '{entity_id}' not found in workspace '{workspace_id}'.",
            )
        return EntityDetailResponse(**res)

    # -----------------------------------------------------------------------
    # Change Events
    # -----------------------------------------------------------------------
    @app.get(
        "/api/workspaces/{workspace_id}/events",
        response_model=List[ChangeEventSummary],
        tags=["Events"],
    )
    def list_change_events(workspace_id: str) -> List[ChangeEventSummary]:
        mgr: WorkspaceManager = app.state.manager
        try:
            events = mgr.list_change_events(workspace_id)
            return [ChangeEventSummary(**e) for e in events]
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # -----------------------------------------------------------------------
    # Comparison & Diagnostics
    # -----------------------------------------------------------------------
    @app.get(
        "/api/workspaces/{workspace_id}/compare",
        response_model=ComparisonResponse,
        tags=["Analysis"],
    )
    def compare_observations(
        workspace_id: str,
        reference_session_id: Optional[str] = Query(None, description="Baseline session ID (defaults to first visit)"),
        query_session_id: Optional[str] = Query(None, description="Query session ID (defaults to latest visit)"),
    ) -> ComparisonResponse:
        mgr: WorkspaceManager = app.state.manager
        try:
            res = mgr.compare_sessions(
                workspace_id=workspace_id,
                ref_session_id=reference_session_id,
                query_session_id=query_session_id,
            )
            return ComparisonResponse(**res)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @app.get(
        "/api/workspaces/{workspace_id}/diagnostics",
        response_model=DiagnosticsResponse,
        tags=["Analysis"],
    )
    def get_diagnostics(workspace_id: str) -> DiagnosticsResponse:
        mgr: WorkspaceManager = app.state.manager
        try:
            res = mgr.get_diagnostics(workspace_id)
            return DiagnosticsResponse(**res)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # -----------------------------------------------------------------------
    # Image Serving
    # -----------------------------------------------------------------------
    @app.get("/api/workspaces/{workspace_id}/images/{filename}", tags=["Images"])
    def get_workspace_image(workspace_id: str, filename: str) -> FileResponse:
        mgr: WorkspaceManager = app.state.manager
        fpath = mgr.get_image_file_path(workspace_id, filename)
        if not fpath:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
        media_type = "image/png" if fpath.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(str(fpath), media_type=media_type)

    # -----------------------------------------------------------------------
    # Frontend SPA Serving (if built in frontend/dist)
    # -----------------------------------------------------------------------
    dist_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

    return app


# Default ASGI application instance for uvicorn
app = create_app()
