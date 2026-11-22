/**
 * Centralized API client for ObjectMemory Studio.
 * Interfaces directly with FastAPI backend routes.
 */

import type {
  ChangeEventSummary,
  ComparisonResponse,
  DatasetMetaResponse,
  DetectorListResponse,
  DiagnosticsResponse,
  EntityDetailResponse,
  EntitySummary,
  EvaluationRunRequest,
  EvaluationRunResponse,
  HealthResponse,
  ObservationSubmissionResult,
  SessionDetailResponse,
  SessionSummary,
  WorkspaceCreatePayload,
  WorkspaceExportResponse,
  WorkspaceResponse,
  WorkspaceSummary,
} from '../types/api';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail || `HTTP Error ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      let detail = `Server returned status ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) {
          detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        const text = await res.text();
        if (text) detail = text;
      }
      throw new ApiError(res.status, detail);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    const message = err instanceof Error ? err.message : 'Unknown network error';
    throw new ApiError(0, `Network failure: ${message}. Ensure the FastAPI server is running at http://127.0.0.1:8000.`);
  }
}

export const api = {
  // System Health
  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>('/api/health');
  },

  // Workspaces
  async listWorkspaces(): Promise<WorkspaceSummary[]> {
    return request<WorkspaceSummary[]>('/api/workspaces');
  },

  async createWorkspace(payload: WorkspaceCreatePayload): Promise<WorkspaceResponse> {
    return request<WorkspaceResponse>('/api/workspaces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  async getWorkspace(workspaceId: string): Promise<WorkspaceResponse> {
    return request<WorkspaceResponse>(`/api/workspaces/${encodeURIComponent(workspaceId)}`);
  },

  async deleteWorkspace(workspaceId: string): Promise<{ status: string; workspace_id: string }> {
    return request<{ status: string; workspace_id: string }>(`/api/workspaces/${encodeURIComponent(workspaceId)}`, {
      method: 'DELETE',
    });
  },

  // Detectors
  async getDetectors(): Promise<DetectorListResponse> {
    return request<DetectorListResponse>('/api/detectors');
  },

  // Observations & Ingest
  async submitObservation(
    workspaceId: string,
    file: File,
    sessionName?: string,
    timestamp?: string,
    notes?: string,
    confidenceThreshold: number = 0.25,
    detectorName?: string,
    prompts?: string,
  ): Promise<ObservationSubmissionResult> {
    const formData = new FormData();
    formData.append('image', file);
    if (sessionName && sessionName.trim()) {
      formData.append('session_name', sessionName.trim());
    }
    if (timestamp && timestamp.trim()) {
      formData.append('timestamp', timestamp.trim());
    }
    if (notes && notes.trim()) {
      formData.append('notes', notes.trim());
    }
    formData.append('confidence_threshold', String(confidenceThreshold));
    if (detectorName && detectorName.trim()) {
      formData.append('detector_name', detectorName.trim());
    }
    if (prompts && prompts.trim()) {
      formData.append('prompts', prompts.trim());
    }

    return request<ObservationSubmissionResult>(
      `/api/workspaces/${encodeURIComponent(workspaceId)}/observations`,
      {
        method: 'POST',
        body: formData,
      }
    );
  },

  async submitObservationBatch(
    workspaceId: string,
    files: File[],
    confidenceThreshold: number = 0.25,
    detectorName?: string,
    prompts?: string,
  ): Promise<ObservationSubmissionResult[]> {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    formData.append('confidence_threshold', String(confidenceThreshold));
    if (detectorName && detectorName.trim()) {
      formData.append('detector_name', detectorName.trim());
    }
    if (prompts && prompts.trim()) {
      formData.append('prompts', prompts.trim());
    }

    return request<ObservationSubmissionResult[]>(
      `/api/workspaces/${encodeURIComponent(workspaceId)}/observations/batch`,
      {
        method: 'POST',
        body: formData,
      }
    );
  },

  async exportWorkspace(workspaceId: string): Promise<WorkspaceExportResponse> {
    return request<WorkspaceExportResponse>(`/api/workspaces/${encodeURIComponent(workspaceId)}/export`);
  },

  getWorkspaceExportUrl(workspaceId: string): string {
    return `${API_BASE_URL}/api/workspaces/${encodeURIComponent(workspaceId)}/export`;
  },

  async listObservations(workspaceId: string): Promise<SessionSummary[]> {
    return request<SessionSummary[]>(`/api/workspaces/${encodeURIComponent(workspaceId)}/observations`);
  },

  async getObservationDetail(workspaceId: string, sessionId: string): Promise<SessionDetailResponse> {
    return request<SessionDetailResponse>(
      `/api/workspaces/${encodeURIComponent(workspaceId)}/observations/${encodeURIComponent(sessionId)}`
    );
  },

  // Entities
  async listEntities(workspaceId: string, state?: string): Promise<EntitySummary[]> {
    const query = state ? `?state=${encodeURIComponent(state)}` : '';
    return request<EntitySummary[]>(`/api/workspaces/${encodeURIComponent(workspaceId)}/entities${query}`);
  },

  async getEntityDetail(workspaceId: string, entityId: string): Promise<EntityDetailResponse> {
    return request<EntityDetailResponse>(
      `/api/workspaces/${encodeURIComponent(workspaceId)}/entities/${encodeURIComponent(entityId)}`
    );
  },

  // Change Events
  async listChangeEvents(workspaceId: string): Promise<ChangeEventSummary[]> {
    return request<ChangeEventSummary[]>(`/api/workspaces/${encodeURIComponent(workspaceId)}/events`);
  },

  // Comparison
  async compareSessions(
    workspaceId: string,
    referenceSessionId?: string,
    querySessionId?: string,
  ): Promise<ComparisonResponse> {
    const params = new URLSearchParams();
    if (referenceSessionId) params.append('reference_session_id', referenceSessionId);
    if (querySessionId) params.append('query_session_id', querySessionId);
    const query = params.toString() ? `?${params.toString()}` : '';
    return request<ComparisonResponse>(`/api/workspaces/${encodeURIComponent(workspaceId)}/compare${query}`);
  },

  // Diagnostics
  async getDiagnostics(workspaceId: string): Promise<DiagnosticsResponse> {
    return request<DiagnosticsResponse>(`/api/workspaces/${encodeURIComponent(workspaceId)}/diagnostics`);
  },

  // Image URL helper
  getImageUrl(workspaceId: string, filename: string): string {
    return `${API_BASE_URL}/api/workspaces/${encodeURIComponent(workspaceId)}/images/${encodeURIComponent(filename)}`;
  },

  // Evaluation Lab
  async listDatasets(): Promise<DatasetMetaResponse[]> {
    return request<DatasetMetaResponse[]>('/api/evaluation/datasets');
  },

  async runEvaluation(payload: EvaluationRunRequest): Promise<EvaluationRunResponse> {
    return request<EvaluationRunResponse>('/api/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  async listEvaluationRuns(): Promise<EvaluationRunResponse[]> {
    return request<EvaluationRunResponse[]>('/api/evaluation/runs');
  },

  getEvaluationExportUrl(runId: string, format: 'json' | 'csv' = 'json'): string {
    return `${API_BASE_URL}/api/evaluation/runs/${encodeURIComponent(runId)}/export?format=${format}`;
  },
};
