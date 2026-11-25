/**
 * TypeScript data contracts matching ObjectMemory FastAPI backend schemas.
 */

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface WorkspaceSummary {
  workspace_id: string;
  name: string;
  description?: string | null;
  created_at: string;
  total_sessions: number;
  total_entities: number;
  total_events: number;
  reference_session_id?: string | null;
  last_session_timestamp?: string | null;
}

export interface WorkspaceResponse {
  workspace: WorkspaceSummary;
  entity_state_counts: Record<string, number>;
}

export interface WorkspaceExportResponse {
  workspace: WorkspaceSummary;
  entity_state_counts: Record<string, number>;
  sessions: SessionDetailResponse[];
  entities: EntityDetailResponse[];
  events: ChangeEventSummary[];
  diagnostics?: DiagnosticsResponse | null;
  exported_at: string;
}

export interface WorkspaceCreatePayload {
  workspace_id: string;
  name: string;
  description?: string;
}

export interface RegistrationSummary {
  success: boolean;
  status: string;
  method_used: string;
  inlier_ratio: number;
  mean_reprojection_error?: number | null;
  inliers_count: number;
  ref_size: [number, number];
  query_size: [number, number];
  failure_reason?: string | null;
  selected_keyframe_id?: string | null;
  is_direct_to_canonical?: boolean;
  spatial_coverage?: number | null;
  decision_status?: string | null;
  decision_reason?: string | null;
  is_promoted?: boolean;
}

export interface DetectionSummary {
  label: string;
  confidence: number;
  raw_bbox: BoundingBox;
  canonical_bbox?: BoundingBox | null;
  is_clipped: boolean;
}

export interface EntityBeliefUpdateSummary {
  entity_id: string;
  label: string;
  prior_state: string;
  new_state: string;
  transition_occurred: boolean;
  visibility_status: string;
  explanation: string;
  event_id?: string | null;
  displacement_px?: number | null;
}

export interface ChangeEventSummary {
  event_id: string;
  entity_id: string;
  from_session_id?: string | null;
  to_session_id: string;
  timestamp: string;
  prior_state: string;
  new_state: string;
  source: string;
  displacement_distance?: number | null;
  explanation: string;
  supporting_evidence: Record<string, any>;
}

export interface ObservationSubmissionResult {
  session_id: string;
  session_name: string;
  timestamp: string;
  image_url: string;
  is_reference_frame: boolean;
  registration: RegistrationSummary;
  detections: DetectionSummary[];
  belief_updates: EntityBeliefUpdateSummary[];
  events: ChangeEventSummary[];
  total_active_entities: number;
  latency_breakdown_ms: Record<string, number>;
  total_latency_ms: number;
}

export interface ObservationItem {
  observation_id: string;
  session_id: string;
  entity_id?: string | null;
  label?: string | null;
  timestamp: string;
  image_reference: string;
  raw_bbox?: BoundingBox | null;
  canonical_bbox?: BoundingBox | null;
  visibility_status: string;
  detector_confidence?: number | null;
  identity_match_score?: number | null;
  alignment_quality?: number | null;
  raw_evidence?: Record<string, any> | null;
}

export interface RawDetectionItem {
  detection_id: string;
  session_id: string;
  timestamp: string;
  label: string;
  confidence: number;
  raw_bbox: BoundingBox;
  canonical_bbox?: BoundingBox | null;
  registration_status: string;
  metadata?: Record<string, any> | null;
}

export interface SessionSummary {
  session_id: string;
  session_name: string;
  timestamp: string;
  image_path: string;
  image_url: string;
  notes?: string | null;
  observations_count: number;
  events_count: number;
  raw_detections_count?: number;
}

export interface SessionDetailResponse {
  session: SessionSummary;
  observations: ObservationItem[];
  events: ChangeEventSummary[];
  raw_detections?: RawDetectionItem[];
  registration?: RegistrationSummary | null;
}

export interface EntitySummary {
  entity_id: string;
  label: string;
  raw_label?: string | null;
  raw_confidence?: number | null;
  confirmed_label?: string | null;
  label_provenance?: string;
  label_confirmed_at?: string | null;
  observed_labels?: string[];
  class_uncertain?: boolean;
  created_at: string;
  current_state: string;
  canonical_bbox?: BoundingBox | null;
  last_seen_session_id?: string | null;
  metadata?: Record<string, any> | null;
}

export interface EntityDetailResponse {
  entity: EntitySummary;
  observations: ObservationItem[];
  change_events: ChangeEventSummary[];
}

export interface ComparisonDiffItem {
  entity_id: string;
  label: string;
  raw_label?: string | null;
  confirmed_label?: string | null;
  label_provenance?: string;
  evidence_source?: string;
  observed_labels?: string[];
  class_uncertain?: boolean;
  status_change: string;
  movement_status?: string;
  prior_state: string;
  new_state: string;
  displacement_px?: number | null;
  ref_confidence?: number | null;
  query_confidence?: number | null;
  ref_bbox?: BoundingBox | null;
  query_bbox?: BoundingBox | null;
  ref_raw_bbox?: BoundingBox | null;
  query_raw_bbox?: BoundingBox | null;
  explanation: string;
}

export interface ConfirmLocationRequest {
  entity_id: string;
  bbox: BoundingBox;
  notes?: string;
}

export interface ConfirmLocationResponse {
  workspace_id: string;
  session_id: string;
  entity_id: string;
  evidence_source: string;
  registration_reliable: boolean;
  prior_state: string;
  new_state: string;
  displacement_px?: number | null;
  canonical_bbox?: BoundingBox | null;
  raw_bbox: BoundingBox;
  explanation: string;
}

export interface ComparisonResponse {
  workspace_id: string;
  reference_session_id: string;
  query_session_id: string;
  registration_status: string;
  registration_inlier_ratio: number;
  ref_detections_count?: number;
  query_detections_count?: number;
  cross_sequence_warning?: string | null;
  items: ComparisonDiffItem[];
}

export interface DiagnosticsResponse {
  workspace_id: string;
  detector_info: Record<string, any>;
  total_sessions: number;
  total_entities: number;
  total_events: number;
  entity_state_distribution: Record<string, number>;
  registration_health: Record<string, any>;
}

export interface HealthResponse {
  status: string;
  detector_available: boolean;
  detector_info: Record<string, any>;
  workspaces_count: number;
  storage_root: string;
}

export interface DetectorInfo {
  name: string;
  is_default: boolean;
  is_available: boolean;
  availability_error?: string | null;
  is_open_vocabulary: boolean;
  is_synthetic: boolean;
  description: string;
  model_info: Record<string, any>;
}

export interface DetectorListResponse {
  default_detector: string;
  detectors: DetectorInfo[];
}

// ---------------------------------------------------------------------------
// Evaluation Lab Types
// ---------------------------------------------------------------------------

export interface DatasetMetaResponse {
  dataset_id: string;
  name: string;
  dataset_type: string;
  provenance: string;
  is_template: boolean;
  total_visits: number;
  total_annotations: number;
  description: string;
}

export interface EvaluationRunRequest {
  dataset_id: string;
  workspace_id?: string;
  confidence_threshold?: number;
}

export interface EvaluationMetrics {
  precision?: number | null;
  recall?: number | null;
  f1_score?: number | null;
  mean_iou?: number | null;
  registration_success_rate: number;
  mean_reprojection_error?: number | null;
  change_confirmation_delay_visits: number;
  false_movement_suppression_count: number;
  total_evaluations: number;
  correct_predictions: number;
}

export interface CPUProfileSummary {
  detector_name: string;
  num_visits: number;
  mean_detector_latency_ms: number;
  mean_registration_latency_ms: number;
  mean_pipeline_latency_ms: number;
  p95_latency_ms: number;
}

export interface EvaluationEventItem {
  visit_id: string;
  entity_id: string;
  entity_label: string;
  expected_state: string;
  predicted_state: string;
  is_match: boolean;
  displacement_px?: number | null;
  notes?: string | null;
}

export interface EvaluationRunResponse {
  run_id: string;
  timestamp: string;
  dataset_id: string;
  status: string;
  provenance_disclaimer: string;
  metrics: EvaluationMetrics;
  cpu_profile: CPUProfileSummary;
  events: EvaluationEventItem[];
}

// Supplementary ETH ASL Lounge Registration Evaluation
export interface EthSessionInfo {
  session_id: string;
  frame_count: number;
  has_mesh: boolean;
  mesh_filename?: string | null;
  has_camera_poses: boolean;
}

export interface EthSessionsResponse {
  dataset_name: string;
  evaluation_role: string;
  available: boolean;
  scope_notice: string;
  sessions: EthSessionInfo[];
}

export interface EthFrameItem {
  index: number;
  filename: string;
}

export interface EthFramesResponse {
  session_id: string;
  total_frames: number;
  strided_total: number;
  offset: number;
  limit: number;
  stride: number;
  frames: EthFrameItem[];
}

export interface EthRegisterRequest {
  ref_session_id: string;
  ref_frame: string;
  query_session_id: string;
  query_frame: string;
}

export interface EthRegisterResponse {
  ref_session_id: string;
  ref_frame: string;
  query_session_id: string;
  query_frame: string;
  success: boolean;
  status: string;
  method_used: string;
  ref_size: number[];
  query_size: number[];
  num_keypoints_ref: number;
  num_keypoints_query: number;
  num_tentative_matches: number;
  num_inliers: number;
  inlier_ratio?: number | null;
  mean_reprojection_error?: number | null;
  median_reprojection_error?: number | null;
  max_reprojection_error?: number | null;
  overlap_area_fraction?: number | null;
  latency_ms: number;
  failure_reason?: string | null;
  unreliable_warning?: string | null;
  aligned_image_data_url?: string | null;
  ref_image_url: string;
  query_image_url: string;
}

export interface EthHistoricalExperimentSummary {
  frames_evaluated: number;
  transitions_attempted: number;
  transitions_succeeded: number;
  registration_success_rate: number;
  registration_status_counts: Record<string, number>;
  mean_inlier_ratio_when_success?: number | null;
  mean_reproj_error_px_when_success?: number | null;
  unavailable_metrics: string[];
  unavailable_reason?: string | null;
}

export interface EthHistoricalSummaryResponse {
  available: boolean;
  experiments: Record<string, EthHistoricalExperimentSummary>;
}
