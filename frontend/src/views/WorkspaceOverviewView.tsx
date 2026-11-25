import React, { useState, useEffect, useRef } from 'react';
import type {
  WorkspaceSummary,
  SessionSummary,
  ObservationSubmissionResult,
  SessionDetailResponse,
  DetectorInfo,
} from '../types/api';
import { api } from '../api/client';
import { StatusBadge } from '../components/StatusBadge';
import { BoundingBoxCanvas } from '../components/BoundingBoxCanvas';
import type { OverlayBox } from '../components/BoundingBoxCanvas';
import {
  Upload,
  Camera,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Sparkles,
  RefreshCw,
  X,
  ChevronLeft,
  ChevronRight,
  Layers,
  Compass,
  Download,
  Video,
  Play,
  Pause,
  Files,
  Square,
} from 'lucide-react';

interface WorkspaceOverviewViewProps {
  workspace: WorkspaceSummary;
  onRefreshWorkspace: () => void;
  onNavigateToComparison: () => void;
}

type IngestMode = 'single' | 'batch' | 'webcam';

export const WorkspaceOverviewView: React.FC<WorkspaceOverviewViewProps> = ({
  workspace,
  onRefreshWorkspace,
  onNavigateToComparison,
}) => {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);

  // Active visit in hero viewer
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [selectedSessionDetail, setSelectedSessionDetail] = useState<SessionDetailResponse | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [highlightedEntityId, setHighlightedEntityId] = useState<string | null>(null);

  // Ingest form state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [sessionName, setSessionName] = useState('');
  const [timestamp, setTimestamp] = useState('');
  const [notes, setNotes] = useState('');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.25);
  const [detectors, setDetectors] = useState<DetectorInfo[]>([]);
  const [selectedDetector, setSelectedDetector] = useState<string>(() => {
    try {
      return localStorage.getItem('objectmemory_selected_detector') || 'yolov8m';
    } catch {
      return 'yolov8m';
    }
  });
  const handleSelectDetector = (val: string) => {
    setSelectedDetector(val);
    try {
      localStorage.setItem('objectmemory_selected_detector', val);
    } catch {
      // ignore storage errors
    }
    if (val.includes('world') && confidenceThreshold > 0.2) {
      setConfidenceThreshold(0.15);
    } else if (!val.includes('world') && confidenceThreshold < 0.2) {
      setConfidenceThreshold(0.25);
    }
  };
  const [customPrompts, setCustomPrompts] = useState<string>(
    'notebook, book, bottle, cup, scissors, drawing compass, pliers'
  );

  const [analyzing, setAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<ObservationSubmissionResult | null>(null);

  // Phase 15 Extensions: Multi-Mode Ingestion, Webcam Monitoring & Export
  const [ingestMode, setIngestMode] = useState<IngestMode>('single');
  const [exporting, setExporting] = useState(false);

  // Batch Sequence state
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);

  // Webcam Live Monitor state
  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamInterval, setWebcamInterval] = useState<number>(0); // 0 = manual, 5, 10, 30 sec
  const [webcamCountdown, setWebcamCountdown] = useState<number | null>(null);
  const [isAutoCapturing, setIsAutoCapturing] = useState(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [droppedWebcamFrames, setDroppedWebcamFrames] = useState<number>(0);
  const analyzingRef = useRef<boolean>(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const countdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchSessions = async (autoSelectLatest = false) => {
    setLoadingSessions(true);
    try {
      const data = await api.listObservations(workspace.workspace_id);
      setSessions(data);

      if (data.length > 0) {
        const targetId = autoSelectLatest
          ? data[data.length - 1].session_id
          : selectedSessionId && data.some((s) => s.session_id === selectedSessionId)
          ? selectedSessionId
          : data[data.length - 1].session_id;

        setSelectedSessionId(targetId);
      } else {
        setSelectedSessionId(null);
        setSelectedSessionDetail(null);
      }
    } catch (err: any) {
      console.error('Failed to load sessions:', err);
    } finally {
      setLoadingSessions(false);
    }
  };

  useEffect(() => {
    setSelectedSessionId(null);
    setSelectedSessionDetail(null);
    fetchSessions();
    api
      .getDetectors()
      .then((res) => {
        setDetectors(res.detectors);
        let saved: string | null = null;
        try {
          saved = localStorage.getItem('objectmemory_selected_detector');
        } catch {
          saved = null;
        }
        const savedValid = saved && res.detectors.some((d) => d.name === saved && d.is_available);
        if (savedValid && saved) {
          setSelectedDetector(saved);
        } else if (res.default_detector) {
          setSelectedDetector(res.default_detector);
        }
      })
      .catch((err) => console.error('Failed to load detectors:', err));
  }, [workspace.workspace_id]);

  // Load session detail whenever selectedSessionId changes
  useEffect(() => {
    if (!selectedSessionId) {
      setSelectedSessionDetail(null);
      return;
    }

    let isMounted = true;
    const loadDetail = async () => {
      setLoadingDetail(true);
      try {
        const det = await api.getObservationDetail(workspace.workspace_id, selectedSessionId);
        if (isMounted) {
          setSelectedSessionDetail(det);
        }
      } catch (err) {
        console.error('Failed to load session detail:', err);
      } finally {
        if (isMounted) setLoadingDetail(false);
      }
    };

    loadDetail();
    return () => {
      isMounted = false;
    };
  }, [workspace.workspace_id, selectedSessionId]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisError(null);
      if (!sessionName) {
        setSessionName(`Visit ${sessions.length + 1}`);
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && (file.type === 'image/png' || file.type === 'image/jpeg')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisError(null);
      if (!sessionName) {
        setSessionName(`Visit ${sessions.length + 1}`);
      }
    }
  };

  const handleClearSelected = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setAnalysisError(null);
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setAnalyzing(true);
    setAnalysisError(null);
    setLastResult(null);

    try {
      const result = await api.submitObservation(
        workspace.workspace_id,
        selectedFile,
        sessionName.trim() || undefined,
        timestamp.trim() || undefined,
        notes.trim() || undefined,
        confidenceThreshold,
        selectedDetector,
        selectedDetector.includes('world') ? customPrompts : undefined
      );
      setLastResult(result);
      handleClearSelected();
      setSessionName('');
      setTimestamp('');
      setNotes('');
      // Refresh and auto-select the newly analyzed visit
      await fetchSessions(true);
      onRefreshWorkspace();
    } catch (err: any) {
      setAnalysisError(err.detail || err.message || 'Scene analysis failed.');
    } finally {
      analyzingRef.current = false;
      setAnalyzing(false);
    }
  };

  // Cleanup webcam stream on unmount
  useEffect(() => {
    return () => {
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const handleExportWorkspace = async () => {
    setExporting(true);
    try {
      const data = await api.exportWorkspace(workspace.workspace_id);
      const jsonStr = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `workspace_${workspace.workspace_id}_experiment_export.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error('Export error:', err);
      alert(`Export error: ${err.message || 'Failed to export workspace.'}`);
    } finally {
      setExporting(false);
    }
  };

  const handleBatchFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      const sorted = files.sort((a, b) =>
        a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' })
      );
      setBatchFiles(sorted);
      setAnalysisError(null);
    }
  };

  const handleProcessBatch = async () => {
    if (batchFiles.length === 0) return;
    setAnalyzing(true);
    setAnalysisError(null);
    setBatchProgress({ current: 0, total: batchFiles.length });
    try {
      const results = await api.submitObservationBatch(
        workspace.workspace_id,
        batchFiles,
        confidenceThreshold,
        selectedDetector,
        selectedDetector.includes('world') ? customPrompts : undefined
      );
      if (results.length > 0) {
        setLastResult(results[results.length - 1]);
      }
      setBatchFiles([]);
      await fetchSessions(true);
      onRefreshWorkspace();
    } catch (err: any) {
      setAnalysisError(err.detail || err.message || 'Batch sequence processing failed.');
    } finally {
      setAnalyzing(false);
      setBatchProgress(null);
    }
  };

  const startWebcam = async () => {
    try {
      setWebcamError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setWebcamActive(true);
    } catch (err: any) {
      console.error('Webcam access error:', err);
      setWebcamError(err.message || 'Unable to access camera device. Check permissions.');
    }
  };

  const stopWebcam = () => {
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setWebcamActive(false);
    setIsAutoCapturing(false);
    setWebcamCountdown(null);
  };

  const captureFrameBlob = (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      if (!videoRef.current) return resolve(null);
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      const ctx = canvas.getContext('2d');
      if (!ctx) return resolve(null);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.92);
    });
  };

  const handleCaptureWebcamVisit = async () => {
    if (!webcamActive) return;
    if (analyzingRef.current) {
      setDroppedWebcamFrames((c) => c + 1);
      return;
    }
    analyzingRef.current = true;
    setAnalyzing(true);
    setAnalysisError(null);
    try {
      const blob = await captureFrameBlob();
      if (!blob) throw new Error('Could not capture frame from video stream.');
      const file = new File([blob], `webcam_visit_${sessions.length + 1}.jpg`, { type: 'image/jpeg' });
      const res = await api.submitObservation(
        workspace.workspace_id,
        file,
        `Live Visit ${sessions.length + 1}`,
        new Date().toISOString(),
        'Captured via live webcam monitoring',
        confidenceThreshold,
        selectedDetector,
        selectedDetector.includes('world') ? customPrompts : undefined
      );
      setLastResult(res);
      await fetchSessions(true);
      onRefreshWorkspace();
    } catch (err: any) {
      setAnalysisError(err.detail || err.message || 'Webcam analysis failed.');
    } finally {
      analyzingRef.current = false;
      setAnalyzing(false);
    }
  };

  // Auto-capture countdown loop
  useEffect(() => {
    if (!isAutoCapturing || webcamInterval <= 0) {
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      setWebcamCountdown(null);
      return;
    }

    setWebcamCountdown(webcamInterval);
    countdownTimerRef.current = setInterval(() => {
      setWebcamCountdown((prev) => {
        if (prev === null || prev <= 1) {
          handleCaptureWebcamVisit();
          return webcamInterval;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    };
  }, [isAutoCapturing, webcamInterval, webcamActive, sessions.length]);

  // Visit navigation helpers
  const currentSessionIndex = sessions.findIndex((s) => s.session_id === selectedSessionId);
  const handlePrevVisit = () => {
    if (currentSessionIndex > 0) {
      setSelectedSessionId(sessions[currentSessionIndex - 1].session_id);
      setHighlightedEntityId(null);
    }
  };

  const handleNextVisit = () => {
    if (currentSessionIndex < sessions.length - 1) {
      setSelectedSessionId(sessions[currentSessionIndex + 1].session_id);
      setHighlightedEntityId(null);
    }
  };

  // Prepare overlay boxes for the hero canvas
  const isUnregisteredSession = Boolean(
    selectedSessionDetail &&
    selectedSessionDetail.observations.length === 0 &&
    (selectedSessionDetail.raw_detections?.length || 0) > 0
  );

  const canvasBoxes: OverlayBox[] = selectedSessionDetail
    ? (isUnregisteredSession
        ? (selectedSessionDetail.raw_detections || []).map((d) => ({
            id: d.detection_id,
            label: `${d.label} [Unregistered]`,
            bbox: d.raw_bbox,
            confidence: d.confidence,
            status: d.registration_status,
            isClipped: false,
            strokeColor: '#F59E0B',
            fillColor: 'rgba(245, 158, 11, 0.12)',
            strokeDasharray: '6 3',
          }))
        : selectedSessionDetail.observations.map((o) => ({
            id: o.observation_id,
            label: o.label || o.entity_id || 'object',
            bbox: o.raw_bbox,
            confidence: o.detector_confidence,
            status: o.visibility_status,
            isClipped: false,
          })))
    : [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 animate-in fade-in duration-150">
      {/* Workspace Header & Telemetry */}
      <div className="bg-[#11141D] border border-[#232A3B] rounded-xl p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-[#F8FAFC] tracking-tight">{workspace.name}</h1>
              <span className="font-mono text-xs px-2.5 py-0.5 rounded bg-[#181D29] text-[#94A3B8] border border-[#232A3B]">
                {workspace.workspace_id}
              </span>
            </div>
            <p className="text-xs text-[#94A3B8] mt-1">
              {workspace.description || 'Monitored physical environment with long-term visual memory.'}
            </p>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <div className="bg-[#090B10] px-3 py-2 rounded-lg border border-[#232A3B]">
              <span className="text-[#64748B] block uppercase text-[10px]">Reference Baseline</span>
              <span className="text-[#22D3EE] font-bold">
                {workspace.reference_session_id || 'Not Established'}
              </span>
            </div>
            <div className="bg-[#090B10] px-3 py-2 rounded-lg border border-[#232A3B]">
              <span className="text-[#64748B] block uppercase text-[10px]">Total Visits</span>
              <span className="text-[#F8FAFC] font-bold">{sessions.length}</span>
            </div>
            <div className="bg-[#090B10] px-3 py-2 rounded-lg border border-[#232A3B]">
              <span className="text-[#64748B] block uppercase text-[10px]">Tracked Entities</span>
              <span className="text-[#F8FAFC] font-bold">{workspace.total_entities}</span>
            </div>
            <button
              type="button"
              onClick={handleExportWorkspace}
              disabled={exporting || sessions.length === 0}
              className="bg-[#090B10] hover:bg-[#181D29] text-[#F8FAFC] px-3 py-2 rounded-lg border border-[#232A3B] flex items-center gap-2 transition disabled:opacity-40 cursor-pointer"
              title="Export complete experiment data as JSON"
            >
              {exporting ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin text-[#06B6D4]" />
              ) : (
                <Download className="w-3.5 h-3.5 text-[#06B6D4]" />
              )}
              <span className="font-semibold">{exporting ? 'Exporting...' : 'Export Experiment'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Hero Section: Scene-First Visual Canvas with Visit Scrubber */}
      {sessions.length > 0 && selectedSessionId && (
        <div className="bg-[#11141D] border border-[#232A3B] rounded-xl p-5 shadow-lg space-y-4">
          {/* Visit Scrubber Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#232A3B]">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[#94A3B8] flex items-center gap-1.5">
                <Compass className="w-4 h-4 text-[#06B6D4]" />
                Active Visit:
              </span>

              {/* Prev / Next controls */}
              <div className="flex items-center gap-1 bg-[#090B10] p-0.5 rounded-lg border border-[#232A3B]">
                <button
                  type="button"
                  onClick={handlePrevVisit}
                  disabled={currentSessionIndex <= 0}
                  className="p-1 text-[#94A3B8] hover:text-[#F8FAFC] disabled:opacity-30 transition cursor-pointer"
                  title="Previous Visit"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="text-xs font-mono font-bold text-[#22D3EE] px-2 min-w-[4rem] text-center">
                  Visit {currentSessionIndex + 1}/{sessions.length}
                </span>
                <button
                  type="button"
                  onClick={handleNextVisit}
                  disabled={currentSessionIndex >= sessions.length - 1}
                  className="p-1 text-[#94A3B8] hover:text-[#F8FAFC] disabled:opacity-30 transition cursor-pointer"
                  title="Next Visit"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              {/* Horizontal Visit Selector Pills */}
              <div className="hidden lg:flex items-center gap-1.5 ml-2">
                {sessions.map((sess, idx) => {
                  const isSelected = sess.session_id === selectedSessionId;
                  const isBaseline = idx === 0;
                  return (
                    <button
                      key={sess.session_id}
                      type="button"
                      onClick={() => {
                        setSelectedSessionId(sess.session_id);
                        setHighlightedEntityId(null);
                      }}
                      className={`px-2.5 py-1 rounded-md text-xs font-mono transition flex items-center gap-1.5 cursor-pointer ${
                        isSelected
                          ? 'bg-[#181D29] text-[#22D3EE] border border-[#06B6D4]/50 font-bold shadow-xs'
                          : 'bg-[#090B10] text-[#94A3B8] border border-[#232A3B] hover:text-[#F8FAFC]'
                      }`}
                    >
                      <span>V{idx + 1}</span>
                      {isBaseline && (
                        <span className="text-[9px] px-1 py-0.2 rounded bg-[#06B6D4]/15 text-[#22D3EE] uppercase font-sans">
                          Baseline
                        </span>
                      )}
                      {sess.events_count > 0 && (
                        <span className="w-1.5 h-1.5 rounded-full bg-[#F59E0B]" title={`${sess.events_count} events`} />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Selected visit metadata chips */}
            {selectedSessionDetail && (
              <div className="flex items-center gap-3 text-xs font-mono text-[#94A3B8]">
                <span>
                  ID: <span className="text-[#F8FAFC]">{selectedSessionDetail.session.session_id}</span>
                </span>
                <span>•</span>
                <span>{new Date(selectedSessionDetail.session.timestamp).toLocaleString()}</span>
                <span>•</span>
                {isUnregisteredSession ? (
                  <span className="text-[#F59E0B] font-bold">
                    {selectedSessionDetail.raw_detections?.length || 0} Raw Detections (Unregistered)
                  </span>
                ) : (
                  <span className="text-[#22D3EE] font-bold">
                    {selectedSessionDetail.observations.length} Detections
                  </span>
                )}
                {selectedSessionDetail.events.length > 0 && (
                  <span className="text-[#F59E0B] font-bold">
                    • {selectedSessionDetail.events.length} Change Events
                  </span>
                )}
                {sessions.length >= 2 && (
                  <button
                    type="button"
                    onClick={onNavigateToComparison}
                    className="ml-2 px-2.5 py-1 rounded-md bg-[#06B6D4]/15 hover:bg-[#06B6D4]/25 text-[#22D3EE] border border-[#06B6D4]/30 text-xs font-sans font-medium transition cursor-pointer"
                    title="Compare this workspace against baseline or previous visits"
                  >
                    Compare Scene →
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Registration Warning Banner for Unregistered Visits */}
          {isUnregisteredSession && (
            <div className="p-3 bg-amber-950/40 border border-amber-600/60 rounded-lg flex items-start gap-2.5 text-xs text-amber-200">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
              <div>
                <span className="font-semibold block text-amber-300">
                  Objects detected, but scene registration failed. Cross-visit change analysis is unavailable.
                </span>
                <span className="text-[11px] text-amber-200/80 mt-0.5 block font-mono">
                  {selectedSessionDetail?.registration?.failure_reason ||
                    'Planar homography could not align this frame with reference baseline. Raw detections are preserved in native sensor coordinates without corrupting temporal memory.'}
                </span>
              </div>
            </div>
          )}

          {/* Hero Image Canvas */}
          {loadingDetail || !selectedSessionDetail ? (
            <div className="aspect-video bg-[#090B10] rounded-xl border border-[#232A3B] flex flex-col items-center justify-center gap-2 text-[#94A3B8]">
              <Loader2 className="w-8 h-8 animate-spin text-[#06B6D4]" />
              <span className="text-xs font-mono">Loading observation image & localized detections...</span>
            </div>
          ) : (
            <BoundingBoxCanvas
              imageUrl={selectedSessionDetail.session.image_url}
              boxes={canvasBoxes}
              title={`Scene View — ${selectedSessionDetail.session.session_name}`}
              subtitle={
                isUnregisteredSession
                  ? `${canvasBoxes.length} unaligned sensor detections`
                  : `${canvasBoxes.length} localized object detections`
              }
              highlightedId={highlightedEntityId}
              onBoxSelect={(boxId) => setHighlightedEntityId(boxId === highlightedEntityId ? null : boxId)}
              aspectRatioClass="aspect-video"
            />
          )}

          {/* Detections Strip & Inspector for Active Visit */}
          {selectedSessionDetail && (selectedSessionDetail.observations.length > 0 || isUnregisteredSession) && (
            <div className="bg-[#090B10] rounded-lg border border-[#232A3B] p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-mono text-[#94A3B8] uppercase tracking-wider flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-[#06B6D4]" />
                  {isUnregisteredSession
                    ? `Raw Detections in Frame (${selectedSessionDetail.raw_detections?.length || 0}) [Unregistered Viewpoint]`
                    : `Detected Objects in Frame (${selectedSessionDetail.observations.length})`}
                </span>
                <span className="text-[11px] font-mono text-[#64748B]">Click an object to highlight in scene canvas</span>
              </div>

              <div className="flex flex-wrap gap-2">
                {isUnregisteredSession
                  ? (selectedSessionDetail.raw_detections || []).map((rd) => {
                      const isSelected = highlightedEntityId === rd.detection_id;
                      return (
                        <button
                          key={rd.detection_id}
                          type="button"
                          onClick={() =>
                            setHighlightedEntityId(isSelected ? null : rd.detection_id)
                          }
                          className={`px-3 py-1.5 rounded-lg border text-xs font-mono flex items-center gap-2 transition cursor-pointer ${
                            isSelected
                              ? 'bg-[#181D29] border-amber-500 text-amber-300 font-bold shadow-xs'
                              : 'bg-[#11141D] border-[#232A3B] text-[#94A3B8] hover:text-[#F8FAFC]'
                          }`}
                        >
                          <span className="text-[#F8FAFC]">{rd.label}</span>
                          {rd.confidence != null && (
                            <span className="text-[10px] text-amber-400">
                              {(rd.confidence * 100).toFixed(0)}%
                            </span>
                          )}
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 uppercase font-mono border border-amber-500/30">
                            UNREGISTERED
                          </span>
                        </button>
                      );
                    })
                  : selectedSessionDetail.observations.map((obs) => {
                      const isSelected = highlightedEntityId === obs.observation_id;
                      return (
                        <button
                          key={obs.observation_id}
                          type="button"
                          onClick={() =>
                            setHighlightedEntityId(isSelected ? null : obs.observation_id)
                          }
                          className={`px-3 py-1.5 rounded-lg border text-xs font-mono flex items-center gap-2 transition cursor-pointer ${
                            isSelected
                              ? 'bg-[#181D29] border-[#06B6D4] text-[#22D3EE] font-bold shadow-xs'
                              : 'bg-[#11141D] border-[#232A3B] text-[#94A3B8] hover:text-[#F8FAFC]'
                          }`}
                        >
                          <span className="text-[#F8FAFC]">{obs.label || obs.entity_id}</span>
                          {obs.detector_confidence != null && (
                            <span className="text-[10px] text-[#22D3EE]">
                              {(obs.detector_confidence * 100).toFixed(0)}%
                            </span>
                          )}
                          <StatusBadge status={obs.visibility_status} size="sm" />
                        </button>
                      );
                    })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Latest Analysis Outcome Feedback Banner */}
      {lastResult && (
        <div className="p-4 rounded-xl bg-[#11141D] border border-[#232A3B] animate-in fade-in duration-200">
          <div className="flex items-center justify-between pb-3 border-b border-[#232A3B] mb-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <span className="font-semibold text-sm text-[#F8FAFC]">
                Observation Successfully Analyzed: {lastResult.session_id}
              </span>
              <StatusBadge status={lastResult.registration.status} size="sm" />
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-[#94A3B8]">
                Pipeline Latency: {lastResult.total_latency_ms.toFixed(1)}ms
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono mb-3">
            <div className="bg-[#090B10] p-2.5 rounded-lg border border-[#232A3B]">
              <span className="text-[#64748B] block text-[10px]">Detections</span>
              <span className="text-[#F8FAFC] font-bold">{lastResult.detections.length} objects</span>
            </div>
            <div className="bg-[#090B10] p-2.5 rounded-lg border border-[#232A3B]">
              <span className="text-[#64748B] block text-[10px]">Registration Inliers</span>
              <span className="text-[#F8FAFC] font-bold">
                {(lastResult.registration.inlier_ratio * 100).toFixed(1)}% (
                {lastResult.registration.inliers_count})
              </span>
            </div>
            <div className="bg-[#090B10] p-2.5 rounded-lg border border-[#232A3B]">
              <span className="text-[#64748B] block text-[10px]">Keyframe Topology</span>
              <span className="text-[#F8FAFC] font-bold truncate block">
                {lastResult.registration.selected_keyframe_id || 'Canonical Baseline'}
              </span>
            </div>
            <div className="bg-[#090B10] p-2.5 rounded-lg border border-[#232A3B]">
              <span className="text-[#64748B] block text-[10px]">Belief State Machine</span>
              <span className="text-emerald-400 font-bold block">
                {lastResult.total_active_entities} Active Entities
              </span>
            </div>
          </div>

          {lastResult.events.length > 0 && (
            <div className="mt-3 pt-3 border-t border-[#232A3B]">
              <span className="text-[11px] font-mono text-[#94A3B8] uppercase tracking-wider block mb-2">
                Environmental State Transitions Emitted:
              </span>
              <div className="space-y-1.5">
                {lastResult.events.map((ev) => (
                  <div
                    key={ev.event_id}
                    className="p-2 rounded bg-[#090B10] border border-[#232A3B] flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[#22D3EE] font-medium">{ev.entity_id}</span>
                      <span className="text-[#64748B]">→</span>
                      <StatusBadge status={ev.new_state} size="sm" />
                      <span className="text-[#F8FAFC] ml-2">{ev.explanation}</span>
                    </div>
                    {ev.displacement_distance != null && (
                      <span className="font-mono text-[#94A3B8] text-[11px]">
                        Δ {ev.displacement_distance.toFixed(1)} px
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Dual Column Lower Grid: Ingest Dock (Left) + Visit Sequence Archive (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Upload / Add Observation Card */}
        <div className="lg:col-span-5 bg-[#11141D] border border-[#232A3B] rounded-xl overflow-hidden shadow-sm">
          {/* Mode Switcher Tabs */}
          <div className="px-5 py-3 border-b border-[#232A3B] flex items-center justify-between bg-[#0B0E14]">
            <div className="flex items-center gap-1.5 p-1 bg-[#11141D] rounded-lg border border-[#232A3B]">
              <button
                type="button"
                onClick={() => {
                  setIngestMode('single');
                  if (webcamActive) stopWebcam();
                }}
                className={`px-2.5 py-1 text-xs rounded-md font-medium transition cursor-pointer flex items-center gap-1.5 ${
                  ingestMode === 'single'
                    ? 'bg-[#06B6D4] text-[#090B10]'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC]'
                }`}
              >
                <Camera className="w-3.5 h-3.5" />
                Single
              </button>
              <button
                type="button"
                onClick={() => {
                  setIngestMode('batch');
                  if (webcamActive) stopWebcam();
                }}
                className={`px-2.5 py-1 text-xs rounded-md font-medium transition cursor-pointer flex items-center gap-1.5 ${
                  ingestMode === 'batch'
                    ? 'bg-[#06B6D4] text-[#090B10]'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC]'
                }`}
              >
                <Files className="w-3.5 h-3.5" />
                Batch Sequence
              </button>
              <button
                type="button"
                onClick={() => setIngestMode('webcam')}
                className={`px-2.5 py-1 text-xs rounded-md font-medium transition cursor-pointer flex items-center gap-1.5 ${
                  ingestMode === 'webcam'
                    ? 'bg-[#06B6D4] text-[#090B10]'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC]'
                }`}
              >
                <Video className="w-3.5 h-3.5" />
                Live Monitor
              </button>
            </div>
            <span className="text-[11px] text-[#64748B] font-mono">
              {sessions.length === 0 ? 'Visit 1 (Baseline)' : `Visit #${sessions.length + 1}`}
            </span>
          </div>

          {/* Mode 1: Single Visit Ingest */}
          {ingestMode === 'single' && (
            <form onSubmit={handleAnalyze} className="p-5 space-y-4">
              {/* Drag & Drop File Zone */}
              <div>
                {!previewUrl ? (
                  <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleDrop}
                    className="border-2 border-dashed border-[#232A3B] hover:border-[#06B6D4]/60 rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer bg-[#090B10] hover:bg-[#181D29]/40 transition aspect-video relative group"
                  >
                    <input
                      type="file"
                      accept="image/png,image/jpeg"
                      onChange={handleFileChange}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    <div className="w-10 h-10 rounded-full bg-[#11141D] border border-[#232A3B] flex items-center justify-center text-[#94A3B8] group-hover:text-[#06B6D4] group-hover:border-[#06B6D4]/40 transition mb-2">
                      <Upload className="w-5 h-5" />
                    </div>
                    <span className="text-xs font-medium text-[#F8FAFC]">
                      Click to browse or drag & drop visit photo
                    </span>
                    <span className="text-[11px] text-[#64748B] mt-1 font-mono">
                      PNG or JPEG (max 25MB)
                    </span>
                  </div>
                ) : (
                  <div className="relative rounded-xl overflow-hidden border border-[#232A3B] bg-[#090B10] aspect-video flex items-center justify-center">
                    <img
                      src={previewUrl}
                      alt="Selected preview"
                      className="w-full h-full object-contain"
                    />
                    <button
                      type="button"
                      onClick={handleClearSelected}
                      className="absolute top-2.5 right-2.5 p-1 rounded-full bg-black/80 hover:bg-black text-[#94A3B8] hover:text-[#F8FAFC] transition cursor-pointer"
                      title="Remove selected image"
                    >
                      <X className="w-4 h-4" />
                    </button>
                    <div className="absolute bottom-2 left-2 px-2.5 py-1 bg-black/85 rounded-md text-[10px] font-mono text-[#F8FAFC] border border-[#232A3B]">
                      {selectedFile?.name} ({(selectedFile!.size / 1024).toFixed(0)} KB)
                    </div>
                  </div>
                )}
              </div>

              {/* Visit Name */}
              <div>
                <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1 font-mono">
                  Visit Name
                </label>
                <input
                  type="text"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  placeholder={`e.g. Visit ${sessions.length + 1}`}
                  className="w-full px-3 py-2 bg-[#090B10] border border-[#232A3B] rounded-lg text-xs text-[#F8FAFC] focus:outline-hidden focus:border-[#06B6D4]"
                />
              </div>

              {/* Detector & Timestamp Row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1 font-mono">
                    Detector
                  </label>
                  <select
                    value={selectedDetector}
                    onChange={(e) => handleSelectDetector(e.target.value)}
                    className="w-full px-2.5 py-2 bg-[#090B10] border border-[#232A3B] rounded-lg text-xs text-[#F8FAFC] focus:outline-hidden focus:border-[#06B6D4] cursor-pointer"
                  >
                    {detectors.map((d) => (
                      <option key={d.name} value={d.name} disabled={!d.is_available}>
                        {d.name.toUpperCase()} {d.is_open_vocabulary ? '(Open-Vocab CLIP)' : '(COCO 80)'}{' '}
                        {!d.is_available ? '[Unavailable]' : ''}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1 font-mono">
                    Timestamp (Optional)
                  </label>
                  <input
                    type="text"
                    value={timestamp}
                    onChange={(e) => setTimestamp(e.target.value)}
                    placeholder="YYYY-MM-DDTHH:MM:SSZ"
                    className="w-full px-2.5 py-2 bg-[#090B10] border border-[#232A3B] rounded-lg text-xs text-[#F8FAFC] font-mono focus:outline-hidden focus:border-[#06B6D4]"
                  />
                </div>
              </div>

              {/* Open-Vocabulary Prompts Input (for YOLO-World) */}
              {selectedDetector.includes('world') && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-xs font-medium text-[#22D3EE] uppercase tracking-wider font-mono">
                      Open-Vocabulary Category Prompts
                    </label>
                    <span className="text-[10px] text-[#64748B] font-mono">Comma-separated CLIP prompts</span>
                  </div>
                  <input
                    type="text"
                    value={customPrompts}
                    onChange={(e) => setCustomPrompts(e.target.value)}
                    placeholder="notebook, drawing compass, pliers, scissors, bottle"
                    className="w-full px-3 py-2 bg-[#090B10] border border-[#06B6D4]/40 rounded-lg text-xs text-[#F8FAFC] focus:outline-hidden focus:border-[#22D3EE]"
                  />
                </div>
              )}

              {/* Confidence Slider */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-medium text-[#94A3B8] uppercase tracking-wider font-mono">
                    Confidence Threshold
                  </label>
                  <span className="text-xs font-mono font-bold text-[#22D3EE]">
                    {confidenceThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.80"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full accent-[#06B6D4] bg-[#090B10] cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-[#64748B] font-mono mt-0.5">
                  <span>0.10 (High Recall)</span>
                  <span>0.25 (Standard)</span>
                  <span>0.80 (High Precision)</span>
                </div>
              </div>

              {/* Observation Notes */}
              <div>
                <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1 font-mono">
                  Notes (Optional)
                </label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="e.g. Moved mug to the left corner; overhead lighting change."
                  rows={2}
                  className="w-full px-3 py-2 bg-[#090B10] border border-[#232A3B] rounded-lg text-xs text-[#F8FAFC] focus:outline-hidden focus:border-[#06B6D4] resize-none"
                />
              </div>

              {analysisError && (
                <div className="p-3 bg-rose-950/60 border border-rose-800 rounded-lg flex items-start gap-2.5 text-xs text-rose-300">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
                  <span>{analysisError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={!selectedFile || analyzing}
                className="w-full py-2.5 px-4 bg-[#06B6D4] hover:bg-[#22D3EE] text-[#090B10] rounded-lg font-semibold text-xs transition flex items-center justify-center gap-2 shadow-lg shadow-cyan-950/40 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Analyzing Scene & Running CV Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Analyze Scene & Ingest Observation</span>
                  </>
                )}
              </button>
            </form>
          )}

          {/* Mode 2: Batch Sequence Ingest */}
          {ingestMode === 'batch' && (
            <div className="p-5 space-y-4">
              <div>
                <div className="border-2 border-dashed border-[#232A3B] hover:border-[#06B6D4]/60 rounded-xl p-5 flex flex-col items-center justify-center text-center cursor-pointer bg-[#090B10] hover:bg-[#181D29]/40 transition relative group">
                  <input
                    type="file"
                    multiple
                    accept="image/png,image/jpeg"
                    onChange={handleBatchFileChange}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <div className="w-10 h-10 rounded-full bg-[#11141D] border border-[#232A3B] flex items-center justify-center text-[#94A3B8] group-hover:text-[#06B6D4] group-hover:border-[#06B6D4]/40 transition mb-2">
                    <Files className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-medium text-[#F8FAFC]">
                    Select multiple sequence photos (visit_01.png, visit_02.png...)
                  </span>
                  <span className="text-[11px] text-[#64748B] mt-1 font-mono">
                    Files will be sorted chronologically by filename
                  </span>
                </div>
              </div>

              {batchFiles.length > 0 && (
                <div className="p-3 bg-[#090B10] border border-[#232A3B] rounded-lg max-h-40 overflow-y-auto space-y-1">
                  <div className="flex items-center justify-between text-[11px] font-mono text-[#06B6D4] pb-1 border-b border-[#232A3B]">
                    <span>{batchFiles.length} sequential files queued:</span>
                    <button
                      type="button"
                      onClick={() => setBatchFiles([])}
                      className="text-rose-400 hover:text-rose-300 transition"
                    >
                      Clear
                    </button>
                  </div>
                  {batchFiles.map((f, i) => (
                    <div key={f.name + i} className="text-[11px] font-mono text-[#94A3B8] flex items-center justify-between">
                      <span>{i + 1}. {f.name}</span>
                      <span className="text-[#64748B]">{(f.size / 1024).toFixed(0)} KB</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Detector & Prompts */}
              <div>
                <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1 font-mono">
                  Detector
                </label>
                <select
                  value={selectedDetector}
                  onChange={(e) => handleSelectDetector(e.target.value)}
                  className="w-full px-2.5 py-2 bg-[#090B10] border border-[#232A3B] rounded-lg text-xs text-[#F8FAFC] focus:outline-hidden focus:border-[#06B6D4] cursor-pointer"
                >
                  {detectors.map((d) => (
                    <option key={d.name} value={d.name} disabled={!d.is_available}>
                      {d.name.toUpperCase()} {d.is_open_vocabulary ? '(Open-Vocab CLIP)' : '(COCO 80)'}{' '}
                      {!d.is_available ? '[Unavailable]' : ''}
                    </option>
                  ))}
                </select>
              </div>

              {selectedDetector.includes('world') && (
                <div>
                  <label className="block text-xs font-medium text-[#22D3EE] uppercase tracking-wider mb-1 font-mono">
                    Open-Vocabulary Category Prompts
                  </label>
                  <input
                    type="text"
                    value={customPrompts}
                    onChange={(e) => setCustomPrompts(e.target.value)}
                    placeholder="notebook, drawing compass, pliers, scissors, bottle"
                    className="w-full px-3 py-2 bg-[#090B10] border border-[#06B6D4]/40 rounded-lg text-xs text-[#F8FAFC] focus:outline-hidden focus:border-[#22D3EE]"
                  />
                </div>
              )}

              {/* Confidence */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-medium text-[#94A3B8] uppercase tracking-wider font-mono">
                    Confidence Threshold: <span className="text-[#22D3EE]">{confidenceThreshold.toFixed(2)}</span>
                  </label>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.80"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full accent-[#06B6D4] bg-[#090B10] cursor-pointer"
                />
              </div>

              {analysisError && (
                <div className="p-3 bg-rose-950/60 border border-rose-800 rounded-lg flex items-start gap-2.5 text-xs text-rose-300">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
                  <span>{analysisError}</span>
                </div>
              )}

              <button
                type="button"
                onClick={handleProcessBatch}
                disabled={batchFiles.length === 0 || analyzing}
                className="w-full py-2.5 px-4 bg-[#06B6D4] hover:bg-[#22D3EE] text-[#090B10] rounded-lg font-semibold text-xs transition flex items-center justify-center gap-2 shadow-lg shadow-cyan-950/40 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>
                      {batchProgress ? `Processing Visit ${batchProgress.current}/${batchProgress.total}...` : 'Processing Sequence...'}
                    </span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Ingest Entire Batch Sequence ({batchFiles.length} files)</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Mode 3: Live Camera Monitoring */}
          {ingestMode === 'webcam' && (
            <div className="p-5 space-y-4">
              <div className="relative rounded-xl overflow-hidden border border-[#232A3B] bg-[#090B10] aspect-video flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-cover ${webcamActive ? 'block' : 'hidden'}`}
                />
                {!webcamActive && (
                  <div className="flex flex-col items-center justify-center p-6 text-center">
                    <Video className="w-10 h-10 text-[#64748B] mb-2" />
                    <p className="text-xs text-[#94A3B8]">Camera is currently offline</p>
                    <p className="text-[11px] text-[#64748B] font-mono mt-1">
                      Start stream to continuously monitor physical workbench
                    </p>
                  </div>
                )}
                {webcamActive && isAutoCapturing && webcamCountdown !== null && (
                  <div className="absolute top-3 left-3 px-2 py-1 bg-black/80 border border-emerald-500/50 rounded-md text-[11px] font-mono text-emerald-400 flex items-center gap-1.5 animate-pulse">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    Auto-capturing in {webcamCountdown}s
                  </div>
                )}
              </div>

              {webcamError && (
                <div className="p-3 bg-rose-950/60 border border-rose-800 rounded-lg flex items-start gap-2.5 text-xs text-rose-300">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
                  <span>{webcamError}</span>
                </div>
              )}

              {/* Camera Activation Controls */}
              <div className="flex gap-2">
                {!webcamActive ? (
                  <button
                    type="button"
                    onClick={startWebcam}
                    className="flex-1 py-2 px-3 bg-[#06B6D4] hover:bg-[#22D3EE] text-[#090B10] rounded-lg font-semibold text-xs transition flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5" />
                    Start Camera Stream
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={stopWebcam}
                    className="flex-1 py-2 px-3 bg-rose-600 hover:bg-rose-500 text-[#F8FAFC] rounded-lg font-semibold text-xs transition flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <Square className="w-3.5 h-3.5" />
                    Stop Camera
                  </button>
                )}
              </div>

              {/* Interval and Trigger Controls when active */}
              {webcamActive && (
                <div className="space-y-3 pt-2 border-t border-[#232A3B]">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1 font-mono">
                        Capture Interval
                      </label>
                      <select
                        value={webcamInterval}
                        onChange={(e) => setWebcamInterval(parseInt(e.target.value, 10))}
                        className="w-full px-2.5 py-1.5 bg-[#090B10] border border-[#232A3B] rounded-lg text-xs text-[#F8FAFC] font-mono focus:outline-hidden focus:border-[#06B6D4] cursor-pointer"
                      >
                        <option value="0">Manual Capture Only</option>
                        <option value="5">Every 5 Seconds</option>
                        <option value="10">Every 10 Seconds</option>
                        <option value="30">Every 30 Seconds</option>
                      </select>
                    </div>

                    <div className="flex flex-col justify-end">
                      {webcamInterval > 0 ? (
                        <button
                          type="button"
                          onClick={() => setIsAutoCapturing(!isAutoCapturing)}
                          className={`w-full py-1.5 px-3 rounded-lg font-semibold text-xs transition flex items-center justify-center gap-1.5 cursor-pointer ${
                            isAutoCapturing
                              ? 'bg-amber-500 hover:bg-amber-400 text-black'
                              : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                          }`}
                        >
                          {isAutoCapturing ? (
                            <>
                              <Pause className="w-3.5 h-3.5" />
                              Pause Auto-Capture
                            </>
                          ) : (
                            <>
                              <Play className="w-3.5 h-3.5" />
                              Resume Auto-Capture
                            </>
                          )}
                        </button>
                      ) : (
                        <button
                          type="button"
                          disabled
                          className="w-full py-1.5 px-3 bg-[#181D29] text-[#64748B] rounded-lg font-mono text-[11px] cursor-not-allowed"
                        >
                          Manual Mode Active
                        </button>
                      )}
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={handleCaptureWebcamVisit}
                    disabled={analyzing}
                    className="w-full py-2.5 px-4 bg-[#06B6D4] hover:bg-[#22D3EE] text-[#090B10] rounded-lg font-semibold text-xs transition flex items-center justify-center gap-2 shadow-lg shadow-cyan-950/40 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                  >
                    {analyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Processing Live Frame through Temporal Engine...</span>
                      </>
                    ) : (
                      <>
                        <Camera className="w-4 h-4" />
                        <span>Capture Single Observation Now</span>
                      </>
                    )}
                  </button>

                  <div className="p-2.5 bg-[#090B10] rounded-lg border border-[#232A3B] text-[11px] text-[#94A3B8] font-mono leading-relaxed space-y-1.5">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="text-[#F8FAFC] font-semibold">
                        Active Detector: <span className="text-[#06B6D4]">{selectedDetector.toUpperCase()}</span>
                        {selectedDetector === 'yolov8m' ? ' (Default)' : ' (Alternative)'}
                      </span>
                      <span>Skipped ticks: {droppedWebcamFrames}</span>
                    </div>
                    {lastResult && (
                      <div className="flex flex-wrap gap-1.5 pt-1 border-t border-[#232A3B]/60">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] ${lastResult.detections.length > 0 ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-700/40' : 'bg-amber-950/60 text-amber-300 border border-amber-700/40'}`}>
                          {lastResult.detections.length > 0 ? `Detected (${lastResult.detections.length})` : 'Not detected (0)'}
                        </span>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] ${lastResult.registration.success ? 'bg-cyan-950/60 text-cyan-300 border border-cyan-700/40' : 'bg-rose-950/60 text-rose-300 border border-rose-700/40'}`}>
                          {lastResult.registration.success ? `Reg: ${lastResult.registration.status}` : `Registration unreliable (${lastResult.registration.status})`}
                        </span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#181D29] text-[#CBD5E1] border border-[#232A3B]">
                          {lastResult.total_latency_ms.toFixed(0)} ms
                        </span>
                      </div>
                    )}
                    <div className="text-[10px] text-[#64748B]">
                      Epistemic gating: a single missed frame marks PENDING_REMOVAL / INSUFFICIENT_EVIDENCE rather than CONFIRMED_REMOVED; unreliable registration withholds movement claims.
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Observation Sequence History Table */}
        <div className="lg:col-span-7 bg-[#11141D] border border-[#232A3B] rounded-xl overflow-hidden shadow-sm">
          <div className="px-5 py-4 border-b border-[#232A3B] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-[#06B6D4]" />
              <h2 className="text-sm font-semibold text-[#F8FAFC]">Observation Sequence History</h2>
            </div>
            <button
              onClick={() => fetchSessions()}
              className="text-xs text-[#94A3B8] hover:text-[#F8FAFC] flex items-center gap-1.5 transition cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh
            </button>
          </div>

          {loadingSessions ? (
            <div className="py-16 flex justify-center text-[#94A3B8]">
              <Loader2 className="w-6 h-6 animate-spin text-[#06B6D4]" />
            </div>
          ) : sessions.length === 0 ? (
            <div className="py-16 text-center text-[#94A3B8] text-xs">
              No observations recorded yet. Upload the first image to establish the baseline reference visit.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-[#F8FAFC]">
                <thead className="bg-[#090B10] text-[10px] uppercase font-mono text-[#94A3B8] border-b border-[#232A3B]">
                  <tr>
                    <th className="px-4 py-3">Visit</th>
                    <th className="px-4 py-3">Timestamp (UTC)</th>
                    <th className="px-4 py-3 text-center">Objects</th>
                    <th className="px-4 py-3 text-center">Events</th>
                    <th className="px-4 py-3 text-right">Canvas</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#232A3B]">
                  {sessions.map((sess, idx) => {
                    const isSelected = sess.session_id === selectedSessionId;
                    return (
                      <tr
                        key={sess.session_id}
                        className={`transition cursor-pointer ${
                          isSelected ? 'bg-[#181D29]/90' : 'hover:bg-[#181D29]/40'
                        }`}
                        onClick={() => setSelectedSessionId(sess.session_id)}
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-3">
                            <img
                              src={sess.image_url}
                              alt={sess.session_name}
                              className="w-12 h-9 object-cover rounded bg-[#090B10] border border-[#232A3B]"
                            />
                            <div>
                              <span className="font-semibold text-[#F8FAFC] block">
                                {sess.session_name}
                              </span>
                              <span className="font-mono text-[10px] text-[#64748B]">
                                {sess.session_id} {idx === 0 && '• Baseline'}
                              </span>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3 font-mono text-[11px] text-[#94A3B8]">
                          {new Date(sess.timestamp).toLocaleString()}
                        </td>
                        <td className="px-4 py-3 font-mono text-center font-bold text-[#F8FAFC]">
                          {sess.observations_count === 0 && (sess.raw_detections_count || 0) > 0 ? (
                            <span className="text-amber-400 font-bold" title="Unregistered frame detections">
                              {sess.raw_detections_count}{' '}
                              <span className="text-[10px] text-[#94A3B8] font-normal font-sans">(unreg)</span>
                            </span>
                          ) : (
                            sess.observations_count
                          )}
                        </td>
                        <td className="px-4 py-3 font-mono text-center">
                          {sess.events_count > 0 ? (
                            <span className="text-[#F59E0B] font-bold">{sess.events_count}</span>
                          ) : (
                            <span className="text-[#64748B]">0</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedSessionId(sess.session_id);
                            }}
                            className={`text-[11px] font-mono px-2.5 py-1 rounded transition cursor-pointer ${
                              isSelected
                                ? 'bg-[#06B6D4]/20 text-[#22D3EE] border border-[#06B6D4]/40 font-bold'
                                : 'bg-[#181D29] text-[#94A3B8] hover:text-[#F8FAFC]'
                            }`}
                          >
                            {isSelected ? 'Active' : 'View'}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
