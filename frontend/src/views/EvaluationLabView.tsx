import React, { useState, useEffect } from 'react';
import type {
  WorkspaceSummary,
  DatasetMetaResponse,
  EvaluationRunResponse,
} from '../types/api';
import { api } from '../api/client';
import {
  BarChart3,
  Play,
  Download,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  Cpu,
  Layers,
  FileSpreadsheet,
  FileCode,
  RefreshCw,
} from 'lucide-react';

interface EvaluationLabViewProps {
  workspace?: WorkspaceSummary | null;
}

export const EvaluationLabView: React.FC<EvaluationLabViewProps> = ({ workspace }) => {
  const [datasets, setDatasets] = useState<DatasetMetaResponse[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('desk_pilot_01');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.35);
  const [activeRun, setActiveRun] = useState<EvaluationRunResponse | null>(null);
  const [pastRuns, setPastRuns] = useState<EvaluationRunResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const fetchMeta = async () => {
      setLoading(true);
      try {
        const [dList, rList] = await Promise.all([
          api.listDatasets(),
          api.listEvaluationRuns().catch(() => []),
        ]);
        if (mounted) {
          setDatasets(dList);
          if (dList.length > 0) {
            setSelectedDatasetId(dList[0].dataset_id);
          }
          setPastRuns(rList);
          if (rList.length > 0) {
            setActiveRun(rList[rList.length - 1]);
          }
        }
      } catch (err: any) {
        if (mounted) setError(err.message || 'Failed to load evaluation metadata.');
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchMeta();
    return () => {
      mounted = false;
    };
  }, []);

  const handleRunEvaluation = async () => {
    setEvaluating(true);
    setError(null);
    try {
      const res = await api.runEvaluation({
        dataset_id: selectedDatasetId,
        workspace_id: workspace?.workspace_id,
        confidence_threshold: confidenceThreshold,
      });
      setActiveRun(res);
      setPastRuns((prev) => [...prev, res]);
    } catch (err: any) {
      setError(err.message || 'Evaluation run failed.');
    } finally {
      setEvaluating(false);
    }
  };

  const selectedDataset = datasets.find((d) => d.dataset_id === selectedDatasetId);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 animate-in fade-in duration-150">
      {/* Header */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-orange-600/10 border border-orange-500/20 text-orange-400">
              <BarChart3 className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-neutral-100 flex items-center gap-2">
                Evaluation Lab & Benchmark Harness
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-400 border border-neutral-700">
                  Reproducible Testing
                </span>
              </h1>
              <p className="text-xs text-neutral-400">
                Measure change detection precision, recall, F1, registration robustness, and latency against ground truth.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {pastRuns.length > 0 && (
              <select
                value={activeRun?.run_id || ''}
                onChange={(e) => {
                  const found = pastRuns.find((r) => r.run_id === e.target.value);
                  if (found) setActiveRun(found);
                }}
                className="bg-neutral-950 border border-neutral-800 text-neutral-300 text-xs font-mono rounded-lg px-2.5 py-1.5 focus:outline-hidden"
              >
                {pastRuns.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} (F1: {r.metrics.f1_score?.toFixed(3) ?? 'N/A'})
                  </option>
                ))}
              </select>
            )}

            <button
              type="button"
              onClick={handleRunEvaluation}
              disabled={evaluating}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-white text-xs font-semibold shadow-md transition cursor-pointer"
            >
              {evaluating || loading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>{loading ? 'Loading...' : 'Evaluating Sequence...'}</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Execute Benchmark Run</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mandatory Provenance & Scientific Disclaimer Banner */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start gap-3 text-amber-200">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1 text-xs">
          <span className="font-bold text-amber-300 block">
            DATASET PROVENANCE NOTICE & SCIENTIFIC CITATION POLICY
          </span>
          <p className="leading-relaxed text-amber-200/90">
            The pilot dataset consists of 7 synthetic AI-generated workbench image crops.
            Evaluation results verify algorithm correctness, boundary movement safeguards, and conservative cross-class association,
            but <strong className="text-amber-100 underline decoration-amber-400/50">DO NOT constitute proof of general physical real-world accuracy</strong>.
            Evaluation on physical workbench photographs is strictly gated pending manual laboratory ground truth annotations.
          </p>
        </div>
      </div>

      {/* Configuration & Dataset Selector Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Dataset Selection */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 flex flex-col gap-3">
          <span className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-semibold flex items-center gap-2">
            <Layers className="w-4 h-4 text-orange-400" />
            Benchmark Dataset
          </span>
          <select
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            className="w-full bg-neutral-950 border border-neutral-800 rounded-lg p-2.5 text-xs text-neutral-200 font-mono focus:border-neutral-700 focus:outline-hidden"
          >
            {datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.name} ({d.total_visits} visits)
              </option>
            ))}
          </select>
          {selectedDataset && (
            <p className="text-[11px] text-neutral-400 leading-relaxed bg-neutral-950 p-2.5 rounded-lg border border-neutral-800/80">
              {selectedDataset.description}
            </p>
          )}
        </div>

        {/* Hyperparameters */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 flex flex-col gap-3">
          <span className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-semibold flex items-center gap-2">
            <Cpu className="w-4 h-4 text-orange-400" />
            Detection Threshold
          </span>
          <div className="flex items-center justify-between text-xs font-mono text-neutral-300">
            <span>Confidence Threshold:</span>
            <span className="text-orange-400 font-bold">{(confidenceThreshold * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            className="w-full accent-orange-500 cursor-pointer"
          />
          <span className="text-[10px] text-neutral-500 font-mono">
            Standard: 0.35 (MobileNet-SSD default operating point for real-world candidate filtering)
          </span>
        </div>

        {/* Export & Actions */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 flex flex-col justify-between gap-3">
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-semibold flex items-center gap-2">
              <Download className="w-4 h-4 text-orange-400" />
              Audit Report Export
            </span>
            <p className="text-[11px] text-neutral-400 mt-1">
              Download complete reproducibility package with per-event confusion matrix.
            </p>
          </div>
          {activeRun ? (
            <div className="flex items-center gap-2">
              <a
                href={api.getEvaluationExportUrl(activeRun.run_id, 'json')}
                target="_blank"
                rel="noreferrer"
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-neutral-700 text-xs font-mono transition"
              >
                <FileCode className="w-3.5 h-3.5 text-blue-400" />
                <span>JSON</span>
              </a>
              <a
                href={api.getEvaluationExportUrl(activeRun.run_id, 'csv')}
                target="_blank"
                rel="noreferrer"
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-neutral-700 text-xs font-mono transition"
              >
                <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
                <span>CSV</span>
              </a>
            </div>
          ) : (
            <span className="text-xs text-neutral-500 font-mono">Run evaluation to enable export.</span>
          )}
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 bg-rose-950/40 border border-rose-800 rounded-xl text-xs text-rose-300 flex items-center gap-2 font-mono">
          <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Evaluation Results Section */}
      {activeRun && (
        <div className="space-y-6">
          {/* Key Quantitative Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
            {/* Precision */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 flex flex-col justify-between">
              <span className="text-[10px] font-mono text-neutral-500 uppercase">Change Precision</span>
              <div className="text-2xl font-bold font-mono text-emerald-400 mt-2">
                {activeRun.metrics.precision != null ? `${(activeRun.metrics.precision * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <span className="text-[10px] font-mono text-neutral-500 mt-1">TP / (TP + FP)</span>
            </div>

            {/* Recall */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 flex flex-col justify-between">
              <span className="text-[10px] font-mono text-neutral-500 uppercase">Change Recall</span>
              <div className="text-2xl font-bold font-mono text-emerald-400 mt-2">
                {activeRun.metrics.recall != null ? `${(activeRun.metrics.recall * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <span className="text-[10px] font-mono text-neutral-500 mt-1">TP / (TP + FN)</span>
            </div>

            {/* F1 Score */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 flex flex-col justify-between">
              <span className="text-[10px] font-mono text-neutral-500 uppercase">F1 Score</span>
              <div className="text-2xl font-bold font-mono text-orange-400 mt-2">
                {activeRun.metrics.f1_score != null ? activeRun.metrics.f1_score.toFixed(3) : 'N/A'}
              </div>
              <span className="text-[10px] font-mono text-neutral-500 mt-1">Harmonic mean</span>
            </div>

            {/* Registration Success */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 flex flex-col justify-between">
              <span className="text-[10px] font-mono text-neutral-500 uppercase">Planar Reg Rate</span>
              <div className="text-2xl font-bold font-mono text-neutral-100 mt-2">
                {(activeRun.metrics.registration_success_rate * 100).toFixed(1)}%
              </div>
              <span className="text-[10px] font-mono text-neutral-500 mt-1">Inliers ≥ 20%</span>
            </div>

            {/* Confirmation Delay */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 flex flex-col justify-between">
              <span className="text-[10px] font-mono text-neutral-500 uppercase">Confirm Delay</span>
              <div className="text-2xl font-bold font-mono text-amber-400 mt-2">
                {activeRun.metrics.change_confirmation_delay_visits.toFixed(1)}
              </div>
              <span className="text-[10px] font-mono text-neutral-500 mt-1">Mean visits to confirm</span>
            </div>

            {/* False Movement Suppression */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 flex flex-col justify-between">
              <span className="text-[10px] font-mono text-neutral-500 uppercase">Gated Movements</span>
              <div className="text-2xl font-bold font-mono text-blue-400 mt-2">
                {activeRun.metrics.false_movement_suppression_count}
              </div>
              <span className="text-[10px] font-mono text-neutral-500 mt-1">Boundary/Reg safe</span>
            </div>
          </div>

          {/* CPU Resource & Latency Profile */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5">
            <span className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-semibold flex items-center gap-2 mb-3">
              <Clock className="w-4 h-4 text-orange-400" />
              CPU Runtime & Latency Profile ({activeRun.cpu_profile.detector_name})
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
              <div className="bg-neutral-950 p-3 rounded-lg border border-neutral-800">
                <span className="text-neutral-500 block text-[10px] uppercase">Mean Detector</span>
                <span className="text-neutral-200 text-sm font-bold">
                  {activeRun.cpu_profile.mean_detector_latency_ms.toFixed(1)} ms
                </span>
              </div>
              <div className="bg-neutral-950 p-3 rounded-lg border border-neutral-800">
                <span className="text-neutral-500 block text-[10px] uppercase">Mean Registration</span>
                <span className="text-neutral-200 text-sm font-bold">
                  {activeRun.cpu_profile.mean_registration_latency_ms.toFixed(1)} ms
                </span>
              </div>
              <div className="bg-neutral-950 p-3 rounded-lg border border-neutral-800">
                <span className="text-neutral-500 block text-[10px] uppercase">Mean Total Pipeline</span>
                <span className="text-orange-400 text-sm font-bold">
                  {activeRun.cpu_profile.mean_pipeline_latency_ms.toFixed(1)} ms
                </span>
              </div>
              <div className="bg-neutral-950 p-3 rounded-lg border border-neutral-800">
                <span className="text-neutral-500 block text-[10px] uppercase">P95 Total Pipeline</span>
                <span className="text-neutral-200 text-sm font-bold">
                  {activeRun.cpu_profile.p95_latency_ms.toFixed(1)} ms
                </span>
              </div>
            </div>
          </div>

          {/* Event Confusion Matrix Table */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl overflow-hidden shadow-sm">
            <div className="px-6 py-4 border-b border-neutral-800 flex items-center justify-between">
              <div>
                <h3 className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-semibold">
                  Ground Truth vs Prediction Audit Matrix ({activeRun.events.length} evaluations)
                </h3>
                <span className="text-[11px] text-neutral-500 font-mono">
                  Accuracy: {activeRun.metrics.correct_predictions} / {activeRun.metrics.total_evaluations} (
                  {((activeRun.metrics.correct_predictions / activeRun.metrics.total_evaluations) * 100).toFixed(1)}%)
                </span>
              </div>
              <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-neutral-800 text-neutral-300 border border-neutral-700">
                Run ID: {activeRun.run_id}
              </span>
            </div>

            <div className="overflow-x-auto max-h-[450px]">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-neutral-950 text-neutral-400 border-b border-neutral-800 uppercase text-[10px] sticky top-0 z-10">
                  <tr>
                    <th className="px-4 py-3">Visit</th>
                    <th className="px-4 py-3">Entity ID</th>
                    <th className="px-4 py-3">Label</th>
                    <th className="px-4 py-3">Expected State</th>
                    <th className="px-4 py-3">Predicted State</th>
                    <th className="px-4 py-3">Match</th>
                    <th className="px-4 py-3">Displacement</th>
                    <th className="px-4 py-3">Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800">
                  {activeRun.events.map((ev, idx) => (
                    <tr key={`${ev.visit_id}-${ev.entity_id}-${idx}`} className="hover:bg-neutral-800/40">
                      <td className="px-4 py-2.5 text-neutral-300 font-bold">{ev.visit_id}</td>
                      <td className="px-4 py-2.5 text-orange-400">{ev.entity_id}</td>
                      <td className="px-4 py-2.5 text-neutral-200">{ev.entity_label}</td>
                      <td className="px-4 py-2.5 text-neutral-400">{ev.expected_state}</td>
                      <td className="px-4 py-2.5 text-neutral-200 font-semibold">{ev.predicted_state}</td>
                      <td className="px-4 py-2.5">
                        {ev.is_match ? (
                          <span className="inline-flex items-center gap-1 text-emerald-400 font-bold text-[11px]">
                            <CheckCircle2 className="w-3.5 h-3.5" />
                            MATCH
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-rose-400 font-bold text-[11px]">
                            <XCircle className="w-3.5 h-3.5" />
                            MISMATCH
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-neutral-400">
                        {ev.displacement_px ? `${ev.displacement_px.toFixed(1)} px` : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-neutral-500 max-w-xs truncate" title={ev.notes || ''}>
                        {ev.notes || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
