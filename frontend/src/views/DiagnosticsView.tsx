import React, { useState, useEffect } from 'react';
import type { WorkspaceSummary, DiagnosticsResponse } from '../types/api';
import { api } from '../api/client';
import { StatusBadge } from '../components/StatusBadge';
import {
  Activity,
  Cpu,
  ShieldAlert,
  Info,
  RefreshCw,
  Loader2,
  AlertTriangle,
} from 'lucide-react';

interface DiagnosticsViewProps {
  workspace: WorkspaceSummary;
}

export const DiagnosticsView: React.FC<DiagnosticsViewProps> = ({ workspace }) => {
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDiagnostics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getDiagnostics(workspace.workspace_id);
      setDiagnostics(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load diagnostics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDiagnostics();
  }, [workspace.workspace_id]);

  const detector = diagnostics?.detector_info;
  const regHealth = diagnostics?.registration_health;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 animate-in fade-in duration-150">
      {/* Header */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-orange-600/10 border border-orange-500/20 text-orange-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-neutral-100">System & Pipeline Diagnostics</h1>
              <p className="text-xs text-neutral-400">
                Inspect computer vision inference performance, planar registration health, and provenance.
              </p>
            </div>
          </div>

          <button
            onClick={fetchDiagnostics}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-neutral-950 border border-neutral-800 hover:border-neutral-700 text-xs text-neutral-300 font-mono transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-orange-500' : ''}`} />
            Refresh Diagnostics
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-950/50 border border-rose-800 rounded-xl text-xs text-rose-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {loading || !diagnostics ? (
        <div className="py-24 flex justify-center text-neutral-500">
          <Loader2 className="w-8 h-8 animate-spin text-orange-500" />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Object Detector Inspection */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-6 shadow-sm space-y-4">
            <div className="flex items-center gap-2 pb-3 border-b border-neutral-800">
              <Cpu className="w-5 h-5 text-orange-500" />
              <h2 className="text-sm font-semibold text-neutral-100 uppercase tracking-wider font-mono">
                Real Object Detector Engine
              </h2>
            </div>

            {detector ? (
              <div className="space-y-3 text-xs font-mono">
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-neutral-950 p-2.5 rounded-lg border border-neutral-800">
                    <span className="text-neutral-500 block text-[10px]">Architecture</span>
                    <span className="text-neutral-200 font-bold">{detector.architecture || 'yolov8'}</span>
                  </div>
                  <div className="bg-neutral-950 p-2.5 rounded-lg border border-neutral-800">
                    <span className="text-neutral-500 block text-[10px]">Framework</span>
                    <span className="text-neutral-200 font-bold">{detector.framework || 'onnx'}</span>
                  </div>
                  <div className="bg-neutral-950 p-2.5 rounded-lg border border-neutral-800">
                    <span className="text-neutral-500 block text-[10px]">Backend</span>
                    <span className="text-neutral-200 font-bold">{detector.backend || 'OpenCV DNN (CPU)'}</span>
                  </div>
                  <div className="bg-neutral-950 p-2.5 rounded-lg border border-neutral-800">
                    <span className="text-neutral-500 block text-[10px]">COCO Classes</span>
                    <span className="text-neutral-200 font-bold">{detector.classes_count || 80}</span>
                  </div>
                </div>

                <div className="bg-neutral-950 p-3 rounded-lg border border-neutral-800">
                  <span className="text-neutral-500 block text-[10px] mb-1">Weights Path</span>
                  <span className="text-neutral-300 break-all">{detector.weights_path || 'models/yolov8n.onnx'}</span>
                </div>

                {detector.limitations && Array.isArray(detector.limitations) && (
                  <div className="mt-4 pt-3 border-t border-neutral-800/80">
                    <span className="text-[11px] font-mono text-neutral-400 uppercase tracking-wider block mb-2">
                      Known Detector Limitations
                    </span>
                    <ul className="space-y-1.5 text-neutral-400 text-xs font-sans list-disc pl-4">
                      {detector.limitations.map((lim: string, idx: number) => (
                        <li key={idx}>{lim}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-xs text-neutral-500 font-mono">No detector details returned.</div>
            )}
          </div>

          {/* Planar Registration Health */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-6 shadow-sm space-y-4">
            <div className="flex items-center gap-2 pb-3 border-b border-neutral-800">
              <ShieldAlert className="w-5 h-5 text-orange-500" />
              <h2 className="text-sm font-semibold text-neutral-100 uppercase tracking-wider font-mono">
                Planar Registration Health
              </h2>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="flex items-center justify-between bg-neutral-950 p-3 rounded-lg border border-neutral-800">
                <span className="text-neutral-400">Alignment Quality Status:</span>
                <StatusBadge status={regHealth?.status || 'UNKNOWN'} size="sm" />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="bg-neutral-950 p-2.5 rounded-lg border border-neutral-800">
                  <span className="text-neutral-500 block text-[10px]">Recent Mean Inlier Ratio</span>
                  <span className="text-neutral-200 font-bold">
                    {regHealth?.recent_alignment_quality_mean != null
                      ? `${(regHealth.recent_alignment_quality_mean * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="bg-neutral-950 p-2.5 rounded-lg border border-neutral-800">
                  <span className="text-neutral-500 block text-[10px]">Samples Evaluated</span>
                  <span className="text-neutral-200 font-bold">{regHealth?.samples_evaluated ?? 0}</span>
                </div>
              </div>

              <div className="p-3 bg-neutral-950/60 rounded-lg border border-neutral-800/80 text-xs font-sans text-neutral-400 space-y-1.5">
                <p>
                  <strong className="text-neutral-300 font-mono">Phase 6 Adaptive Keyframes:</strong> Maintains a verified topology
                  of reference keyframes to register queries across wide perspective shifts and crops without mutating the initial canonical origin.
                </p>
                <p>
                  <strong className="text-neutral-300 font-mono">Uncertainty-Aware Movement:</strong> Movement decisions incorporate a dynamic margin
                  &Delta;U derived from reprojection error, inlier density, spatial coverage, and resolution scaling.
                </p>
              </div>
            </div>

            {/* Entity State Distribution */}
            <div className="mt-6 pt-4 border-t border-neutral-800">
              <span className="text-[11px] font-mono text-neutral-400 uppercase tracking-wider block mb-3">
                Entity State Distribution ({diagnostics.total_entities} Active Entities)
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {Object.entries(diagnostics.entity_state_distribution || {}).map(([st, cnt]) => (
                  <div
                    key={st}
                    className="p-2.5 rounded-lg bg-neutral-950 border border-neutral-800 flex items-center justify-between"
                  >
                    <span className="text-[11px] font-mono text-neutral-400 truncate mr-2">{st}</span>
                    <span className="font-mono font-bold text-neutral-100">{cnt}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Dataset Provenance Disclosure */}
          <div className="lg:col-span-2 bg-neutral-900 border border-neutral-800 rounded-xl p-6 shadow-sm">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-neutral-400 shrink-0 mt-0.5" />
              <div>
                <h3 className="text-sm font-semibold text-neutral-200 mb-1">
                  Dataset Provenance & Rigorous Evaluation Notice
                </h3>
                <p className="text-xs text-neutral-400 font-sans leading-relaxed">
                  The initial 7-image pilot fixtures in{' '}
                  <code className="text-neutral-300 bg-neutral-950 px-1 py-0.5 rounded font-mono">
                    data/real_workbench/visits
                  </code>{' '}
                  are AI-generated synthetic collage crops designed for pipeline integration and temporal
                  regression testing. They do NOT constitute proof of general physical real-world accuracy.
                  ObjectMemory strictly enforces separation between synthetic pilot verification and real-world
                  generalization.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
