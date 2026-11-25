import React, { useState, useEffect } from 'react';
import type {
  EthFrameItem,
  EthHistoricalSummaryResponse,
  EthRegisterResponse,
  EthSessionInfo,
} from '../types/api';
import { api } from '../api/client';
import {
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Play,
  ScanSearch,
  Info,
  Loader2,
} from 'lucide-react';

export const EthRegistrationView: React.FC = () => {
  const [available, setAvailable] = useState<boolean>(false);
  const [scopeNotice, setScopeNotice] = useState<string>('');
  const [sessions, setSessions] = useState<EthSessionInfo[]>([]);
  const [loadingSessions, setLoadingSessions] = useState<boolean>(true);

  // Reference selection
  const [refSessionId, setRefSessionId] = useState<string>('observation_01');
  const [refStride, setRefStride] = useState<number>(25);
  const [refFrames, setRefFrames] = useState<EthFrameItem[]>([]);
  const [refTotal, setRefTotal] = useState<number>(0);
  const [refFrame, setRefFrame] = useState<string>('frame_00000.png');

  // Query selection
  const [querySessionId, setQuerySessionId] = useState<string>('observation_01');
  const [queryStride, setQueryStride] = useState<number>(25);
  const [queryFrames, setQueryFrames] = useState<EthFrameItem[]>([]);
  const [queryTotal, setQueryTotal] = useState<number>(0);
  const [queryFrame, setQueryFrame] = useState<string>('frame_00001.png');

  // Registration state
  const [running, setRunning] = useState<boolean>(false);
  const [result, setResult] = useState<EthRegisterResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [historical, setHistorical] = useState<EthHistoricalSummaryResponse | null>(null);

  useEffect(() => {
    let mounted = true;
    const init = async () => {
      setLoadingSessions(true);
      try {
        const [sessResp, histResp] = await Promise.all([
          api.listEthSessions(),
          api.getEthHistoricalSummary().catch(() => null),
        ]);
        if (!mounted) return;
        setAvailable(sessResp.available);
        setScopeNotice(sessResp.scope_notice);
        setSessions(sessResp.sessions);
        if (sessResp.sessions.length > 0) {
          const first = sessResp.sessions[0].session_id;
          setRefSessionId(first);
          setQuerySessionId(first);
        }
        if (histResp) setHistorical(histResp);
      } catch (err: any) {
        if (mounted) setError(err.message || 'Failed to load ETH ASL Lounge sessions.');
      } finally {
        if (mounted) setLoadingSessions(false);
      }
    };
    init();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!available || !refSessionId) return;
    api
      .listEthFrames(refSessionId, 0, 200, refStride)
      .then((res) => {
        setRefFrames(res.frames);
        setRefTotal(res.total_frames);
        if (res.frames.length > 0 && !res.frames.some((f) => f.filename === refFrame)) {
          setRefFrame(res.frames[0].filename);
        }
      })
      .catch((err) => setError(err.message));
  }, [available, refSessionId, refStride]);

  useEffect(() => {
    if (!available || !querySessionId) return;
    api
      .listEthFrames(querySessionId, 0, 200, queryStride)
      .then((res) => {
        setQueryFrames(res.frames);
        setQueryTotal(res.total_frames);
        if (res.frames.length > 0 && !res.frames.some((f) => f.filename === queryFrame)) {
          setQueryFrame(res.frames[Math.min(1, res.frames.length - 1)].filename);
        }
      })
      .catch((err) => setError(err.message));
  }, [available, querySessionId, queryStride]);

  const handleRunRegistration = async (override?: {
    refSession: string;
    refF: string;
    qSession: string;
    qF: string;
  }) => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.registerEthPair({
        ref_session_id: override ? override.refSession : refSessionId,
        ref_frame: override ? override.refF : refFrame,
        query_session_id: override ? override.qSession : querySessionId,
        query_frame: override ? override.qF : queryFrame,
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Registration request failed.');
    } finally {
      setRunning(false);
    }
  };

  const applyPreset = (
    refS: string,
    refF: string,
    qS: string,
    qF: string,
    strideVal: number
  ) => {
    setRefSessionId(refS);
    setQuerySessionId(qS);
    setRefStride(strideVal);
    setQueryStride(strideVal);
    setRefFrame(refF);
    setQueryFrame(qF);
    handleRunRegistration({ refSession: refS, refF, qSession: qS, qF });
  };

  const activeRefUrl = available && refSessionId && refFrame
    ? api.getEthFrameUrl(refSessionId, refFrame)
    : null;
  const activeQueryUrl = available && querySessionId && queryFrame
    ? api.getEthFrameUrl(querySessionId, queryFrame)
    : null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 animate-in fade-in duration-150">
      {/* Header & Scientific Boundary Notice */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-6 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <ScanSearch className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-neutral-100 flex flex-wrap items-center gap-2">
                ETH ASL Lounge — Planar Registration Evaluator
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-950/60 text-amber-300 border border-amber-700/50 uppercase">
                  Supplementary Registration Only
                </span>
              </h1>
              <p className="text-xs text-neutral-400 mt-0.5">
                Tests frozen 2D PlanarRegistrar (SIFT/ORB + RANSAC) under 3D room viewpoint changes without altering production thresholds.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={!available || running}
              onClick={() => applyPreset('observation_01', 'frame_00000.png', 'observation_01', 'frame_00001.png', 1)}
              className="px-3 py-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-xs font-mono text-neutral-200 border border-neutral-700 disabled:opacity-40 cursor-pointer"
            >
              Preset: Adjacent (0 → 1)
            </button>
            <button
              type="button"
              disabled={!available || running}
              onClick={() => applyPreset('observation_01', 'frame_00000.png', 'observation_01', 'frame_00150.png', 50)}
              className="px-3 py-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-xs font-mono text-neutral-200 border border-neutral-700 disabled:opacity-40 cursor-pointer"
            >
              Preset: 150-Frame Gap (0 → 150)
            </button>
            <button
              type="button"
              disabled={!available || running}
              onClick={() => applyPreset('observation_01', 'frame_00000.png', 'observation_02', 'frame_00000.png', 1)}
              className="px-3 py-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-xs font-mono text-neutral-200 border border-neutral-700 disabled:opacity-40 cursor-pointer"
            >
              Preset: Cross-Obs (01 → 02)
            </button>
          </div>
        </div>

        <div className="bg-amber-950/30 border border-amber-800/40 rounded-lg p-3.5 flex items-start gap-3 text-xs text-amber-200/90">
          <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-amber-300">Scientific Scope Boundary: </span>
            {scopeNotice ||
              'ETH ASL Lounge provides 3D meshes and camera poses only, with no 2D bounding boxes or object-lifecycle ground truth. Object state accuracy (such as the 89.07% YOLOv8m physical-pilot result) does NOT apply here.'}
          </div>
        </div>
      </div>

      {loadingSessions ? (
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-10 text-center text-xs font-mono text-neutral-400">
          Discovering ETH ASL Lounge sessions...
        </div>
      ) : !available ? (
        <div className="bg-neutral-900 border border-rose-900/50 rounded-xl p-8 text-center space-y-2">
          <AlertTriangle className="w-6 h-6 text-rose-400 mx-auto" />
          <p className="text-sm font-medium text-neutral-200">ETH ASL Lounge extracted frames not found locally</p>
          <p className="text-xs text-neutral-400 font-mono">
            Expected under data/eth_asl_lounge/extracted/observation_01..10/rgb/
          </p>
        </div>
      ) : (
        <>
          {/* Frame Pair Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Reference Frame Selector */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-semibold">
                  Reference Frame
                </h2>
                <span className="text-[11px] font-mono text-neutral-500">
                  {refTotal} frames in {refSessionId}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] text-neutral-400 mb-1">ETH Session</label>
                  <select
                    value={refSessionId}
                    onChange={(e) => setRefSessionId(e.target.value)}
                    className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-2.5 py-2 text-xs font-mono text-neutral-200"
                  >
                    {sessions.map((s) => (
                      <option key={s.session_id} value={s.session_id}>
                        {s.session_id} ({s.frame_count} frames)
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] text-neutral-400 mb-1">Browse Stride</label>
                  <select
                    value={refStride}
                    onChange={(e) => setRefStride(Number(e.target.value))}
                    className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-2.5 py-2 text-xs font-mono text-neutral-200"
                  >
                    <option value={1}>Every 1 frame (first 200)</option>
                    <option value={10}>Every 10 frames</option>
                    <option value={25}>Every 25 frames</option>
                    <option value={50}>Every 50 frames</option>
                    <option value={150}>Every 150 frames</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-[11px] text-neutral-400 mb-1">Reference RGB Frame</label>
                <select
                  value={refFrame}
                  onChange={(e) => setRefFrame(e.target.value)}
                  className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-2.5 py-2 text-xs font-mono text-neutral-200"
                >
                  {refFrames.map((f) => (
                    <option key={f.filename} value={f.filename}>
                      #{f.index} — {f.filename}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Query Frame Selector */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-xs font-mono uppercase tracking-wider text-indigo-400 font-semibold">
                  Query Frame
                </h2>
                <span className="text-[11px] font-mono text-neutral-500">
                  {queryTotal} frames in {querySessionId}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] text-neutral-400 mb-1">ETH Session</label>
                  <select
                    value={querySessionId}
                    onChange={(e) => setQuerySessionId(e.target.value)}
                    className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-2.5 py-2 text-xs font-mono text-neutral-200"
                  >
                    {sessions.map((s) => (
                      <option key={s.session_id} value={s.session_id}>
                        {s.session_id} ({s.frame_count} frames)
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] text-neutral-400 mb-1">Browse Stride</label>
                  <select
                    value={queryStride}
                    onChange={(e) => setQueryStride(Number(e.target.value))}
                    className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-2.5 py-2 text-xs font-mono text-neutral-200"
                  >
                    <option value={1}>Every 1 frame (first 200)</option>
                    <option value={10}>Every 10 frames</option>
                    <option value={25}>Every 25 frames</option>
                    <option value={50}>Every 50 frames</option>
                    <option value={150}>Every 150 frames</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-[11px] text-neutral-400 mb-1">Query RGB Frame</label>
                <select
                  value={queryFrame}
                  onChange={(e) => setQueryFrame(e.target.value)}
                  className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-2.5 py-2 text-xs font-mono text-neutral-200"
                >
                  {queryFrames.map((f) => (
                    <option key={f.filename} value={f.filename}>
                      #{f.index} — {f.filename}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Run Button */}
          <div className="flex items-center justify-between bg-neutral-900 border border-neutral-800 rounded-xl px-5 py-3.5">
            <div className="text-xs font-mono text-neutral-400">
              Pair: <span className="text-neutral-200">{refSessionId}/{refFrame}</span> →{' '}
              <span className="text-neutral-200">{querySessionId}/{queryFrame}</span>
            </div>
            <button
              type="button"
              disabled={running}
              onClick={() => handleRunRegistration()}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold text-white transition disabled:opacity-50 cursor-pointer"
            >
              {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {running ? 'Running PlanarRegistrar...' : 'Run Planar Registration'}
            </button>
          </div>

          {error && (
            <div className="bg-rose-950/40 border border-rose-800/60 rounded-xl p-4 text-xs text-rose-300 font-mono">
              {error}
            </div>
          )}

          {/* Selected Frame Previews & Aligned Result */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 space-y-2">
              <div className="text-xs font-mono text-cyan-400">
                Reference: {refSessionId}/{refFrame}
              </div>
              {activeRefUrl && (
                <img
                  src={activeRefUrl}
                  alt="ETH Reference Frame"
                  className="w-full rounded-lg border border-neutral-800 bg-black object-contain"
                />
              )}
            </div>

            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 space-y-2">
              <div className="text-xs font-mono text-indigo-400">
                Query: {querySessionId}/{queryFrame}
              </div>
              {activeQueryUrl && (
                <img
                  src={activeQueryUrl}
                  alt="ETH Query Frame"
                  className="w-full rounded-lg border border-neutral-800 bg-black object-contain"
                />
              )}
            </div>

            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 space-y-2">
              <div className="text-xs font-mono text-emerald-400">
                Aligned Output (Warped to Reference)
              </div>
              {result?.aligned_image_data_url ? (
                <img
                  src={result.aligned_image_data_url}
                  alt="Aligned Query Frame"
                  className="w-full rounded-lg border border-neutral-800 bg-black object-contain"
                />
              ) : (
                <div className="h-48 rounded-lg border border-dashed border-neutral-800 flex items-center justify-center text-center p-4 text-xs font-mono text-neutral-500">
                  {result
                    ? `No aligned warp produced (${result.status})`
                    : 'Run registration to inspect aligned warp when available'}
                </div>
              )}
            </div>
          </div>

          {/* Registration Outcome & Measured Metrics */}
          {result && (
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2.5">
                  {result.success ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-rose-400" />
                  )}
                  <span className="text-sm font-bold text-neutral-100">
                    Registration Status: {result.status}
                  </span>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-neutral-800 text-neutral-300">
                    Method: {result.method_used}
                  </span>
                </div>
                <span className="text-xs font-mono text-neutral-400">
                  Latency: {result.latency_ms.toFixed(1)} ms
                </span>
              </div>

              {result.unreliable_warning && (
                <div className="bg-rose-950/40 border border-rose-800/60 rounded-lg p-3.5 flex items-start gap-2.5 text-xs text-rose-200">
                  <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{result.unreliable_warning}</span>
                </div>
              )}

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
                <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3">
                  <div className="text-[10px] font-mono text-neutral-500 uppercase">Inliers</div>
                  <div className="text-sm font-mono font-bold text-neutral-100 mt-1">
                    {result.num_inliers}
                  </div>
                </div>
                <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3">
                  <div className="text-[10px] font-mono text-neutral-500 uppercase">Inlier Ratio</div>
                  <div className="text-sm font-mono font-bold text-neutral-100 mt-1">
                    {result.inlier_ratio !== null && result.inlier_ratio !== undefined
                      ? `${(result.inlier_ratio * 100).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                </div>
                <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3">
                  <div className="text-[10px] font-mono text-neutral-500 uppercase">Mean Reproj Err</div>
                  <div className="text-sm font-mono font-bold text-neutral-100 mt-1">
                    {result.mean_reprojection_error !== null && result.mean_reprojection_error !== undefined
                      ? `${result.mean_reprojection_error.toFixed(2)} px`
                      : 'N/A'}
                  </div>
                </div>
                <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3">
                  <div className="text-[10px] font-mono text-neutral-500 uppercase">Overlap Fraction</div>
                  <div className="text-sm font-mono font-bold text-neutral-100 mt-1">
                    {result.overlap_area_fraction !== null && result.overlap_area_fraction !== undefined
                      ? `${(result.overlap_area_fraction * 100).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                </div>
                <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3">
                  <div className="text-[10px] font-mono text-neutral-500 uppercase">Tentative Matches</div>
                  <div className="text-sm font-mono font-bold text-neutral-100 mt-1">
                    {result.num_tentative_matches}
                  </div>
                </div>
                <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3">
                  <div className="text-[10px] font-mono text-neutral-500 uppercase">Keypoints (R / Q)</div>
                  <div className="text-sm font-mono font-bold text-neutral-100 mt-1">
                    {result.num_keypoints_ref} / {result.num_keypoints_query}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Historical ETH Baseline Summary */}
          {historical?.available && (
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-6 space-y-3">
              <h2 className="text-sm font-semibold text-neutral-200">
                Historical ETH ASL Lounge Planar Registration Summary (Frozen Baseline)
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="text-neutral-500 border-b border-neutral-800">
                    <tr>
                      <th className="py-2 pr-4">Experiment</th>
                      <th className="py-2 pr-4">Frames</th>
                      <th className="py-2 pr-4">Succeeded / Attempted</th>
                      <th className="py-2 pr-4">Success Rate</th>
                      <th className="py-2">Object Lifecycle Metrics</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-800/60 text-neutral-300">
                    {Object.entries(historical.experiments).map(([name, exp]) => (
                      <tr key={name}>
                        <td className="py-2 pr-4 text-cyan-300">{name}</td>
                        <td className="py-2 pr-4">{exp.frames_evaluated}</td>
                        <td className="py-2 pr-4">
                          {exp.transitions_succeeded} / {exp.transitions_attempted}
                        </td>
                        <td className="py-2 pr-4">
                          {(exp.registration_success_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 text-amber-300/90">
                          UNAVAILABLE (No 2D Object Lifecycle GT)
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
