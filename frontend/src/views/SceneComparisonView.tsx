import React, { useState, useEffect } from 'react';
import type { WorkspaceSummary, SessionSummary, ComparisonResponse } from '../types/api';
import { api } from '../api/client';
import { StatusBadge } from '../components/StatusBadge';
import { BoundingBoxCanvas } from '../components/BoundingBoxCanvas';
import type { OverlayBox, DisplacementVector } from '../components/BoundingBoxCanvas';
import { ComparisonSlider } from '../components/ComparisonSlider';
import { Tooltip } from '../components/Tooltip';
import {
  GitCompare,
  ArrowRight,
  RefreshCw,
  Loader2,
  AlertTriangle,
  Info,
  Sliders,
  Columns,
  ArrowUpRight,
  Lock,
  Unlock,
} from 'lucide-react';

interface SceneComparisonViewProps {
  workspace: WorkspaceSummary;
}

export const SceneComparisonView: React.FC<SceneComparisonViewProps> = ({ workspace }) => {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [refSessionId, setRefSessionId] = useState<string>('');
  const [querySessionId, setQuerySessionId] = useState<string>('');

  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [viewMode, setViewMode] = useState<'side-by-side' | 'slider'>('side-by-side');
  const [eventFilter, setEventFilter] = useState<string>('ALL');
  const [highlightedEntityId, setHighlightedEntityId] = useState<string | null>(null);

  // Synchronized Pan & Zoom state for dual canvases
  const [isSyncLocked, setIsSyncLocked] = useState(true);
  const [sharedZoom, setSharedZoom] = useState(1.0);
  const [sharedPan, setSharedPan] = useState({ x: 0, y: 0 });

  // Load session list on mount
  useEffect(() => {
    const load = async () => {
      try {
        const sList = await api.listObservations(workspace.workspace_id);
        setSessions(sList);
        if (sList.length >= 2) {
          setRefSessionId(sList[0].session_id);
          setQuerySessionId(sList[sList.length - 1].session_id);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load sessions for comparison.');
      }
    };
    load();
  }, [workspace.workspace_id]);

  const fetchComparison = async () => {
    if (!refSessionId || !querySessionId) return;
    if (refSessionId === querySessionId) {
      setError('Select two distinct sessions to perform a temporal comparison.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const comp = await api.compareSessions(workspace.workspace_id, refSessionId, querySessionId);
      setComparison(comp);
    } catch (err: any) {
      setError(err.detail || err.message || 'Comparison failed.');
      setComparison(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (refSessionId && querySessionId && refSessionId !== querySessionId) {
      fetchComparison();
    }
  }, [refSessionId, querySessionId]);

  const refSessionObj = sessions.find((s) => s.session_id === refSessionId);
  const querySessionObj = sessions.find((s) => s.session_id === querySessionId);

  // Filter comparison items
  const filteredItems = comparison
    ? comparison.items.filter((item) => {
        if (eventFilter === 'ALL') return true;
        if (eventFilter === 'MOVED') return item.status_change === 'CONFIRMED_MOVED' || item.status_change === 'MOVED';
        if (eventFilter === 'ADDED') return item.status_change === 'CONFIRMED_ADDED' || item.status_change === 'ADDED';
        if (eventFilter === 'REMOVED')
          return (
            item.status_change === 'CONFIRMED_REMOVED' ||
            item.status_change === 'PENDING_REMOVAL' ||
            item.status_change === 'MISSING_IN_QUERY' ||
            item.status_change === 'REMOVED'
          );
        if (eventFilter === 'UNCHANGED') return item.status_change === 'UNCHANGED';
        return true;
      })
    : [];

  const refBoxes: OverlayBox[] = filteredItems
    .filter((it) => it.ref_bbox != null)
    .map((it) => ({
      id: it.entity_id,
      label: `${it.label} (${it.entity_id})`,
      bbox: it.ref_bbox,
      status: it.prior_state,
    }));

  const queryBoxes: OverlayBox[] = filteredItems
    .filter((it) => it.query_bbox != null)
    .map((it) => ({
      id: it.entity_id,
      label: `${it.label} (${it.entity_id})`,
      bbox: it.query_bbox,
      status: it.new_state,
    }));

  // Build displacement vectors for items with both bboxes and non-zero displacement
  const displacementVectors: DisplacementVector[] = filteredItems
    .filter(
      (it) =>
        it.ref_bbox != null &&
        it.query_bbox != null &&
        it.displacement_px != null &&
        it.displacement_px > 0.5
    )
    .map((it) => {
      const refB = it.ref_bbox!;
      const qB = it.query_bbox!;
      const startX = (refB.x_min + refB.x_max) / 2;
      const startY = (refB.y_min + refB.y_max) / 2;
      const endX = (qB.x_min + qB.x_max) / 2;
      const endY = (qB.y_min + qB.y_max) / 2;
      return {
        id: it.entity_id,
        startX,
        startY,
        endX,
        endY,
        distancePx: it.displacement_px,
        label: `${it.label}: ${it.displacement_px?.toFixed(1)}px`,
      };
    });

  const isRegistrationDegraded = comparison && comparison.registration_inlier_ratio < 0.20;

  const handleSharedZoomPanChange = (newZoom: number, newPan: { x: number; y: number }) => {
    setSharedZoom(newZoom);
    setSharedPan(newPan);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 animate-in fade-in duration-150">
      {/* Header & Controls */}
      <div className="bg-[#11141D] border border-[#232A3B] rounded-xl p-5 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-[#06B6D4]/10 border border-[#06B6D4]/20 text-[#22D3EE]">
              <GitCompare className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-[#F8FAFC] flex items-center gap-2">
                Temporal Scene Comparison
                <Tooltip content="Compares localized bounding boxes projected into a shared canonical coordinate space using planar SIFT homography." />
              </h1>
              <p className="text-xs text-[#94A3B8]">
                Inspect physical displacements and state transitions between any two observation visits.
              </p>
            </div>
          </div>

          {/* View mode toggle & sync lock */}
          <div className="flex items-center gap-2 self-start md:self-auto">
            {viewMode === 'side-by-side' && (
              <button
                type="button"
                onClick={() => setIsSyncLocked(!isSyncLocked)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-medium border transition cursor-pointer ${
                  isSyncLocked
                    ? 'bg-[#181D29] text-[#22D3EE] border-[#06B6D4]/40'
                    : 'bg-[#090B10] text-[#94A3B8] border-[#232A3B] hover:text-[#F8FAFC]'
                }`}
                title="Synchronize pan and zoom across dual reference and query canvases"
              >
                {isSyncLocked ? <Lock className="w-3.5 h-3.5 text-[#22D3EE]" /> : <Unlock className="w-3.5 h-3.5 text-[#94A3B8]" />}
                <span>{isSyncLocked ? 'Pan/Zoom: Synced' : 'Pan/Zoom: Independent'}</span>
              </button>
            )}

            <div className="flex items-center gap-1 bg-[#090B10] p-1 rounded-lg border border-[#232A3B]">
              <button
                type="button"
                onClick={() => setViewMode('side-by-side')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition cursor-pointer ${
                  viewMode === 'side-by-side'
                    ? 'bg-[#06B6D4] text-[#090B10] font-bold'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC]'
                }`}
              >
                <Columns className="w-3.5 h-3.5" />
                Side-by-Side
              </button>
              <button
                type="button"
                onClick={() => setViewMode('slider')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition cursor-pointer ${
                  viewMode === 'slider'
                    ? 'bg-[#06B6D4] text-[#090B10] font-bold'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC]'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                Wipe Slider
              </button>
            </div>
          </div>
        </div>

        {/* Observation Pickers & Filter Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-[#232A3B]">
          {sessions.length >= 2 ? (
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 bg-[#090B10] px-3 py-1.5 rounded-lg border border-[#232A3B]">
                <span className="text-[10px] uppercase font-mono text-[#94A3B8]">Reference (T1):</span>
                <select
                  value={refSessionId}
                  onChange={(e) => setRefSessionId(e.target.value)}
                  className="bg-transparent text-xs text-[#F8FAFC] font-mono focus:outline-hidden cursor-pointer"
                >
                  {sessions.map((s) => (
                    <option key={s.session_id} value={s.session_id} className="bg-[#11141D] text-[#F8FAFC]">
                      {s.session_name} ({s.session_id})
                    </option>
                  ))}
                </select>
              </div>

              <ArrowRight className="w-4 h-4 text-[#64748B] shrink-0" />

              <div className="flex items-center gap-2 bg-[#090B10] px-3 py-1.5 rounded-lg border border-[#232A3B]">
                <span className="text-[10px] uppercase font-mono text-[#22D3EE]">Query (T2):</span>
                <select
                  value={querySessionId}
                  onChange={(e) => setQuerySessionId(e.target.value)}
                  className="bg-transparent text-xs text-[#F8FAFC] font-mono focus:outline-hidden cursor-pointer"
                >
                  {sessions.map((s) => (
                    <option key={s.session_id} value={s.session_id} className="bg-[#11141D] text-[#F8FAFC]">
                      {s.session_name} ({s.session_id})
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="button"
                onClick={fetchComparison}
                disabled={loading}
                className="p-1.5 rounded hover:bg-[#181D29] text-[#94A3B8] hover:text-[#F8FAFC] border border-[#232A3B] transition cursor-pointer"
                title="Re-run comparison"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-[#22D3EE]' : ''}`} />
              </button>
            </div>
          ) : (
            <div className="text-xs text-[#94A3B8] font-mono">
              Requires at least 2 recorded visits to compare (current: {sessions.length}).
            </div>
          )}

          {/* Event Filter Pills */}
          {comparison && (
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-[10px] uppercase font-mono text-[#64748B] mr-1">Filter:</span>
              {(['ALL', 'MOVED', 'ADDED', 'REMOVED', 'UNCHANGED'] as const).map((filter) => (
                <button
                  key={filter}
                  type="button"
                  onClick={() => setEventFilter(filter)}
                  className={`px-2.5 py-1 rounded text-[11px] font-mono transition cursor-pointer ${
                    eventFilter === filter
                      ? 'bg-[#181D29] text-[#22D3EE] font-bold border border-[#06B6D4]/40'
                      : 'bg-[#090B10] text-[#94A3B8] hover:text-[#F8FAFC] border border-[#232A3B]'
                  }`}
                >
                  {filter}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Prominent Registration Gating Warning Banner */}
      {isRegistrationDegraded && (
        <div className="p-4 bg-amber-950/40 border border-amber-800/80 rounded-xl flex items-start gap-3 text-xs text-amber-200 animate-in fade-in duration-150">
          <AlertTriangle className="w-5 h-5 shrink-0 text-amber-400 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold text-amber-300 block">
              Warning: Planar Registration Reliability Degraded
            </span>
            <p className="text-amber-300/80 leading-relaxed">
              SIFT keypoint inlier ratio is{' '}
              <strong className="text-amber-200">
                {(comparison.registration_inlier_ratio * 100).toFixed(1)}%
              </strong>{' '}
              (below the 20% safety threshold). Coordinate projections cannot be certified reliable.
              Spatial conclusions and displacements below 100px have been conservatively suppressed to
              prevent false movement classification.
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="p-4 bg-rose-950/50 border border-rose-800 rounded-xl text-xs text-rose-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="py-24 flex flex-col items-center justify-center gap-3 text-[#94A3B8]">
          <Loader2 className="w-8 h-8 animate-spin text-[#06B6D4]" />
          <span className="text-xs font-mono">Computing planar homography & temporal alignment...</span>
        </div>
      ) : sessions.length < 2 ? (
        <div className="py-16 text-center text-[#94A3B8] bg-[#11141D] border border-[#232A3B] rounded-xl">
          <Info className="w-8 h-8 mx-auto mb-2 text-[#64748B]" />
          <p className="text-sm font-medium text-[#F8FAFC]">Need At Least Two Observations</p>
          <p className="text-xs text-[#94A3B8] mt-1">
            Upload another visit image in the Overview tab to enable comparative analysis.
          </p>
        </div>
      ) : comparison && refSessionObj && querySessionObj ? (
        <div className="space-y-6">
          {/* Render selected view mode */}
          {viewMode === 'side-by-side' ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <BoundingBoxCanvas
                imageUrl={refSessionObj.image_url}
                boxes={refBoxes}
                vectors={displacementVectors}
                title={`Reference (T1): ${refSessionObj.session_name}`}
                subtitle={refSessionObj.session_id}
                theme="reference"
                highlightedId={highlightedEntityId}
                onBoxSelect={(id) => setHighlightedEntityId(id)}
                zoom={isSyncLocked ? sharedZoom : undefined}
                pan={isSyncLocked ? sharedPan : undefined}
                onZoomPanChange={isSyncLocked ? handleSharedZoomPanChange : undefined}
              />

              <BoundingBoxCanvas
                imageUrl={querySessionObj.image_url}
                boxes={queryBoxes}
                vectors={displacementVectors}
                title={`Query (T2): ${querySessionObj.session_name}`}
                subtitle={`${querySessionObj.session_id} • Inliers: ${(comparison.registration_inlier_ratio * 100).toFixed(0)}%`}
                theme="query"
                highlightedId={highlightedEntityId}
                onBoxSelect={(id) => setHighlightedEntityId(id)}
                zoom={isSyncLocked ? sharedZoom : undefined}
                pan={isSyncLocked ? sharedPan : undefined}
                onZoomPanChange={isSyncLocked ? handleSharedZoomPanChange : undefined}
              />
            </div>
          ) : (
            <ComparisonSlider
              beforeImageUrl={refSessionObj.image_url}
              afterImageUrl={querySessionObj.image_url}
              beforeTitle={`T1 (Ref): ${refSessionObj.session_name}`}
              afterTitle={`T2 (Query): ${querySessionObj.session_name}`}
              beforeBoxes={refBoxes}
              afterBoxes={queryBoxes}
              highlightedId={highlightedEntityId}
              onBoxSelect={(id) => setHighlightedEntityId(id)}
            />
          )}

          {/* Vector Geometry Disclaimer */}
          <div className="px-4 py-2.5 bg-[#11141D]/60 border border-[#232A3B] rounded-lg flex items-center justify-between text-[11px] font-mono text-[#94A3B8]">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              <span>
                Displacement vectors depict 2D centroid shifts on the canonical workbench plane. Does NOT reconstruct 3D geometry.
              </span>
            </div>
            {displacementVectors.length > 0 && (
              <span className="text-[#22D3EE] font-bold">
                {displacementVectors.length} active displacement vector{displacementVectors.length === 1 ? '' : 's'}
              </span>
            )}
          </div>

          {/* Change Table with Displacement Vector Details */}
          <div className="bg-[#11141D] border border-[#232A3B] rounded-xl overflow-hidden shadow-sm">
            <div className="px-6 py-3.5 border-b border-[#232A3B] flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-[#94A3B8] font-semibold flex items-center gap-2">
                Comparative Entity Changes ({filteredItems.length} of {comparison.items.length})
                <Tooltip content="Spatial distances are computed in canonical pixels between reference baseline bounding box centroid and query bounding box centroid." />
              </span>
              <span className="text-xs font-mono text-[#64748B]">
                Hover or click row to inspect bounding box
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#090B10] font-mono uppercase text-[10px] text-[#94A3B8] border-b border-[#232A3B]">
                  <tr>
                    <th className="px-5 py-3">Entity</th>
                    <th className="px-5 py-3">Class</th>
                    <th className="px-5 py-3">Status Change</th>
                    <th className="px-5 py-3">Prior State</th>
                    <th className="px-5 py-3">New State</th>
                    <th className="px-5 py-3">Displacement</th>
                    <th className="px-5 py-3">Physical Evidence Explanation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#232A3B] font-mono">
                  {filteredItems.map((item) => {
                    const isSelected = highlightedEntityId === item.entity_id;
                    return (
                      <tr
                        key={item.entity_id}
                        onMouseEnter={() => setHighlightedEntityId(item.entity_id)}
                        onMouseLeave={() => setHighlightedEntityId(null)}
                        onClick={() =>
                          setHighlightedEntityId((prev) => (prev === item.entity_id ? null : item.entity_id))
                        }
                        className={`cursor-pointer transition ${
                          isSelected ? 'bg-[#181D29] text-[#F8FAFC]' : 'hover:bg-[#181D29]/40 text-[#94A3B8]'
                        }`}
                      >
                        <td className="px-5 py-3 font-bold text-[#F8FAFC] flex items-center gap-1.5">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              isSelected ? 'bg-[#22D3EE]' : 'bg-[#64748B]'
                            }`}
                          />
                          {item.entity_id}
                        </td>
                        <td className="px-5 py-3 text-[#F8FAFC] font-sans">{item.label}</td>
                        <td className="px-5 py-3">
                          <StatusBadge status={item.status_change} size="sm" />
                        </td>
                        <td className="px-5 py-3 text-[#94A3B8]">{item.prior_state}</td>
                        <td className="px-5 py-3 text-[#F8FAFC]">{item.new_state}</td>
                        <td className="px-5 py-3">
                          {item.displacement_px != null ? (
                            <span className="text-[#22D3EE] font-bold inline-flex items-center gap-1">
                              <ArrowUpRight className="w-3.5 h-3.5" />
                              {item.displacement_px.toFixed(1)} px
                            </span>
                          ) : (
                            <span className="text-[#64748B]">—</span>
                          )}
                        </td>
                        <td className="px-5 py-3 text-[#F8FAFC] font-sans text-xs max-w-md leading-relaxed">
                          {item.explanation}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
