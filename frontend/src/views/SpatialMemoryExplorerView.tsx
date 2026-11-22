import React, { useState, useEffect, useMemo } from 'react';
import type { WorkspaceSummary, EntitySummary, EntityDetailResponse } from '../types/api';
import { api } from '../api/client';
import { StatusBadge } from '../components/StatusBadge';
import {
  Compass,
  RotateCcw,
  Layers,
  AlertTriangle,
  Move,
  Search,
} from 'lucide-react';

interface SpatialMemoryExplorerViewProps {
  workspace: WorkspaceSummary;
}

interface EntitySpatialData {
  entity: EntitySummary;
  initialBbox?: { x_min: number; y_min: number; x_max: number; y_max: number };
  latestBbox?: { x_min: number; y_min: number; x_max: number; y_max: number };
  displacementPx?: number;
  dx?: number;
  dy?: number;
  isClipped?: boolean;
}

export const SpatialMemoryExplorerView: React.FC<SpatialMemoryExplorerViewProps> = ({ workspace }) => {
  const [entities, setEntities] = useState<EntitySummary[]>([]);
  const [spatialData, setSpatialData] = useState<Map<string, EntitySpatialData>>(new Map());
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedEntityDetail, setSelectedEntityDetail] = useState<EntityDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);

  // View Options
  const [isIsometric, setIsIsometric] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showVectors, setShowVectors] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [stateFilter, setStateFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [zoomLevel, setZoomLevel] = useState(1.0);

  // Workbench reference dimensions (canonical space)
  const [refWidth, setRefWidth] = useState(1920);
  const [refHeight, setRefHeight] = useState(1080);

  useEffect(() => {
    let mounted = true;
    const loadData = async () => {
      setLoading(true);
      try {
        const [entityList, diag] = await Promise.all([
          api.listEntities(workspace.workspace_id),
          api.getDiagnostics(workspace.workspace_id).catch(() => null),
        ]);

        if (!mounted) return;

        setEntities(entityList);

        // Derive ref dimensions if available in diagnostics
        if (diag && diag.registration_health && diag.registration_health.ref_size) {
          const sz = diag.registration_health.ref_size;
          if (sz.length >= 2 && sz[0] > 0 && sz[1] > 0) {
            setRefWidth(sz[0]);
            setRefHeight(sz[1]);
          }
        }

        // Fetch details for entities to resolve initial and latest canonical bboxes
        const sMap = new Map<string, EntitySpatialData>();
        const detailPromises = entityList.map(async (ent) => {
          try {
            const detail = await api.getEntityDetail(workspace.workspace_id, ent.entity_id);
            const obsWithBbox = detail.observations.filter((o) => o.canonical_bbox !== null && o.canonical_bbox !== undefined);
            if (obsWithBbox.length > 0) {
              const firstObs = obsWithBbox[0];
              const lastObs = obsWithBbox[obsWithBbox.length - 1];

              const b0 = firstObs.canonical_bbox!;
              const b1 = lastObs.canonical_bbox!;

              const c0x = (b0.x_min + b0.x_max) / 2;
              const c0y = (b0.y_min + b0.y_max) / 2;
              const c1x = (b1.x_min + b1.x_max) / 2;
              const c1y = (b1.y_min + b1.y_max) / 2;

              const dx = c1x - c0x;
              const dy = c1y - c0y;
              const dist = Math.hypot(dx, dy);

              // Check if clipped (within 15px of boundary)
              const isClipped =
                b1.x_min <= 15 || b1.y_min <= 15 || b1.x_max >= refWidth - 15 || b1.y_max >= refHeight - 15;

              sMap.set(ent.entity_id, {
                entity: ent,
                initialBbox: b0,
                latestBbox: b1,
                displacementPx: dist > 1.0 ? dist : undefined,
                dx,
                dy,
                isClipped,
              });
            } else if (ent.canonical_bbox) {
              sMap.set(ent.entity_id, {
                entity: ent,
                latestBbox: ent.canonical_bbox,
              });
            }
          } catch (e) {
            console.error(`Failed to load detail for ${ent.entity_id}`, e);
          }
        });

        await Promise.all(detailPromises);
        if (mounted) {
          setSpatialData(sMap);
        }
      } catch (err) {
        console.error('Failed to load spatial data:', err);
      } finally {
        if (mounted) setLoading(false);
      }
    };

    loadData();
    return () => {
      mounted = false;
    };
  }, [workspace.workspace_id]);

  // Load detail when entity is selected
  useEffect(() => {
    if (!selectedEntityId) {
      setSelectedEntityDetail(null);
      return;
    }
    let mounted = true;
    setLoadingDetail(true);
    api
      .getEntityDetail(workspace.workspace_id, selectedEntityId)
      .then((detail) => {
        if (mounted) setSelectedEntityDetail(detail);
      })
      .catch((e) => {
        console.error('Error fetching entity detail:', e);
      })
      .finally(() => {
        if (mounted) setLoadingDetail(false);
      });
    return () => {
      mounted = false;
    };
  }, [selectedEntityId, workspace.workspace_id]);

  // Filter entities
  const filteredEntities = useMemo(() => {
    return entities.filter((ent) => {
      if (stateFilter !== 'ALL' && ent.current_state !== stateFilter) return false;
      if (
        searchQuery &&
        !ent.label.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !ent.entity_id.toLowerCase().includes(searchQuery.toLowerCase())
      ) {
        return false;
      }
      return true;
    });
  }, [entities, stateFilter, searchQuery]);

  const selectedSpatial = selectedEntityId ? spatialData.get(selectedEntityId) : undefined;

  // State styling helper
  const getStateColors = (st: string) => {
    switch (st) {
      case 'CONFIRMED_PRESENT':
        return {
          stroke: '#10b981', // emerald-500
          fill: 'rgba(16, 185, 129, 0.15)',
          badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
        };
      case 'CONFIRMED_MOVED':
        return {
          stroke: '#f59e0b', // amber-500
          fill: 'rgba(245, 158, 11, 0.22)',
          badge: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
        };
      case 'PENDING_REMOVAL':
        return {
          stroke: '#f43f5e', // rose-500
          fill: 'rgba(244, 63, 94, 0.15)',
          badge: 'bg-rose-500/20 text-rose-400 border-rose-500/30',
        };
      case 'CONFIRMED_ABSENT':
        return {
          stroke: '#64748b', // slate-500
          fill: 'rgba(100, 116, 139, 0.1)',
          badge: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
        };
      case 'REAPPEARED':
        return {
          stroke: '#a855f7', // purple-500
          fill: 'rgba(168, 85, 247, 0.2)',
          badge: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
        };
      default:
        return {
          stroke: '#3b82f6', // blue-500
          fill: 'rgba(59, 130, 246, 0.15)',
          badge: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
        };
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-neutral-950 text-neutral-100">
      {/* Top Control Bar */}
      <div className="bg-neutral-900 border-b border-neutral-800 px-6 py-3 flex flex-wrap items-center justify-between gap-4 shrink-0">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-[#06B6D4]/10 border border-[#06B6D4]/20 text-[#22D3EE]">
            <Compass className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white flex items-center gap-2">
              Spatial Memory Explorer
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#181D29] text-[#94A3B8] border border-[#232A3B]">
                Canonical Space ({refWidth} × {refHeight} px)
              </span>
            </h1>
            <p className="text-xs text-[#94A3B8]">
              Canonical 2D Workbench Plane: Planar pixel coordinates derived from SIFT homography (no 3D reconstruction).
            </p>
          </div>
        </div>

        {/* View & Projection Controls */}
        <div className="flex items-center gap-2">
          {/* Planar Projection Inspection Toggle */}
          <button
            type="button"
            onClick={() => setIsIsometric(!isIsometric)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition cursor-pointer ${
              isIsometric
                ? 'bg-[#06B6D4]/20 border-[#06B6D4]/50 text-[#22D3EE]'
                : 'bg-[#181D29] border-[#232A3B] text-[#94A3B8] hover:text-[#F8FAFC]'
            }`}
            title="Planar inspection tilt of Canonical 2D Workbench Plane"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>{isIsometric ? 'Angled Inspection Plane' : 'Canonical 2D Workbench Plane'}</span>
          </button>

          {/* Grid Toggle */}
          <button
            type="button"
            onClick={() => setShowGrid(!showGrid)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-mono border transition ${
              showGrid
                ? 'bg-neutral-800 border-neutral-600 text-neutral-200'
                : 'bg-neutral-900 border-neutral-800 text-neutral-500'
            }`}
            title="Toggle workbench coordinate calibration grid"
          >
            Grid
          </button>

          {/* Vectors Toggle */}
          <button
            type="button"
            onClick={() => setShowVectors(!showVectors)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-mono border transition ${
              showVectors
                ? 'bg-orange-950/40 border-orange-700/50 text-orange-300'
                : 'bg-neutral-900 border-neutral-800 text-neutral-500'
            }`}
            title="Toggle displacement vector arrows"
          >
            Vectors
          </button>

          {/* Labels Toggle */}
          <button
            type="button"
            onClick={() => setShowLabels(!showLabels)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-mono border transition ${
              showLabels
                ? 'bg-neutral-800 border-neutral-600 text-neutral-200'
                : 'bg-neutral-900 border-neutral-800 text-neutral-500'
            }`}
            title="Toggle entity text labels"
          >
            Labels
          </button>

          {/* Zoom controls */}
          <div className="flex items-center border border-neutral-800 rounded-lg bg-neutral-900 overflow-hidden">
            <button
              type="button"
              onClick={() => setZoomLevel((z) => Math.max(0.6, z - 0.15))}
              className="px-2 py-1 hover:bg-neutral-800 text-neutral-400 text-xs font-mono"
              title="Zoom out"
            >
              -
            </button>
            <span className="px-2 text-[11px] font-mono text-neutral-300 min-w-[3rem] text-center">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              type="button"
              onClick={() => setZoomLevel((z) => Math.min(2.5, z + 0.15))}
              className="px-2 py-1 hover:bg-neutral-800 text-neutral-400 text-xs font-mono"
              title="Zoom in"
            >
              +
            </button>
            <button
              type="button"
              onClick={() => setZoomLevel(1.0)}
              className="px-2 py-1 hover:bg-neutral-800 text-neutral-400 text-xs"
              title="Reset Zoom"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Workspace Layout: Visualizer + Sidebar Drawer */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Interactive Canvas Area */}
        <div className="flex-1 relative flex flex-col overflow-hidden bg-neutral-950">
          {/* Sub-header Filter Toolbar */}
          <div className="bg-neutral-900/60 border-b border-neutral-800/60 px-6 py-2 flex items-center justify-between gap-4 z-10">
            {/* Epistemic State Filter Pills */}
            <div className="flex items-center gap-1.5 overflow-x-auto text-xs">
              {['ALL', 'CONFIRMED_PRESENT', 'CONFIRMED_MOVED', 'PENDING_REMOVAL', 'CONFIRMED_ABSENT'].map((st) => (
                <button
                  key={st}
                  type="button"
                  onClick={() => setStateFilter(st)}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition ${
                    stateFilter === st
                      ? 'bg-neutral-700 text-white font-semibold'
                      : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60'
                  }`}
                >
                  {st === 'ALL' ? 'All States' : st.replace('CONFIRMED_', '')}
                </button>
              ))}
            </div>

            {/* Quick Search */}
            <div className="relative w-48">
              <Search className="w-3.5 h-3.5 text-neutral-500 absolute left-2.5 top-2" />
              <input
                type="text"
                placeholder="Filter entities..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-neutral-900 border border-neutral-800 rounded-md pl-8 pr-2.5 py-1 text-xs text-neutral-200 placeholder-neutral-500 focus:outline-hidden focus:border-neutral-600 font-mono"
              />
            </div>
          </div>

          {/* Interactive Workbench Canvas Container */}
          <div className="flex-1 overflow-auto flex items-center justify-center p-8 bg-[radial-gradient(#1f2937_1px,transparent_1px)] [background-size:24px_24px]">
            {loading ? (
              <div className="flex flex-col items-center gap-2 text-neutral-400 font-mono text-xs">
                <div className="w-6 h-6 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
                <span>Projecting canonical workbench scene...</span>
              </div>
            ) : (
              <div
                style={{
                  transform: `scale(${zoomLevel}) ${
                    isIsometric ? 'perspective(1200px) rotateX(42deg) rotateZ(-12deg) skewX(-2deg)' : ''
                  }`,
                  transformOrigin: 'center center',
                  transition: 'transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1)',
                  boxShadow: isIsometric
                    ? '0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05)'
                    : '0 10px 25px -5px rgba(0, 0, 0, 0.6)',
                }}
                className="relative bg-neutral-900/90 border border-neutral-700/80 rounded-xl overflow-hidden shadow-2xl select-none"
              >
                {/* SVG Canonical Space Layer */}
                <svg
                  viewBox={`0 0 ${refWidth} ${refHeight}`}
                  width={refWidth * 0.45}
                  height={refHeight * 0.45}
                  className="block cursor-crosshair"
                >
                  <defs>
                    {/* Grid Pattern */}
                    <pattern id="workbenchGrid" width="100" height="100" patternUnits="userSpaceOnUse">
                      <path d="M 100 0 L 0 0 0 100" fill="none" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                    </pattern>
                    {/* Vector Arrowhead */}
                    <marker
                      id="vectorArrow"
                      viewBox="0 0 10 10"
                      refX="8"
                      refY="5"
                      markerWidth="6"
                      markerHeight="6"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 1 L 10 5 L 0 9 z" fill="#f59e0b" />
                    </marker>
                    {/* Striped fill pattern for absent entities */}
                    <pattern id="absentStripe" width="8" height="8" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
                      <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(148, 163, 184, 0.3)" strokeWidth="2" />
                    </pattern>
                  </defs>

                  {/* Calibration Grid Background */}
                  {showGrid && <rect width={refWidth} height={refHeight} fill="url(#workbenchGrid)" />}

                  {/* Workbench Border & Dimension markings */}
                  <rect
                    x="2"
                    y="2"
                    width={refWidth - 4}
                    height={refHeight - 4}
                    fill="none"
                    stroke="rgba(255, 255, 255, 0.15)"
                    strokeWidth="3"
                  />

                  {/* Coordinate ticks */}
                  {showGrid && (
                    <g fill="rgba(255, 255, 255, 0.3)" fontSize="18" fontFamily="monospace">
                      <text x="15" y="30">
                        (0,0) Reference Origin
                      </text>
                      <text x={refWidth - 220} y={refHeight - 15}>
                        ({refWidth}, {refHeight})
                      </text>
                    </g>
                  )}

                  {/* Render Displacement Vectors for Moved Entities */}
                  {showVectors &&
                    filteredEntities.map((ent) => {
                      const s = spatialData.get(ent.entity_id);
                      if (!s || !s.initialBbox || !s.latestBbox || !s.displacementPx) return null;
                      if (s.displacementPx < 5.0) return null;

                      const c0x = (s.initialBbox.x_min + s.initialBbox.x_max) / 2;
                      const c0y = (s.initialBbox.y_min + s.initialBbox.y_max) / 2;
                      const c1x = (s.latestBbox.x_min + s.latestBbox.x_max) / 2;
                      const c1y = (s.latestBbox.y_min + s.latestBbox.y_max) / 2;

                      return (
                        <g key={`vec-${ent.entity_id}`}>
                          {/* Ghost outline of initial position */}
                          <rect
                            x={s.initialBbox.x_min}
                            y={s.initialBbox.y_min}
                            width={s.initialBbox.x_max - s.initialBbox.x_min}
                            height={s.initialBbox.y_max - s.initialBbox.y_min}
                            fill="none"
                            stroke="#f59e0b"
                            strokeWidth="1.5"
                            strokeDasharray="4 4"
                            opacity="0.4"
                          />
                          {/* Vector Arrow Line */}
                          <line
                            x1={c0x}
                            y1={c0y}
                            x2={c1x}
                            y2={c1y}
                            stroke="#f59e0b"
                            strokeWidth="3"
                            strokeDasharray="6 3"
                            markerEnd="url(#vectorArrow)"
                          />
                          {/* Displacement label */}
                          <rect
                            x={(c0x + c1x) / 2 - 40}
                            y={(c0y + c1y) / 2 - 14}
                            width="80"
                            height="22"
                            rx="4"
                            fill="#1e1e1e"
                            stroke="#f59e0b"
                            strokeWidth="1"
                          />
                          <text
                            x={(c0x + c1x) / 2}
                            y={(c0y + c1y) / 2 + 1}
                            fill="#f59e0b"
                            fontSize="12"
                            fontFamily="monospace"
                            fontWeight="bold"
                            textAnchor="middle"
                          >
                            Δ {s.displacementPx.toFixed(1)}px
                          </text>
                        </g>
                      );
                    })}

                  {/* Render Entity Bounding Box Footprints */}
                  {filteredEntities.map((ent) => {
                    const s = spatialData.get(ent.entity_id);
                    const bbox = s?.latestBbox || ent.canonical_bbox;
                    if (!bbox) return null;

                    const width = Math.max(10, bbox.x_max - bbox.x_min);
                    const height = Math.max(10, bbox.y_max - bbox.y_min);
                    const isSelected = selectedEntityId === ent.entity_id;
                    const colors = getStateColors(ent.current_state);

                    return (
                      <g
                        key={`footprint-${ent.entity_id}`}
                        onClick={() => setSelectedEntityId(ent.entity_id)}
                        className="cursor-pointer transition-opacity"
                        style={{
                          opacity: selectedEntityId && !isSelected ? 0.35 : 1.0,
                        }}
                      >
                        {/* Interactive Object Tile */}
                        <rect
                          x={bbox.x_min}
                          y={bbox.y_min}
                          width={width}
                          height={height}
                          rx="6"
                          fill={ent.current_state === 'CONFIRMED_ABSENT' ? 'url(#absentStripe)' : colors.fill}
                          stroke={isSelected ? '#ffffff' : colors.stroke}
                          strokeWidth={isSelected ? '4' : '2'}
                          strokeDasharray={ent.current_state === 'PENDING_REMOVAL' ? '6 4' : undefined}
                          className="hover:stroke-white transition"
                        />

                        {/* Label Pill */}
                        {showLabels && (
                          <g transform={`translate(${bbox.x_min + 4}, ${bbox.y_min + 18})`}>
                            <rect
                              x="0"
                              y="-13"
                              width={Math.min(180, Math.max(80, ent.label.length * 9 + 40))}
                              height="20"
                              rx="4"
                              fill="rgba(15, 23, 42, 0.85)"
                              stroke={colors.stroke}
                              strokeWidth="1"
                            />
                            <text
                              x="8"
                              y="1"
                              fill="#f8fafc"
                              fontSize="11"
                              fontFamily="monospace"
                              fontWeight="bold"
                            >
                              {ent.label}
                            </text>
                          </g>
                        )}
                      </g>
                    );
                  })}
                </svg>

                {/* Floating Canvas Legend */}
                <div className="absolute bottom-3 left-3 bg-neutral-950/85 backdrop-blur-md border border-neutral-800 rounded-lg p-2.5 flex items-center gap-3 text-[10px] font-mono text-neutral-300">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-xs bg-emerald-500 inline-block" />
                    <span>Present</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-xs bg-amber-500 inline-block" />
                    <span>Moved</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-xs bg-rose-500 border border-dashed border-rose-300 inline-block" />
                    <span>Pending Removal</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-xs bg-slate-600 inline-block" />
                    <span>Absent</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Inspection Drawer */}
        <div className="w-96 bg-neutral-900 border-l border-neutral-800 flex flex-col shrink-0 overflow-y-auto">
          {selectedEntityId && selectedSpatial ? (
            <div className="p-5 flex flex-col gap-5">
              <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
                <div className="flex items-center gap-2">
                  <span className="text-base font-semibold text-white">
                    {selectedSpatial.entity.label}
                  </span>
                  <span className="text-xs font-mono text-neutral-400">
                    #{selectedSpatial.entity.entity_id.slice(0, 8)}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedEntityId(null)}
                  className="text-neutral-400 hover:text-white text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700"
                >
                  Close
                </button>
              </div>

              {/* State and Status */}
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-neutral-400 block mb-1">
                  Epistemic State
                </span>
                <StatusBadge status={selectedSpatial.entity.current_state} size="md" />
              </div>

              {/* Boundary Proximity Warning */}
              {selectedSpatial.isClipped && (
                <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-start gap-2.5">
                  <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
                  <div>
                    <span className="font-semibold block">Boundary Proximity</span>
                    <span className="text-amber-200/80 text-[11px]">
                      This object is close to or intersects the camera frame boundary. Temporal movement gating rules apply.
                    </span>
                  </div>
                </div>
              )}

              {/* Spatial Coordinates Readout */}
              <div className="bg-neutral-950 p-3.5 rounded-lg border border-neutral-800 flex flex-col gap-2">
                <span className="text-[11px] font-mono text-neutral-400 font-semibold flex items-center gap-1.5">
                  <Compass className="w-3.5 h-3.5 text-orange-400" />
                  Canonical Coordinates
                </span>
                {selectedSpatial.latestBbox ? (
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono text-neutral-300">
                    <div className="bg-neutral-900 p-2 rounded border border-neutral-800/80">
                      <span className="text-[10px] text-neutral-500 block">X Range</span>
                      {selectedSpatial.latestBbox.x_min.toFixed(1)} → {selectedSpatial.latestBbox.x_max.toFixed(1)} px
                    </div>
                    <div className="bg-neutral-900 p-2 rounded border border-neutral-800/80">
                      <span className="text-[10px] text-neutral-500 block">Y Range</span>
                      {selectedSpatial.latestBbox.y_min.toFixed(1)} → {selectedSpatial.latestBbox.y_max.toFixed(1)} px
                    </div>
                    <div className="bg-neutral-900 p-2 rounded border border-neutral-800/80">
                      <span className="text-[10px] text-neutral-500 block">Width</span>
                      {(selectedSpatial.latestBbox.x_max - selectedSpatial.latestBbox.x_min).toFixed(1)} px
                    </div>
                    <div className="bg-neutral-900 p-2 rounded border border-neutral-800/80">
                      <span className="text-[10px] text-neutral-500 block">Height</span>
                      {(selectedSpatial.latestBbox.y_max - selectedSpatial.latestBbox.y_min).toFixed(1)} px
                    </div>
                  </div>
                ) : (
                  <span className="text-xs text-neutral-500">No canonical bounding box stored.</span>
                )}
              </div>

              {/* Displacement Analysis */}
              {selectedSpatial.displacementPx !== undefined && (
                <div className="bg-neutral-950 p-3.5 rounded-lg border border-neutral-800 flex flex-col gap-2">
                  <span className="text-[11px] font-mono text-neutral-400 font-semibold flex items-center gap-1.5">
                    <Move className="w-3.5 h-3.5 text-amber-400" />
                    Spatial Displacement
                  </span>
                  <div className="flex items-baseline justify-between text-xs font-mono">
                    <span className="text-neutral-400">Total Distance:</span>
                    <span className="text-amber-400 font-bold text-sm">
                      {selectedSpatial.displacementPx.toFixed(1)} px
                    </span>
                  </div>
                  {selectedSpatial.dx !== undefined && selectedSpatial.dy !== undefined && (
                    <div className="text-[11px] font-mono text-neutral-400 flex items-center justify-between">
                      <span>Vector (ΔX, ΔY):</span>
                      <span className="text-neutral-200">
                        ({selectedSpatial.dx > 0 ? '+' : ''}
                        {selectedSpatial.dx.toFixed(1)}, {selectedSpatial.dy > 0 ? '+' : ''}
                        {selectedSpatial.dy.toFixed(1)}) px
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Observation Timeline in Drawer */}
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-neutral-400 block mb-2">
                  Observation Trace ({selectedEntityDetail?.observations.length || 0})
                </span>
                {loadingDetail ? (
                  <div className="text-xs font-mono text-neutral-500">Loading trace...</div>
                ) : selectedEntityDetail && selectedEntityDetail.observations.length > 0 ? (
                  <div className="flex flex-col gap-2">
                    {selectedEntityDetail.observations.map((obs, idx) => (
                      <div
                        key={obs.observation_id}
                        className="bg-neutral-950 p-2.5 rounded-lg border border-neutral-800 text-xs font-mono flex items-center justify-between"
                      >
                        <div>
                          <span className="text-neutral-300 font-medium block">
                            Visit {idx + 1}
                          </span>
                          <span className="text-[10px] text-neutral-500">
                            {obs.session_id.slice(0, 12)}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-[11px] text-neutral-400 block">
                            {obs.visibility_status}
                          </span>
                          {obs.detector_confidence !== null && obs.detector_confidence !== undefined && (
                            <span className="text-[10px] text-neutral-500">
                              conf: {(obs.detector_confidence * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <span className="text-xs text-neutral-500">No observation history recorded.</span>
                )}
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-neutral-500">
              <Compass className="w-10 h-10 mb-3 text-neutral-600" />
              <p className="text-xs font-medium text-neutral-400 mb-1">No Entity Selected</p>
              <p className="text-[11px] text-neutral-500 max-w-[200px]">
                Click on any object footprint in the workbench canvas to inspect canonical coordinates and movement history.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
