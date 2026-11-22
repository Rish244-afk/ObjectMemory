import React, { useState, useRef, useCallback, useEffect } from 'react';
import type { BoundingBox } from '../types/api';
import { ZoomIn, ZoomOut, RotateCcw, Eye, Layers } from 'lucide-react';

export interface OverlayBox {
  id: string;
  label: string;
  bbox?: BoundingBox | null;
  confidence?: number | null;
  isClipped?: boolean;
  status?: string;
  strokeColor?: string;
  fillColor?: string;
  strokeDasharray?: string;
}

export interface DisplacementVector {
  id: string;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  distancePx?: number | null;
  label?: string;
  color?: string;
}

interface BoundingBoxCanvasProps {
  imageUrl: string;
  boxes: OverlayBox[];
  vectors?: DisplacementVector[];
  title?: string;
  subtitle?: string;
  highlightedId?: string | null;
  onBoxSelect?: (boxId: string) => void;
  aspectRatioClass?: string;
  theme?: 'default' | 'reference' | 'query';
  // Controlled Zoom & Pan (for synchronized inspection across dual canvases)
  zoom?: number;
  pan?: { x: number; y: number };
  onZoomPanChange?: (zoom: number, pan: { x: number; y: number }) => void;
}

export const BoundingBoxCanvas: React.FC<BoundingBoxCanvasProps> = ({
  imageUrl,
  boxes,
  vectors = [],
  title,
  subtitle,
  highlightedId,
  onBoxSelect,
  aspectRatioClass = 'aspect-video',
  theme = 'default',
  zoom: controlledZoom,
  pan: controlledPan,
  onZoomPanChange,
}) => {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showVectors, setShowVectors] = useState(true);
  const [imageError, setImageError] = useState(false);

  // Internal Pan & Zoom fallback (when not controlled externally)
  const [internalZoom, setInternalZoom] = useState(1.0);
  const [internalPan, setInternalPan] = useState({ x: 0, y: 0 });

  const isControlled = controlledZoom !== undefined && controlledPan !== undefined;
  const zoom = isControlled ? controlledZoom : internalZoom;
  const pan = isControlled ? controlledPan : internalPan;

  const updateZoomPan = useCallback(
    (newZoom: number, newPan: { x: number; y: number }) => {
      if (isControlled && onZoomPanChange) {
        onZoomPanChange(newZoom, newPan);
      } else {
        setInternalZoom(newZoom);
        setInternalPan(newPan);
      }
    },
    [isControlled, onZoomPanChange]
  );

  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ startX: number; startY: number; initialPanX: number; initialPanY: number } | null>(null);
  const viewportRef = useRef<HTMLDivElement>(null);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
      setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    }
  };

  const handleZoomIn = () => {
    const nextZoom = Math.min(3.5, Math.round((zoom + 0.25) * 100) / 100);
    updateZoomPan(nextZoom, pan);
  };

  const handleZoomOut = () => {
    const nextZoom = Math.max(1.0, Math.round((zoom - 0.25) * 100) / 100);
    const nextPan = nextZoom === 1.0 ? { x: 0, y: 0 } : pan;
    updateZoomPan(nextZoom, nextPan);
  };

  const handleResetZoom = () => {
    updateZoomPan(1.0, { x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1.0) return;
    setIsDragging(true);
    dragStartRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      initialPanX: pan.x,
      initialPanY: pan.y,
    };
  };

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !dragStartRef.current) return;
      const dx = (e.clientX - dragStartRef.current.startX) / zoom;
      const dy = (e.clientY - dragStartRef.current.startY) / zoom;
      const newPan = {
        x: dragStartRef.current.initialPanX + dx,
        y: dragStartRef.current.initialPanY + dy,
      };
      if (isControlled && onZoomPanChange) {
        onZoomPanChange(zoom, newPan);
      } else {
        setInternalPan(newPan);
      }
    },
    [isDragging, zoom, isControlled, onZoomPanChange]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    dragStartRef.current = null;
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Filter boxes that have valid, finite coordinates
  const validBoxes = boxes.filter(
    (b) =>
      b.bbox &&
      Number.isFinite(b.bbox.x_min) &&
      Number.isFinite(b.bbox.y_min) &&
      Number.isFinite(b.bbox.x_max) &&
      Number.isFinite(b.bbox.y_max) &&
      b.bbox.x_max >= b.bbox.x_min &&
      b.bbox.y_max >= b.bbox.y_min
  );

  // Filter valid vectors (non-zero displacement and finite coordinates)
  const validVectors = vectors.filter(
    (v) =>
      Number.isFinite(v.startX) &&
      Number.isFinite(v.startY) &&
      Number.isFinite(v.endX) &&
      Number.isFinite(v.endY) &&
      (v.startX !== v.endX || v.startY !== v.endY)
  );

  return (
    <div className="flex flex-col bg-[#090B10] border border-[#232A3B] rounded-xl overflow-hidden shadow-md">
      {/* Canvas Top Bar */}
      <div className="px-4 py-2.5 bg-[#11141D] border-b border-[#232A3B] flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {title && <span className="font-semibold text-sm text-[#F8FAFC] tracking-tight">{title}</span>}
          {subtitle && <span className="text-xs text-[#94A3B8] font-mono">({subtitle})</span>}
          {theme === 'reference' && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-[#181D29] text-[#94A3B8] border border-[#334155]">
              REFERENCE
            </span>
          )}
          {theme === 'query' && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-[#06B6D4]/15 text-[#22D3EE] border border-[#06B6D4]/40">
              QUERY
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Layer toggles */}
          <button
            type="button"
            onClick={() => setShowBoxes(!showBoxes)}
            className={`text-xs px-2.5 py-1 rounded-md font-mono border transition flex items-center gap-1.5 cursor-pointer ${
              showBoxes
                ? 'bg-[#181D29] text-[#22D3EE] border-[#06B6D4]/40'
                : 'bg-[#11141D] text-[#94A3B8] border-[#232A3B] hover:text-[#F8FAFC]'
            }`}
            title="Toggle bounding box layer"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>{showBoxes ? 'Boxes On' : 'Boxes Off'}</span>
          </button>

          <button
            type="button"
            onClick={() => setShowLabels(!showLabels)}
            className={`text-xs px-2.5 py-1 rounded-md font-mono border transition flex items-center gap-1.5 cursor-pointer ${
              showLabels
                ? 'bg-[#181D29] text-[#22D3EE] border-[#06B6D4]/40'
                : 'bg-[#11141D] text-[#94A3B8] border-[#232A3B] hover:text-[#F8FAFC]'
            }`}
            title="Toggle label & confidence tags"
          >
            <Eye className="w-3.5 h-3.5" />
            <span>{showLabels ? 'Labels On' : 'Labels Off'}</span>
          </button>

          {validVectors.length > 0 && (
            <button
              type="button"
              onClick={() => setShowVectors(!showVectors)}
              className={`text-xs px-2 py-1 rounded-md font-mono border transition cursor-pointer ${
                showVectors
                  ? 'bg-[#181D29] text-amber-400 border-amber-500/40'
                  : 'bg-[#11141D] text-[#94A3B8] border-[#232A3B] hover:text-[#F8FAFC]'
              }`}
              title="Toggle displacement vector overlays"
            >
              Vectors ({validVectors.length})
            </button>
          )}

          {/* Zoom controls */}
          <div className="flex items-center border border-[#232A3B] rounded-lg bg-[#090B10] overflow-hidden ml-1">
            <button
              type="button"
              onClick={handleZoomOut}
              disabled={zoom <= 1.0}
              className="p-1 hover:bg-[#181D29] text-[#94A3B8] hover:text-[#F8FAFC] disabled:opacity-40 transition cursor-pointer"
              title="Zoom out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="px-2 text-[11px] font-mono text-[#F8FAFC] min-w-[2.8rem] text-center select-none">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={handleZoomIn}
              disabled={zoom >= 3.5}
              className="p-1 hover:bg-[#181D29] text-[#94A3B8] hover:text-[#F8FAFC] disabled:opacity-40 transition cursor-pointer"
              title="Zoom in"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            {zoom > 1.0 && (
              <button
                type="button"
                onClick={handleResetZoom}
                className="p-1 hover:bg-[#181D29] text-[#06B6D4] transition border-l border-[#232A3B] cursor-pointer"
                title="Reset zoom to 100%"
              >
                <RotateCcw className="w-3 h-3" />
              </button>
            )}
          </div>

          {naturalSize && (
            <span className="text-[11px] font-mono text-[#64748B] ml-2 hidden sm:inline">
              {naturalSize.width}×{naturalSize.height}px • {validBoxes.length} {validBoxes.length === 1 ? 'obj' : 'objs'}
            </span>
          )}
        </div>
      </div>

      {/* Main Viewport Container */}
      <div
        ref={viewportRef}
        onMouseDown={handleMouseDown}
        className={`relative w-full ${aspectRatioClass} bg-[#090B10] flex items-center justify-center overflow-hidden select-none ${
          zoom > 1.0 ? 'cursor-grab active:cursor-grabbing' : ''
        }`}
      >
        {imageError ? (
          <div className="text-[#94A3B8] text-xs flex flex-col items-center gap-1.5 p-6">
            <span className="font-semibold text-rose-400">Unable to load observation frame</span>
            <span className="font-mono text-[10px] text-[#64748B] break-all">{imageUrl}</span>
          </div>
        ) : (
          <div
            className="relative w-full h-full flex items-center justify-center transition-transform duration-75 ease-out"
            style={{
              transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
              transformOrigin: 'center center',
            }}
          >
            <img
              src={imageUrl}
              alt={title || 'Workbench frame'}
              onLoad={handleImageLoad}
              onError={() => setImageError(true)}
              className="w-full h-full object-contain pointer-events-none select-none"
            />

            {naturalSize && (
              <svg
                viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
                className="absolute inset-0 w-full h-full object-contain pointer-events-auto"
                preserveAspectRatio="xMidYMid meet"
              >
                <defs>
                  <marker
                    id="arrowhead-amber"
                    markerWidth="8"
                    markerHeight="8"
                    refX="7"
                    refY="4"
                    orient="auto"
                  >
                    <polygon points="0 0, 8 4, 0 8" fill="#F59E0B" />
                  </marker>
                  <marker
                    id="arrowhead-cyan"
                    markerWidth="8"
                    markerHeight="8"
                    refX="7"
                    refY="4"
                    orient="auto"
                  >
                    <polygon points="0 0, 8 4, 0 8" fill="#22D3EE" />
                  </marker>
                </defs>

                {/* Displacement Vectors Layer */}
                {showVectors &&
                  validVectors.map((vec) => {
                    const isHigh = highlightedId === vec.id;
                    const strokeColor = vec.color || (isHigh ? '#22D3EE' : '#F59E0B');
                    const midX = (vec.startX + vec.endX) / 2;
                    const midY = (vec.startY + vec.endY) / 2;

                    return (
                      <g key={`vec-${vec.id}`} className="cursor-pointer">
                        {/* Background line for high contrast */}
                        <line
                          x1={vec.startX}
                          y1={vec.startY}
                          x2={vec.endX}
                          y2={vec.endY}
                          stroke="#090B10"
                          strokeWidth={4}
                          strokeLinecap="round"
                        />
                        <line
                          x1={vec.startX}
                          y1={vec.startY}
                          x2={vec.endX}
                          y2={vec.endY}
                          stroke={strokeColor}
                          strokeWidth={isHigh ? 2.5 : 2}
                          strokeDasharray="4 2"
                          markerEnd={`url(#${strokeColor === '#22D3EE' ? 'arrowhead-cyan' : 'arrowhead-amber'})`}
                        />
                        {/* Circle at origin */}
                        <circle
                          cx={vec.startX}
                          cy={vec.startY}
                          r={3.5}
                          fill={strokeColor}
                          stroke="#090B10"
                          strokeWidth={1}
                        />
                        {/* Magnitude label */}
                        <g transform={`translate(${midX}, ${midY})`}>
                          <rect
                            x={-24}
                            y={-10}
                            width={48}
                            height={14}
                            fill="#11141D"
                            stroke={strokeColor}
                            strokeWidth={1}
                            rx={3}
                          />
                          <text
                            x={0}
                            y={1}
                            textAnchor="middle"
                            fill="#F8FAFC"
                            fontSize={9}
                            fontFamily="monospace"
                            fontWeight="bold"
                          >
                            {vec.distancePx != null ? `${vec.distancePx.toFixed(0)}px` : 'vec'}
                          </text>
                        </g>
                      </g>
                    );
                  })}

                {/* Bounding Boxes Layer */}
                {showBoxes &&
                  validBoxes.map((b) => {
                    const bbox = b.bbox!;
                    const isHighlighted = highlightedId === b.id;
                    const w = bbox.x_max - bbox.x_min;
                    const h = bbox.y_max - bbox.y_min;

                    // Compute theme-specific styling
                    let defaultStroke = '#06B6D4';
                    let defaultFill = 'rgba(6, 182, 212, 0.08)';

                    if (theme === 'reference') {
                      defaultStroke = '#94A3B8'; // slate-400
                      defaultFill = 'rgba(148, 163, 184, 0.10)';
                    } else if (theme === 'query') {
                      defaultStroke = '#06B6D4'; // cyan-500
                      defaultFill = 'rgba(6, 182, 212, 0.12)';
                    }

                    if (b.strokeColor) defaultStroke = b.strokeColor;
                    if (b.fillColor) defaultFill = b.fillColor;

                    const strokeColor = isHighlighted
                      ? '#22D3EE' // cyan-400
                      : b.isClipped
                      ? '#F59E0B' // amber-500
                      : defaultStroke;

                    const fillColor = isHighlighted
                      ? 'rgba(34, 211, 238, 0.22)'
                      : defaultFill;

                    return (
                      <g
                        key={b.id}
                        className="cursor-pointer transition-opacity hover:opacity-100"
                        onClick={() => onBoxSelect && onBoxSelect(b.id)}
                      >
                        <rect
                          x={bbox.x_min}
                          y={bbox.y_min}
                          width={w}
                          height={h}
                          fill={fillColor}
                          stroke={strokeColor}
                          strokeWidth={isHighlighted ? 3 : 1.75}
                          strokeDasharray={b.strokeDasharray || (b.isClipped ? '4 2' : undefined)}
                        />
                        {showLabels && (
                          <g transform={`translate(${bbox.x_min}, ${Math.max(16, bbox.y_min - 4)})`}>
                            <rect
                              x={0}
                              y={-14}
                              width={Math.max(48, b.label.length * 8 + (b.confidence ? 38 : 12))}
                              height={16}
                              fill="#11141D"
                              stroke={strokeColor}
                              strokeWidth={1}
                              rx={2}
                            />
                            <text
                              x={4}
                              y={-2}
                              fill="#F8FAFC"
                              fontSize={11}
                              fontFamily="monospace"
                              fontWeight="bold"
                            >
                              {b.label}
                              {b.confidence != null && (
                                <tspan fill="#22D3EE" fontWeight="normal">
                                  {' '}
                                  {(b.confidence * 100).toFixed(0)}%
                                </tspan>
                              )}
                            </text>
                          </g>
                        )}
                      </g>
                    );
                  })}
              </svg>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
