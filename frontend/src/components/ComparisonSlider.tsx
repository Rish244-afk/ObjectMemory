import React, { useState, useRef, useCallback, useEffect } from 'react';
import type { OverlayBox } from './BoundingBoxCanvas';

interface ComparisonSliderProps {
  beforeImageUrl: string;
  afterImageUrl: string;
  beforeTitle?: string;
  afterTitle?: string;
  beforeBoxes?: OverlayBox[];
  afterBoxes?: OverlayBox[];
  highlightedId?: string | null;
  onBoxSelect?: (boxId: string) => void;
}

export const ComparisonSlider: React.FC<ComparisonSliderProps> = ({
  beforeImageUrl,
  afterImageUrl,
  beforeTitle = 'Reference (T1)',
  afterTitle = 'Query (T2)',
  beforeBoxes = [],
  afterBoxes = [],
  highlightedId,
  onBoxSelect,
}) => {
  const [sliderPosition, setSliderPosition] = useState(50); // percentage 0 - 100
  const [isDragging, setIsDragging] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (img.naturalWidth > 0 && img.naturalHeight > 0 && !naturalSize) {
      setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    }
  };

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clamped = Math.max(0, Math.min(rect.width, x));
    setSliderPosition((clamped / rect.width) * 100);
  }, []);

  const handleMouseDown = () => setIsDragging(true);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setSliderPosition((pos) => Math.max(0, pos - (e.shiftKey ? 10 : 2)));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setSliderPosition((pos) => Math.min(100, pos + (e.shiftKey ? 10 : 2)));
    } else if (e.key === 'Home') {
      e.preventDefault();
      setSliderPosition(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      setSliderPosition(100);
    }
  };

  useEffect(() => {
    const handleMouseUp = () => setIsDragging(false);
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        handleMove(e.clientX);
      }
    };
    const handleTouchMove = (e: TouchEvent) => {
      if (isDragging && e.touches.length > 0) {
        handleMove(e.touches[0].clientX);
      }
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging, handleMove]);

  const renderBoxesSvg = (boxes: OverlayBox[], strokeBase: string, fillBase: string) => {
    if (!naturalSize || !showBoxes) return null;
    return (
      <svg
        viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
        className="absolute inset-0 w-full h-full object-contain pointer-events-auto"
        preserveAspectRatio="xMidYMid meet"
      >
        {boxes
          .filter((b) => b.bbox)
          .map((b) => {
            const bbox = b.bbox!;
            const isHighlighted = highlightedId === b.id;
            const stroke = isHighlighted ? '#22D3EE' : strokeBase;
            const fill = isHighlighted ? 'rgba(34, 211, 238, 0.25)' : fillBase;

            return (
              <g
                key={b.id}
                className="cursor-pointer"
                onClick={() => onBoxSelect && onBoxSelect(b.id)}
              >
                <rect
                  x={bbox.x_min}
                  y={bbox.y_min}
                  width={bbox.x_max - bbox.x_min}
                  height={bbox.y_max - bbox.y_min}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={isHighlighted ? 3 : 1.5}
                />
                <text
                  x={bbox.x_min + 3}
                  y={Math.max(12, bbox.y_min - 4)}
                  fill="#F8FAFC"
                  fontSize={10}
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  {b.label}
                </text>
              </g>
            );
          })}
      </svg>
    );
  };

  return (
    <div className="flex flex-col bg-[#090B10] border border-[#232A3B] rounded-xl overflow-hidden shadow-lg select-none">
      {/* Top Toolbar */}
      <div className="px-4 py-2.5 bg-[#11141D] border-b border-[#232A3B] flex items-center justify-between text-xs">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-[#F8FAFC] tracking-tight">Interactive Wipe Comparison</span>
          <span className="text-[#94A3B8] font-mono">
            {sliderPosition.toFixed(0)}% Ref / {(100 - sliderPosition).toFixed(0)}% Query
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowBoxes(!showBoxes)}
            className="px-2.5 py-1 rounded bg-[#181D29] hover:bg-[#232A3B] text-[#94A3B8] hover:text-[#F8FAFC] border border-[#232A3B] transition cursor-pointer font-mono text-[11px]"
          >
            {showBoxes ? 'Hide BBoxes' : 'Show BBoxes'}
          </button>
          <button
            type="button"
            onClick={() => setSliderPosition(50)}
            className="px-2.5 py-1 rounded bg-[#181D29] hover:bg-[#232A3B] text-[#94A3B8] hover:text-[#F8FAFC] border border-[#232A3B] transition cursor-pointer font-mono text-[11px]"
          >
            Reset (50/50)
          </button>
        </div>
      </div>

      {/* Wipe Container */}
      <div
        ref={containerRef}
        role="slider"
        tabIndex={0}
        aria-valuenow={Math.round(sliderPosition)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Image wipe comparison slider"
        onKeyDown={handleKeyDown}
        className="relative w-full aspect-video bg-[#090B10] overflow-hidden cursor-ew-resize focus:outline-hidden focus:ring-2 focus:ring-[#06B6D4]"
        onMouseDown={handleMouseDown}
        onTouchStart={handleMouseDown}
      >
        {/* Layer 1: Query (After) Image - Full width under Layer 2 */}
        <div className="absolute inset-0 w-full h-full">
          <img
            src={afterImageUrl}
            alt={afterTitle}
            onLoad={handleImageLoad}
            className="w-full h-full object-contain pointer-events-none"
          />
          {renderBoxesSvg(afterBoxes, '#06B6D4', 'rgba(6, 182, 212, 0.10)')}
          <div className="absolute bottom-3 right-3 px-2.5 py-1 bg-[#11141D]/90 rounded-md text-[11px] font-mono text-[#22D3EE] border border-[#06B6D4]/40 z-10">
            {afterTitle}
          </div>
        </div>

        {/* Layer 2: Baseline (Before) Image - Clipped to sliderPosition */}
        <div
          className="absolute inset-0 h-full overflow-hidden"
          style={{ width: `${sliderPosition}%` }}
        >
          <div
            className="relative h-full"
            style={{
              width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%',
            }}
          >
            <img
              src={beforeImageUrl}
              alt={beforeTitle}
              className="w-full h-full object-contain pointer-events-none"
            />
            {renderBoxesSvg(beforeBoxes, '#94A3B8', 'rgba(148, 163, 184, 0.10)')}
            <div className="absolute bottom-3 left-3 px-2.5 py-1 bg-[#11141D]/90 rounded-md text-[11px] font-mono text-[#94A3B8] border border-[#334155] z-10">
              {beforeTitle}
            </div>
          </div>
        </div>

        {/* Draggable Divider Handle */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-[#22D3EE] shadow-lg shadow-black flex items-center justify-center z-20 pointer-events-none"
          style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
        >
          <div className="w-7 h-7 rounded-full bg-[#11141D] border-2 border-[#06B6D4] shadow-xl flex items-center justify-center text-[10px] text-[#22D3EE] font-bold tracking-tighter">
            ⇆
          </div>
        </div>
      </div>
    </div>
  );
};
