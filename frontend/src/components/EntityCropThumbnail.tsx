import React, { useEffect, useRef, useState } from 'react';
import type { BoundingBox } from '../types/api';
import { ImageOff, EyeOff, Loader2 } from 'lucide-react';

interface EntityCropThumbnailProps {
  imageUrl?: string | null;
  bbox?: BoundingBox | null;
  alt?: string;
  size?: number; // width/height in px (default 80)
}

export const EntityCropThumbnail: React.FC<EntityCropThumbnailProps> = ({
  imageUrl,
  bbox,
  alt = 'Observation crop',
  size = 72,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'no-bbox' | 'error'>('loading');

  useEffect(() => {
    if (!bbox || !Number.isFinite(bbox.x_min) || !Number.isFinite(bbox.y_min) || !Number.isFinite(bbox.x_max) || !Number.isFinite(bbox.y_max)) {
      setStatus('no-bbox');
      return;
    }

    if (!imageUrl) {
      setStatus('error');
      return;
    }

    let active = true;
    setStatus('loading');

    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      if (!active) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const natW = img.naturalWidth;
      const natH = img.naturalHeight;

      if (!natW || !natH) {
        setStatus('error');
        return;
      }

      // 1. Strict Coordinate Clamping to Natural Image Bounds
      const clampX1 = Math.max(0, Math.min(natW, Math.round(bbox.x_min)));
      const clampY1 = Math.max(0, Math.min(natH, Math.round(bbox.y_min)));
      const clampX2 = Math.max(0, Math.min(natW, Math.round(bbox.x_max)));
      const clampY2 = Math.max(0, Math.min(natH, Math.round(bbox.y_max)));

      const cropW = clampX2 - clampX1;
      const cropH = clampY2 - clampY1;

      if (cropW <= 0 || cropH <= 0) {
        setStatus('no-bbox');
        return;
      }

      // Add a slight margin (10%) around the crop for workbench spatial context if within bounds
      const padX = Math.round(cropW * 0.1);
      const padY = Math.round(cropH * 0.1);

      const srcX = Math.max(0, clampX1 - padX);
      const srcY = Math.max(0, clampY1 - padY);
      const srcW = Math.min(natW - srcX, cropW + padX * 2);
      const srcH = Math.min(natH - srcY, cropH + padY * 2);

      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        setStatus('error');
        return;
      }

      // Clear canvas background
      ctx.fillStyle = '#090B10';
      ctx.fillRect(0, 0, size, size);

      // Compute aspect ratio preserving fit
      const scale = Math.min(size / srcW, size / srcH);
      const destW = srcW * scale;
      const destH = srcH * scale;
      const destX = (size - destW) / 2;
      const destY = (size - destH) / 2;

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, srcX, srcY, srcW, srcH, destX, destY, destW, destH);

      // Draw subtle cyan highlight outline
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.5)';
      ctx.lineWidth = 1;
      ctx.strokeRect(destX + 0.5, destY + 0.5, destW - 1, destH - 1);

      setStatus('success');
    };

    img.onerror = () => {
      if (active) setStatus('error');
    };

    img.src = imageUrl;

    return () => {
      active = false;
    };
  }, [imageUrl, bbox, size]);

  return (
    <div
      className="relative flex items-center justify-center rounded-lg bg-[#090B10] border border-[#232A3B] overflow-hidden shrink-0 select-none shadow-xs"
      style={{ width: `${size}px`, height: `${size}px` }}
      title={alt}
    >
      <canvas
        ref={canvasRef}
        className={`w-full h-full object-cover transition-opacity duration-150 ${
          status === 'success' ? 'opacity-100' : 'opacity-0'
        }`}
      />

      {status === 'loading' && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#090B10]/80">
          <Loader2 className="w-4 h-4 animate-spin text-[#06B6D4]" />
        </div>
      )}

      {status === 'no-bbox' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-1 text-center bg-[#11141D] text-[#64748B]">
          <EyeOff className="w-4 h-4 mb-0.5 text-[#64748B]" />
          <span className="text-[8px] font-mono leading-tight">No BBox</span>
        </div>
      )}

      {status === 'error' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-1 text-center bg-[#11141D] text-[#64748B]">
          <ImageOff className="w-4 h-4 mb-0.5 text-rose-400" />
          <span className="text-[8px] font-mono leading-tight text-rose-300">No Image</span>
        </div>
      )}
    </div>
  );
};
