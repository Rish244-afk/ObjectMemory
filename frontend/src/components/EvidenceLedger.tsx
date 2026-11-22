import React, { useState } from 'react';
import {
  ShieldCheck,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Code,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';
import { StatusBadge } from './StatusBadge';

export type DecisionCertainty = 'CONFIRMED' | 'PROVISIONAL' | 'UNCERTAIN' | 'UNAVAILABLE';

export interface EvidenceLedgerItem {
  id: string; // Event ID or observation ID
  timestamp: string;
  entityId?: string;
  label?: string;
  priorState: string;
  newState: string;
  certainty: DecisionCertainty;
  reasonCode?: string;
  explanation: string;
  // Quantitative measurements
  displacementPx?: number | null;
  movementThresholdPx?: number | null;
  patchSimilarity?: number | null;
  vacancyThreshold?: number | null;
  unobservedStreak?: number | null;
  // Gating & Diagnostics
  registrationSuccess?: boolean | null;
  inlierRatio?: number | null;
  reprojectionError?: number | null;
  iouWithPrior?: number | null;
  isClipped?: boolean | null;
  detectorConfidence?: number | null;
  // Raw payload
  rawEvidence?: Record<string, any> | null;
}

interface EvidenceLedgerProps {
  items: EvidenceLedgerItem[];
  title?: string;
  emptyMessage?: string;
}

export const EvidenceLedger: React.FC<EvidenceLedgerProps> = ({
  items,
  title = 'Explainable Evidence Ledger',
  emptyMessage = 'No decision records available.',
}) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const getCertaintyBadge = (certainty: DecisionCertainty) => {
    switch (certainty) {
      case 'CONFIRMED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            <CheckCircle2 className="w-3 h-3" />
            CONFIRMED
          </span>
        );
      case 'PROVISIONAL':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-amber-500/15 text-amber-400 border border-amber-500/30">
            <AlertCircle className="w-3 h-3" />
            PROVISIONAL
          </span>
        );
      case 'UNCERTAIN':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-purple-500/15 text-purple-400 border border-purple-500/30">
            <AlertCircle className="w-3 h-3" />
            UNCERTAIN
          </span>
        );
      case 'UNAVAILABLE':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-[#181D29] text-[#94A3B8] border border-[#334155]">
            <HelpCircle className="w-3 h-3" />
            UNAVAILABLE
          </span>
        );
    }
  };

  if (!items || items.length === 0) {
    return (
      <div className="p-8 text-center text-xs font-mono text-[#94A3B8] bg-[#11141D] rounded-xl border border-[#232A3B]">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Ledger Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-[#232A3B]">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-[#22D3EE]" />
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-[#F8FAFC]">
            {title} ({items.length})
          </h3>
        </div>
        <span className="text-[11px] font-mono text-[#64748B]">
          Auditable Scientific Evidence Trail • Raw Measurements & Gating Gated
        </span>
      </div>

      {/* Structured Ledger Rows */}
      <div className="flex flex-col gap-3">
        {items.map((item) => {
          const isExpanded = expandedId === item.id;
          const hasRawPayload = item.rawEvidence && Object.keys(item.rawEvidence).length > 0;

          return (
            <div
              key={item.id}
              className="bg-[#11141D] border border-[#232A3B] rounded-xl p-4 flex flex-col gap-3.5 transition hover:border-[#334155] shadow-xs"
            >
              {/* Top Row: Event ID, Entity Identity, State Transition, Certainty, Timestamp */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#232A3B]">
                <div className="flex flex-wrap items-center gap-2.5">
                  <span className="text-xs font-mono font-bold text-[#F8FAFC] bg-[#090B10] px-2.5 py-1 rounded border border-[#232A3B]">
                    {item.id}
                  </span>
                  {item.entityId && (
                    <span className="text-xs font-mono text-[#22D3EE] font-semibold">
                      {item.entityId} {item.label && <span className="text-[#94A3B8]">({item.label})</span>}
                    </span>
                  )}
                  <div className="flex items-center gap-1.5 text-xs font-mono">
                    <StatusBadge status={item.priorState} size="sm" />
                    <ChevronRight className="w-3.5 h-3.5 text-[#64748B]" />
                    <StatusBadge status={item.newState} size="sm" />
                  </div>
                </div>

                <div className="flex items-center gap-2.5">
                  {getCertaintyBadge(item.certainty)}
                  <span className="text-[11px] font-mono text-[#64748B]">{item.timestamp}</span>
                </div>
              </div>

              {/* Middle Row: Verbatim Explanation */}
              <div className="flex flex-col gap-1.5">
                <div className="flex items-start gap-2">
                  {item.reasonCode && (
                    <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-[#181D29] text-[#22D3EE] border border-[#06B6D4]/30 shrink-0">
                      {item.reasonCode}
                    </span>
                  )}
                  <p className="text-xs text-[#F8FAFC] leading-relaxed font-sans">{item.explanation}</p>
                </div>
              </div>

              {/* Scientific Measurements & Gating Constraints Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 pt-2 border-t border-[#232A3B] text-[11px] font-mono">
                {/* Measured Displacement d */}
                <div className="bg-[#090B10] p-2 rounded-lg border border-[#232A3B]">
                  <span className="text-[#64748B] block text-[9px] uppercase tracking-wider">
                    Displacement (d)
                  </span>
                  {item.displacementPx != null ? (
                    <span className="text-[#22D3EE] font-bold">{item.displacementPx.toFixed(1)} px</span>
                  ) : (
                    <span className="text-[#64748B]">Not available</span>
                  )}
                </div>

                {/* Adaptive Movement Threshold T_mov */}
                <div className="bg-[#090B10] p-2 rounded-lg border border-[#232A3B]">
                  <span className="text-[#64748B] block text-[9px] uppercase tracking-wider">
                    Threshold (T_mov)
                  </span>
                  {item.movementThresholdPx != null ? (
                    <span className="text-[#F8FAFC] font-medium">{item.movementThresholdPx.toFixed(1)} px</span>
                  ) : (
                    <span className="text-[#64748B]">Not available</span>
                  )}
                </div>

                {/* Patch / Vacancy Similarity & Threshold */}
                <div className="bg-[#090B10] p-2 rounded-lg border border-[#232A3B]">
                  <span className="text-[#64748B] block text-[9px] uppercase tracking-wider">
                    Patch / Vacancy
                  </span>
                  {item.patchSimilarity != null ? (
                    <span className="text-[#F8FAFC] font-medium">
                      {item.patchSimilarity.toFixed(3)}
                      {item.vacancyThreshold != null && (
                        <span className="text-[#64748B] text-[9px]"> (th: {item.vacancyThreshold.toFixed(2)})</span>
                      )}
                    </span>
                  ) : (
                    <span className="text-[#64748B]">Not available</span>
                  )}
                </div>

                {/* Consecutive Unobserved Streak */}
                <div className="bg-[#090B10] p-2 rounded-lg border border-[#232A3B]">
                  <span className="text-[#64748B] block text-[9px] uppercase tracking-wider">
                    Absence Streak
                  </span>
                  {item.unobservedStreak != null ? (
                    <span className="text-amber-400 font-bold">{item.unobservedStreak} visit(s)</span>
                  ) : (
                    <span className="text-[#64748B]">Not available</span>
                  )}
                </div>

                {/* Planar Homography Registration Gating */}
                <div className="bg-[#090B10] p-2 rounded-lg border border-[#232A3B]">
                  <span className="text-[#64748B] block text-[9px] uppercase tracking-wider">
                    Planar Reg / Inliers
                  </span>
                  {item.registrationSuccess != null ? (
                    <span className={item.registrationSuccess ? 'text-emerald-400' : 'text-rose-400 font-bold'}>
                      {item.registrationSuccess ? 'OK' : 'FAILED'}
                      {item.inlierRatio != null && ` (${(item.inlierRatio * 100).toFixed(0)}%)`}
                    </span>
                  ) : (
                    <span className="text-[#64748B]">Not available</span>
                  )}
                </div>

                {/* Boundary Clamping / FoV Gating */}
                <div className="bg-[#090B10] p-2 rounded-lg border border-[#232A3B]">
                  <span className="text-[#64748B] block text-[9px] uppercase tracking-wider">
                    Boundary Clip
                  </span>
                  {item.isClipped != null ? (
                    <span className={item.isClipped ? 'text-amber-400 font-bold' : 'text-emerald-400'}>
                      {item.isClipped ? 'CLIPPED' : 'INTERIOR'}
                    </span>
                  ) : (
                    <span className="text-[#64748B]">Not available</span>
                  )}
                </div>
              </div>

              {/* Bottom Row: Raw Evidence JSON Drawer Toggle */}
              {hasRawPayload && (
                <div className="pt-1 flex justify-end">
                  <button
                    type="button"
                    onClick={() => setExpandedId(isExpanded ? null : item.id)}
                    className="flex items-center gap-1.5 text-xs font-mono text-[#94A3B8] hover:text-[#22D3EE] transition cursor-pointer"
                  >
                    <Code className="w-3.5 h-3.5" />
                    <span>Evidence JSON Payload</span>
                    {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  </button>
                </div>
              )}

              {/* Expanded JSON Payload Block */}
              {isExpanded && hasRawPayload && (
                <div className="mt-1 p-3.5 bg-[#090B10] rounded-lg border border-[#232A3B] text-[11px] font-mono text-[#22D3EE] overflow-x-auto shadow-inner">
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-[#232A3B] text-[10px] text-[#64748B]">
                    <span>Raw Supporting Evidence Payload</span>
                    <span>JSON Schema: EventEvidence</span>
                  </div>
                  <pre className="text-[#F8FAFC] leading-relaxed whitespace-pre-wrap break-all">
                    {JSON.stringify(item.rawEvidence, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
