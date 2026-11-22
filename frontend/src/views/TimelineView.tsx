import React, { useState, useEffect } from 'react';
import type { WorkspaceSummary, ChangeEventSummary } from '../types/api';
import { api } from '../api/client';
import { EvidenceLedger } from '../components/EvidenceLedger';
import type { EvidenceLedgerItem, DecisionCertainty } from '../components/EvidenceLedger';
import {
  Clock,
  Filter,
  RefreshCw,
  Loader2,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';

interface TimelineViewProps {
  workspace: WorkspaceSummary;
}

export const TimelineView: React.FC<TimelineViewProps> = ({ workspace }) => {
  const [events, setEvents] = useState<ChangeEventSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedEntityFilter, setSelectedEntityFilter] = useState<string>('ALL');

  const fetchEvents = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listChangeEvents(workspace.workspace_id);
      setEvents(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load change events.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [workspace.workspace_id]);

  const uniqueEntities = Array.from(new Set(events.map((e) => e.entity_id))).sort();

  const filteredEvents = events.filter((e) =>
    selectedEntityFilter === 'ALL' ? true : e.entity_id === selectedEntityFilter
  );

  const ledgerItems: EvidenceLedgerItem[] = filteredEvents.map((ev) => {
    const sup = ev.supporting_evidence || {};

    let certainty: DecisionCertainty = 'CONFIRMED';
    if (ev.new_state.includes('PENDING') || ev.new_state.includes('PROVISIONAL')) {
      certainty = 'PROVISIONAL';
    } else if (
      ev.new_state === 'INSUFFICIENT_EVIDENCE' ||
      sup.registration_success === false ||
      sup.spatial_comparison_reliable === false
    ) {
      certainty = 'UNCERTAIN';
    }

    const reasonCode =
      (sup.reason_code as string) ||
      (ev.new_state === 'CONFIRMED_MOVED'
        ? 'DISPLACEMENT_EXCEEDS_THRESHOLD'
        : ev.new_state === 'PENDING_REMOVAL'
        ? 'ABSENCE_PROVISIONAL_COUNTER_1'
        : ev.new_state === 'CONFIRMED_REMOVED' || ev.new_state === 'CONFIRMED_ABSENT'
        ? 'CONSECUTIVE_ABSENCE_CONFIRMED'
        : ev.new_state === 'REAPPEARED'
        ? 'OBSERVED_AFTER_ABSENCE'
        : 'STATE_CONFIRMED');

    return {
      id: ev.event_id,
      timestamp: new Date(ev.timestamp).toLocaleString(),
      entityId: ev.entity_id,
      priorState: ev.prior_state,
      newState: ev.new_state,
      certainty,
      reasonCode,
      explanation: ev.explanation,
      displacementPx: ev.displacement_distance ?? (sup.displacement as number | undefined) ?? (sup.displacement_px as number | undefined),
      movementThresholdPx: (sup.movement_threshold as number | undefined) ?? (sup.adaptive_movement_threshold as number | undefined),
      patchSimilarity: (sup.patch_similarity as number | undefined) ?? (sup.vacancy_similarity as number | undefined),
      vacancyThreshold: (sup.vacancy_threshold as number | undefined),
      unobservedStreak: (sup.consecutive_vacant_visits as number | undefined) ?? (sup.unobserved_streak as number | undefined),
      registrationSuccess: sup.registration_success as boolean | undefined,
      inlierRatio: sup.inlier_ratio as number | undefined,
      reprojectionError: sup.mean_reprojection_error as number | undefined,
      iouWithPrior: sup.iou as number | undefined,
      isClipped: sup.is_clipped as boolean | undefined,
      detectorConfidence: sup.confidence as number | undefined,
      rawEvidence: sup,
    };
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 animate-in fade-in duration-150">
      {/* Header */}
      <div className="bg-[#11141D] border border-[#232A3B] rounded-xl p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-[#06B6D4]/10 border border-[#06B6D4]/20 text-[#22D3EE]">
              <Clock className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-[#F8FAFC]">Environmental Change Event Audit Log</h1>
              <p className="text-xs text-[#94A3B8]">
                Immutable, auditable chronological record of all persistent environmental transitions.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {uniqueEntities.length > 0 && (
              <div className="flex items-center gap-1.5 bg-[#090B10] px-2.5 py-1.5 rounded-lg border border-[#232A3B] text-xs">
                <Filter className="w-3.5 h-3.5 text-[#64748B]" />
                <select
                  value={selectedEntityFilter}
                  onChange={(e) => setSelectedEntityFilter(e.target.value)}
                  className="bg-transparent text-xs text-[#F8FAFC] font-mono focus:outline-hidden cursor-pointer"
                >
                  <option value="ALL" className="bg-[#11141D]">All Entities ({events.length})</option>
                  {uniqueEntities.map((eid) => (
                    <option key={eid} value={eid} className="bg-[#11141D]">
                      {eid}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <button
              onClick={fetchEvents}
              className="p-1.5 rounded hover:bg-[#181D29] text-[#94A3B8] hover:text-[#F8FAFC] border border-[#232A3B] transition cursor-pointer"
              title="Refresh timeline"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-[#22D3EE]' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-950/50 border border-rose-800 rounded-xl text-xs text-rose-300 flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="py-24 flex justify-center text-[#94A3B8]">
          <Loader2 className="w-8 h-8 animate-spin text-[#06B6D4]" />
        </div>
      ) : filteredEvents.length === 0 ? (
        <div className="py-20 text-center bg-[#11141D] border border-[#232A3B] rounded-xl text-[#94A3B8] text-xs">
          <HelpCircle className="w-6 h-6 mx-auto mb-2 text-[#64748B]" />
          No change events recorded in this workspace yet.
        </div>
      ) : (
        <EvidenceLedger
          title="Chronological Change Events"
          emptyMessage="No change events match the current filter."
          items={ledgerItems}
        />
      )}
    </div>
  );
};
