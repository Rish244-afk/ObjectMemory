import React, { useState, useEffect } from 'react';
import type { WorkspaceSummary, EntitySummary, EntityDetailResponse, SessionSummary } from '../types/api';
import { api } from '../api/client';
import { StatusBadge } from '../components/StatusBadge';
import { EvidenceLedger } from '../components/EvidenceLedger';
import type { DecisionCertainty, EvidenceLedgerItem } from '../components/EvidenceLedger';
import { EntityCropThumbnail } from '../components/EntityCropThumbnail';
import {
  Layers,
  Search,
  Filter,
  Loader2,
  Crosshair,
  Info,
} from 'lucide-react';

interface EntityHistoryViewProps {
  workspace: WorkspaceSummary;
}

export const EntityHistoryView: React.FC<EntityHistoryViewProps> = ({ workspace }) => {
  const [entities, setEntities] = useState<EntitySummary[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [detail, setDetail] = useState<EntityDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [filterState, setFilterState] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const fetchEntities = async () => {
    setLoading(true);
    setError(null);
    try {
      const [eData, sData] = await Promise.all([
        api.listEntities(workspace.workspace_id, filterState !== 'ALL' ? filterState : undefined),
        api.listObservations(workspace.workspace_id),
      ]);
      setEntities(eData);
      setSessions(sData);
      if (eData.length > 0 && (!selectedEntityId || !eData.some((e) => e.entity_id === selectedEntityId))) {
        setSelectedEntityId(eData[0].entity_id);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load entities.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEntities();
  }, [workspace.workspace_id, filterState]);

  useEffect(() => {
    if (!selectedEntityId) {
      setDetail(null);
      return;
    }

    let active = true;
    const loadDetail = async () => {
      setDetailLoading(true);
      try {
        const d = await api.getEntityDetail(workspace.workspace_id, selectedEntityId);
        if (active) setDetail(d);
      } catch (err: any) {
        console.error('Failed to load entity detail:', err);
      } finally {
        if (active) setDetailLoading(false);
      }
    };
    loadDetail();
    return () => {
      active = false;
    };
  }, [workspace.workspace_id, selectedEntityId]);

  const filteredEntities = entities.filter(
    (e) =>
      e.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      e.entity_id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getObservationImageUrl = (sessionId: string, imageReference?: string | null): string => {
    const s = sessions.find((sess) => sess.session_id === sessionId);
    if (s?.image_url) return s.image_url;
    if (imageReference) {
      const cleanFilename = imageReference.replace(/^memory:\/\//, '').split('/').pop() || '';
      return api.getImageUrl(workspace.workspace_id, cleanFilename);
    }
    return '';
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 animate-in fade-in duration-150">
      {/* Header */}
      <div className="bg-[#11141D] border border-[#232A3B] rounded-xl p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-[#06B6D4]/10 border border-[#06B6D4]/20 text-[#22D3EE]">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-[#F8FAFC]">Tracked Entities & Visual Memory Beliefs</h1>
              <p className="text-xs text-[#94A3B8]">
                Persistent physical objects decoupled from transient per-frame detector labels.
              </p>
            </div>
          </div>

          {/* Filter and Search */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="w-4 h-4 text-[#64748B] absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search entity..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 pr-3 py-1.5 bg-[#090B10] border border-[#232A3B] rounded-lg text-xs text-[#F8FAFC] placeholder-[#64748B] focus:outline-hidden focus:border-[#06B6D4] w-44 font-mono"
              />
            </div>

            <div className="flex items-center gap-1.5 bg-[#090B10] px-2 py-1.5 rounded-lg border border-[#232A3B]">
              <Filter className="w-3.5 h-3.5 text-[#64748B]" />
              <select
                value={filterState}
                onChange={(e) => setFilterState(e.target.value)}
                className="bg-transparent text-xs text-[#F8FAFC] font-mono focus:outline-hidden cursor-pointer"
              >
                <option value="ALL" className="bg-[#11141D]">All States</option>
                <option value="CONFIRMED_PRESENT" className="bg-[#11141D]">CONFIRMED_PRESENT</option>
                <option value="CONFIRMED_MOVED" className="bg-[#11141D]">CONFIRMED_MOVED</option>
                <option value="PENDING_REMOVAL" className="bg-[#11141D]">PENDING_REMOVAL</option>
                <option value="CONFIRMED_REMOVED" className="bg-[#11141D]">CONFIRMED_REMOVED</option>
                <option value="INSUFFICIENT_EVIDENCE" className="bg-[#11141D]">INSUFFICIENT_EVIDENCE</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-950/50 border border-rose-800 rounded-xl text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* Master-Detail Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left List: Entities */}
        <div className="lg:col-span-4 bg-[#11141D] border border-[#232A3B] rounded-xl overflow-hidden shadow-sm">
          <div className="px-4 py-3 bg-[#090B10] border-b border-[#232A3B] flex items-center justify-between text-xs font-mono text-[#94A3B8]">
            <span>Known Entities ({filteredEntities.length})</span>
            <span>State</span>
          </div>

          <div className="divide-y divide-[#232A3B] max-h-[650px] overflow-y-auto">
            {loading ? (
              <div className="py-12 flex justify-center text-[#94A3B8]">
                <Loader2 className="w-6 h-6 animate-spin text-[#06B6D4]" />
              </div>
            ) : filteredEntities.length === 0 ? (
              <div className="py-12 text-center text-xs text-[#64748B]">
                No entities found matching criteria.
              </div>
            ) : (
              filteredEntities.map((ent) => {
                const isSelected = selectedEntityId === ent.entity_id;
                return (
                  <div
                    key={ent.entity_id}
                    onClick={() => setSelectedEntityId(ent.entity_id)}
                    className={`p-4 cursor-pointer transition flex items-start justify-between gap-3 ${
                      isSelected
                        ? 'bg-[#181D29] border-l-2 border-[#06B6D4]'
                        : 'hover:bg-[#181D29]/40'
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-sm text-[#F8FAFC]">{ent.label}</span>
                      </div>
                      <span className="font-mono text-xs text-[#22D3EE] block mb-1">
                        {ent.entity_id}
                      </span>
                      <span className="text-[11px] text-[#64748B] font-mono block">
                        Last seen: {ent.last_seen_session_id || 'never'}
                      </span>
                    </div>

                    <StatusBadge status={ent.current_state} size="sm" />
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Right Panel: Detail View */}
        <div className="lg:col-span-8 space-y-6">
          {detailLoading ? (
            <div className="bg-[#11141D] border border-[#232A3B] rounded-xl p-16 flex flex-col items-center justify-center gap-3 text-[#94A3B8]">
              <Loader2 className="w-8 h-8 animate-spin text-[#06B6D4]" />
              <span className="text-xs font-mono">Loading entity history...</span>
            </div>
          ) : !detail ? (
            <div className="bg-[#11141D] border border-[#232A3B] rounded-xl p-16 text-center text-[#94A3B8] text-xs">
              Select an entity from the list to inspect its temporal history.
            </div>
          ) : (
            <>
              {/* Entity Overview Card */}
              <div className="bg-[#11141D] border border-[#232A3B] rounded-xl p-6 shadow-sm">
                <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-[#232A3B] gap-3">
                  <div>
                    <div className="flex items-center gap-3">
                      <h2 className="text-xl font-bold text-[#F8FAFC]">{detail.entity.label}</h2>
                      <StatusBadge status={detail.entity.current_state} />
                    </div>
                    <span className="font-mono text-xs text-[#94A3B8] mt-1 block">
                      Entity ID: <span className="text-[#22D3EE] font-bold">{detail.entity.entity_id}</span> •
                      First observed: {new Date(detail.entity.created_at).toLocaleString()}
                    </span>
                  </div>

                  <div className="text-xs font-mono bg-[#090B10] px-3 py-2 rounded-lg border border-[#232A3B] text-[#94A3B8]">
                    <span className="text-[#64748B] block text-[10px] uppercase">Last Seen In</span>
                    <span className="text-[#F8FAFC] font-bold">
                      {detail.entity.last_seen_session_id || 'None'}
                    </span>
                  </div>
                </div>

                {detail.entity.canonical_bbox && (
                  <div className="mt-4 pt-2 flex items-center gap-2 text-xs font-mono text-[#94A3B8]">
                    <Crosshair className="w-4 h-4 text-[#64748B] shrink-0" />
                    <span>Canonical Reference Coordinates:</span>
                    <span className="text-[#22D3EE] font-bold bg-[#090B10] px-2 py-0.5 rounded border border-[#232A3B]">
                      [{detail.entity.canonical_bbox.x_min.toFixed(1)},{' '}
                      {detail.entity.canonical_bbox.y_min.toFixed(1)},{' '}
                      {detail.entity.canonical_bbox.x_max.toFixed(1)},{' '}
                      {detail.entity.canonical_bbox.y_max.toFixed(1)}]
                    </span>
                  </div>
                )}
              </div>

              {/* Per-Visit Raw Observations & Visual Crop Gallery */}
              <div className="bg-[#11141D] border border-[#232A3B] rounded-xl overflow-hidden shadow-sm">
                <div className="px-6 py-3.5 border-b border-[#232A3B] flex items-center justify-between">
                  <div>
                    <span className="text-xs font-mono uppercase tracking-wider text-[#94A3B8] font-semibold block">
                      Observation History & Visual Crop Gallery ({detail.observations.length})
                    </span>
                    <span className="text-[11px] text-[#64748B] font-mono">
                      Dynamic HTML5 canvas crops bounded to camera pixel space
                    </span>
                  </div>
                </div>

                {/* Scientific Disclaimer Notice */}
                <div className="px-6 py-2.5 bg-[#090B10] border-b border-[#232A3B] flex items-center gap-2 text-[11px] font-mono text-[#94A3B8]">
                  <Info className="w-4 h-4 text-[#06B6D4] shrink-0" />
                  <span>
                    <strong>Disclaimer:</strong> Visual crops are dynamic extracts from raw camera frames provided as supporting visual context. They do not constitute biometric proof of persistent entity identity.
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#090B10] font-mono uppercase text-[10px] text-[#94A3B8] border-b border-[#232A3B]">
                      <tr>
                        <th className="px-4 py-2.5">Crop</th>
                        <th className="px-4 py-2.5">Visit</th>
                        <th className="px-4 py-2.5">Timestamp</th>
                        <th className="px-4 py-2.5">Visibility</th>
                        <th className="px-4 py-2.5">Detector Conf</th>
                        <th className="px-4 py-2.5">Identity Match</th>
                        <th className="px-4 py-2.5">Raw Coordinates</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#232A3B] font-mono">
                      {detail.observations.map((obs) => {
                        const imgUrl = getObservationImageUrl(obs.session_id, obs.image_reference);
                        return (
                          <tr key={obs.observation_id} className="hover:bg-[#181D29]/30">
                            <td className="px-4 py-2.5">
                              <EntityCropThumbnail
                                imageUrl={imgUrl}
                                bbox={obs.raw_bbox}
                                alt={`Crop of ${obs.label || detail.entity.label} in ${obs.session_id}`}
                                size={64}
                              />
                            </td>
                            <td className="px-4 py-2.5 text-[#F8FAFC] font-bold">{obs.session_id}</td>
                            <td className="px-4 py-2.5 text-[#94A3B8]">
                              {new Date(obs.timestamp).toLocaleDateString()}
                            </td>
                            <td className="px-4 py-2.5">
                              <StatusBadge status={obs.visibility_status} size="sm" />
                            </td>
                            <td className="px-4 py-2.5 text-[#F8FAFC]">
                              {obs.detector_confidence != null
                                ? `${(obs.detector_confidence * 100).toFixed(1)}%`
                                : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-[#22D3EE]">
                              {obs.identity_match_score != null
                                ? obs.identity_match_score.toFixed(3)
                                : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-[#94A3B8] text-[11px]">
                              {obs.raw_bbox ? (
                                `[${Math.round(obs.raw_bbox.x_min)}, ${Math.round(obs.raw_bbox.y_min)}, ${Math.round(obs.raw_bbox.x_max)}, ${Math.round(obs.raw_bbox.y_max)}]`
                              ) : (
                                <span className="text-[#64748B]">None (Unobserved)</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* State Transitions Audit Trail */}
              <div className="bg-[#11141D] border border-[#232A3B] rounded-xl overflow-hidden shadow-sm p-4">
                <EvidenceLedger
                  title="Belief Transitions & Evidence Audit"
                  emptyMessage="No state transitions recorded (initial belief state maintained)."
                  items={detail.change_events.map((ev) => {
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

                    const item: EvidenceLedgerItem = {
                      id: ev.event_id,
                      timestamp: new Date(ev.timestamp).toLocaleString(),
                      entityId: ev.entity_id,
                      label: detail.entity.label,
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
                    return item;
                  })}
                />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
