import React, { useState } from 'react';
import type { WorkspaceSummary } from '../types/api';
import { api } from '../api/client';
import {
  Plus,
  Trash2,
  Clock,
  ArrowRight,
  Camera,
  Loader2,
} from 'lucide-react';

interface DashboardViewProps {
  workspaces: WorkspaceSummary[];
  loading: boolean;
  onSelectWorkspace: (ws: WorkspaceSummary) => void;
  onOpenCreateModal: () => void;
  onRefresh: () => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  workspaces,
  loading,
  onSelectWorkspace,
  onOpenCreateModal,
  onRefresh,
}) => {
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const handleDelete = async (wsId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (deleteConfirmId !== wsId) {
      setDeleteConfirmId(wsId);
      return;
    }

    setDeletingId(wsId);
    try {
      await api.deleteWorkspace(wsId);
      setDeleteConfirmId(null);
      onRefresh();
    } catch (err: any) {
      alert(`Failed to delete workspace: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-in fade-in duration-150">
      {/* Hero header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-8 border-b border-[#232A3B] gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-[#F8FAFC] flex items-center gap-3">
            Physical Workspaces
            <span className="text-xs font-mono font-normal px-2.5 py-0.5 rounded-full bg-[#181D29] text-[#94A3B8] border border-[#232A3B]">
              {workspaces.length} Total
            </span>
          </h1>
          <p className="text-sm text-[#94A3B8] mt-1 max-w-2xl">
            Each workspace represents a physical desk or laboratory bench with isolated visual memory,
            tracking long-term object state across visits.
          </p>
        </div>

        <button
          onClick={onOpenCreateModal}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-[#06B6D4] hover:bg-[#22D3EE] text-[#090B10] text-sm font-semibold transition shadow-lg shadow-cyan-950/40 shrink-0 cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          Create Workspace
        </button>
      </div>

      {/* Loading state */}
      {loading ? (
        <div className="py-24 flex flex-col items-center justify-center gap-3 text-[#94A3B8]">
          <Loader2 className="w-8 h-8 animate-spin text-[#06B6D4]" />
          <span className="text-sm font-mono">Loading workspaces...</span>
        </div>
      ) : workspaces.length === 0 ? (
        /* Empty state */
        <div className="py-24 flex flex-col items-center justify-center text-center max-w-md mx-auto">
          <div className="w-16 h-16 rounded-2xl bg-[#11141D] border border-[#232A3B] flex items-center justify-center text-[#06B6D4] mb-4 shadow-inner">
            <Camera className="w-8 h-8" />
          </div>
          <h3 className="text-lg font-semibold text-[#F8FAFC] mb-1">No Physical Workspaces Yet</h3>
          <p className="text-sm text-[#94A3B8] mb-6">
            Create your first workspace to begin uploading visit imagery, detecting objects, and
            tracking environmental changes over time.
          </p>
          <button
            onClick={onOpenCreateModal}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-[#06B6D4] hover:bg-[#22D3EE] text-[#090B10] text-sm font-semibold transition cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            Create First Workspace
          </button>
        </div>
      ) : (
        /* Workspaces Grid */
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mt-8">
          {workspaces.map((ws) => {
            const isConfirmingDelete = deleteConfirmId === ws.workspace_id;
            return (
              <div
                key={ws.workspace_id}
                onClick={() => onSelectWorkspace(ws)}
                className="group bg-[#11141D] hover:bg-[#181D29] border border-[#232A3B] hover:border-[#06B6D4]/50 rounded-xl p-5 transition flex flex-col justify-between cursor-pointer shadow-sm relative overflow-hidden"
              >
                <div className="absolute top-0 left-0 right-0 h-1 bg-[#232A3B] group-hover:bg-[#06B6D4] transition" />

                <div>
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <h3 className="font-semibold text-[#F8FAFC] group-hover:text-[#22D3EE] transition truncate">
                      {ws.name}
                    </h3>

                    <button
                      type="button"
                      onClick={(e) => handleDelete(ws.workspace_id, e)}
                      disabled={deletingId === ws.workspace_id}
                      className={`p-1.5 rounded-md transition text-xs flex items-center gap-1 cursor-pointer ${
                        isConfirmingDelete
                          ? 'bg-rose-950 text-rose-300 border border-rose-800'
                          : 'text-[#64748B] hover:text-rose-400 hover:bg-[#181D29]'
                      }`}
                      title={isConfirmingDelete ? 'Click again to confirm deletion' : 'Delete workspace'}
                    >
                      {deletingId === ws.workspace_id ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="w-3.5 h-3.5" />
                      )}
                      {isConfirmingDelete && <span>Confirm?</span>}
                    </button>
                  </div>

                  <p className="text-xs font-mono text-[#64748B] mb-3">{ws.workspace_id}</p>

                  <p className="text-xs text-[#94A3B8] line-clamp-2 min-h-[32px] mb-4">
                    {ws.description || 'No description provided.'}
                  </p>

                  {/* Metrics Row */}
                  <div className="grid grid-cols-3 gap-2 py-3 border-y border-[#232A3B] mb-4 bg-[#090B10]/60 rounded-lg px-3">
                    <div>
                      <span className="block text-[10px] uppercase font-mono text-[#64748B]">Visits</span>
                      <span className="text-base font-bold font-mono text-[#F8FAFC]">
                        {ws.total_sessions}
                      </span>
                    </div>
                    <div>
                      <span className="block text-[10px] uppercase font-mono text-[#64748B]">Entities</span>
                      <span className="text-base font-bold font-mono text-[#F8FAFC]">
                        {ws.total_entities}
                      </span>
                    </div>
                    <div>
                      <span className="block text-[10px] uppercase font-mono text-[#64748B]">Events</span>
                      <span className="text-base font-bold font-mono text-[#F8FAFC]">
                        {ws.total_events}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2 text-xs font-mono text-[#64748B]">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {new Date(ws.created_at).toLocaleDateString()}
                  </span>

                  <span className="text-[#06B6D4] group-hover:translate-x-1 transition flex items-center gap-1 font-sans font-medium text-xs">
                    Open Workspace <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
