import React, { useState } from 'react';
import { api } from '../api/client';
import { X, Plus, AlertCircle, Loader2 } from 'lucide-react';
import type { WorkspaceResponse } from '../types/api';

interface CreateWorkspaceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreated: (workspace: WorkspaceResponse) => void;
}

export const CreateWorkspaceModal: React.FC<CreateWorkspaceModalProps> = ({
  isOpen,
  onClose,
  onCreated,
}) => {
  const [workspaceId, setWorkspaceId] = useState('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!workspaceId.trim() || !name.trim()) {
      setError('Workspace identifier and name are required.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const resp = await api.createWorkspace({
        workspace_id: workspaceId.trim(),
        name: name.trim(),
        description: description.trim() || undefined,
      });
      onCreated(resp);
      onClose();
      setWorkspaceId('');
      setName('');
      setDescription('');
    } catch (err: any) {
      setError(err.detail || err.message || 'Failed to create workspace.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4">
      <div className="bg-[#11141D] border border-[#232A3B] rounded-xl max-w-md w-full shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#232A3B]">
          <div className="flex items-center gap-2">
            <Plus className="w-5 h-5 text-[#06B6D4]" />
            <h3 className="font-semibold text-[#F8FAFC]">Create Physical Workspace</h3>
          </div>
          <button
            onClick={onClose}
            className="text-[#94A3B8] hover:text-[#F8FAFC] transition p-1 cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          {error && (
            <div className="p-3 bg-rose-950/60 border border-rose-800 rounded-lg flex items-start gap-2.5 text-xs text-rose-300">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1.5 font-mono">
              Workspace Slug ID <span className="text-[#06B6D4]">*</span>
            </label>
            <input
              type="text"
              value={workspaceId}
              onChange={(e) => setWorkspaceId(e.target.value.toLowerCase().replace(/[^a-z0-9_\-]/g, '_'))}
              placeholder="e.g. lab_workbench_01"
              required
              className="w-full px-3 py-2 bg-[#090B10] border border-[#232A3B] rounded-lg text-sm text-[#F8FAFC] placeholder-[#64748B] focus:outline-hidden focus:border-[#06B6D4] font-mono"
            />
            <p className="text-[11px] text-[#64748B] mt-1 font-mono">
              Unique identifier used in filesystem paths and API routes.
            </p>
          </div>

          <div>
            <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1.5 font-mono">
              Display Name <span className="text-[#06B6D4]">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Electronics Assembly Bench"
              required
              className="w-full px-3 py-2 bg-[#090B10] border border-[#232A3B] rounded-lg text-sm text-[#F8FAFC] placeholder-[#64748B] focus:outline-hidden focus:border-[#06B6D4]"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-[#94A3B8] uppercase tracking-wider mb-1.5 font-mono">
              Description (Optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Physical setup context, camera mount notes, lighting..."
              rows={3}
              className="w-full px-3 py-2 bg-[#090B10] border border-[#232A3B] rounded-lg text-sm text-[#F8FAFC] placeholder-[#64748B] focus:outline-hidden focus:border-[#06B6D4] resize-none"
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-3 border-t border-[#232A3B]">
            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="px-4 py-2 rounded-lg text-xs font-medium text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#181D29] transition cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[#06B6D4] hover:bg-[#22D3EE] text-[#090B10] text-xs font-semibold transition cursor-pointer shadow-md disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Creating...</span>
                </>
              ) : (
                <span>Create Workspace</span>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
