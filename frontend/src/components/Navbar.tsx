import React, { useState, useEffect } from 'react';
import type { WorkspaceSummary, HealthResponse } from '../types/api';
import { api } from '../api/client';
import {
  Layers,
  ChevronDown,
  Plus,
  Activity,
  GitCompare,
  Eye,
  Clock,
  Server,
  Compass,
  BarChart3,
} from 'lucide-react';

export type ActiveTab =
  | 'overview'
  | 'comparison'
  | 'spatial'
  | 'entities'
  | 'timeline'
  | 'diagnostics'
  | 'evaluation';

interface NavbarProps {
  currentWorkspace: WorkspaceSummary | null;
  workspaces: WorkspaceSummary[];
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  onSelectWorkspace: (ws: WorkspaceSummary | null) => void;
  onOpenCreateModal: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentWorkspace,
  workspaces,
  activeTab,
  onTabChange,
  onSelectWorkspace,
  onOpenCreateModal,
}) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      try {
        const h = await api.getHealth();
        if (mounted) {
          setHealth(h);
          setBackendOnline(true);
        }
      } catch {
        if (mounted) {
          setBackendOnline(false);
        }
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems: { tab: ActiveTab; label: string; icon: React.ReactNode }[] = [
    { tab: 'overview', label: 'Overview & Upload', icon: <Eye className="w-4 h-4" /> },
    { tab: 'comparison', label: 'Scene Comparison', icon: <GitCompare className="w-4 h-4" /> },
    { tab: 'spatial', label: 'Spatial Memory', icon: <Compass className="w-4 h-4" /> },
    { tab: 'entities', label: 'Entities & Beliefs', icon: <Layers className="w-4 h-4" /> },
    { tab: 'timeline', label: 'Change Timeline', icon: <Clock className="w-4 h-4" /> },
    { tab: 'diagnostics', label: 'Diagnostics', icon: <Activity className="w-4 h-4" /> },
    { tab: 'evaluation', label: 'Evaluation Lab', icon: <BarChart3 className="w-4 h-4" /> },
  ];

  return (
    <header className="bg-[#090B10] border-b border-[#232A3B] sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Workspace Selector */}
          <div className="flex items-center gap-6">
            <div
              onClick={() => onSelectWorkspace(null)}
              className="flex items-center gap-2.5 cursor-pointer hover:opacity-95 transition group"
            >
              <div className="w-8 h-8 rounded-lg bg-[#06B6D4] flex items-center justify-center font-mono font-bold text-[#090B10] shadow-md shadow-cyan-950/40">
                OM
              </div>
              <div>
                <span className="font-semibold text-sm tracking-tight text-[#F8FAFC] flex items-center gap-1.5">
                  ObjectMemory <span className="text-[#22D3EE] font-mono text-xs font-normal">Studio</span>
                </span>
                <span className="block text-[10px] text-[#94A3B8] font-mono">
                  Persistent Visual Memory
                </span>
              </div>
            </div>

            {/* Workspace Dropdown */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#11141D] border border-[#232A3B] text-xs text-[#F8FAFC] hover:border-[#06B6D4]/50 transition cursor-pointer"
              >
                <span className="w-2 h-2 rounded-full bg-[#06B6D4]" />
                <span className="font-medium max-w-[160px] truncate">
                  {currentWorkspace ? currentWorkspace.name : 'Select Workspace'}
                </span>
                <ChevronDown className="w-3.5 h-3.5 text-[#94A3B8]" />
              </button>

              {dropdownOpen && (
                <div
                  className="absolute left-0 mt-1.5 w-64 rounded-xl bg-[#11141D] border border-[#232A3B] shadow-2xl py-1.5 z-50 animate-in fade-in zoom-in-95 duration-100"
                  onMouseLeave={() => setDropdownOpen(false)}
                >
                  <div className="px-3 py-1.5 text-[11px] font-mono text-[#94A3B8] uppercase tracking-wider border-b border-[#232A3B] flex items-center justify-between">
                    <span>Workspaces</span>
                    <span>{workspaces.length}</span>
                  </div>

                  <div className="max-h-56 overflow-y-auto py-1">
                    {workspaces.length === 0 ? (
                      <div className="px-3 py-3 text-xs text-[#94A3B8] text-center">
                        No workspaces found
                      </div>
                    ) : (
                      workspaces.map((ws) => (
                        <button
                          key={ws.workspace_id}
                          type="button"
                          onClick={() => {
                            onSelectWorkspace(ws);
                            setDropdownOpen(false);
                          }}
                          className={`w-full text-left px-3 py-2 text-xs flex items-center justify-between hover:bg-[#181D29] transition ${
                            currentWorkspace?.workspace_id === ws.workspace_id
                              ? 'bg-[#181D29] text-[#22D3EE] font-medium'
                              : 'text-[#94A3B8] hover:text-[#F8FAFC]'
                          }`}
                        >
                          <span className="truncate">{ws.name}</span>
                          <span className="text-[10px] font-mono text-[#64748B] ml-2">
                            {ws.total_sessions} visits
                          </span>
                        </button>
                      ))
                    )}
                  </div>

                  <div className="border-t border-[#232A3B] p-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setDropdownOpen(false);
                        onOpenCreateModal();
                      }}
                      className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#06B6D4]/10 hover:bg-[#06B6D4]/20 text-[#22D3EE] border border-[#06B6D4]/30 text-xs font-medium transition cursor-pointer"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      New Workspace
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Navigation Tabs (visible when a workspace is active) */}
          {currentWorkspace && (
            <nav className="hidden md:flex items-center gap-1 bg-[#11141D] p-1 rounded-lg border border-[#232A3B]">
              {navItems.map((item) => {
                const isActive = activeTab === item.tab;
                return (
                  <button
                    key={item.tab}
                    type="button"
                    onClick={() => onTabChange(item.tab)}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition cursor-pointer ${
                      isActive
                        ? 'bg-[#181D29] text-[#22D3EE] border border-[#06B6D4]/40 shadow-xs'
                        : 'text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#181D29]/60'
                    }`}
                  >
                    {item.icon}
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          )}

          {/* System Status Indicator */}
          <div className="flex items-center gap-3">
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono border ${
                backendOnline === true
                  ? 'bg-emerald-950/40 text-emerald-400 border-emerald-800/50'
                  : backendOnline === false
                  ? 'bg-rose-950/40 text-rose-400 border-rose-800/50'
                  : 'bg-[#11141D] text-[#94A3B8] border-[#232A3B]'
              }`}
              title={
                backendOnline === true
                  ? `FastAPI Online (${health?.detector_info?.architecture || 'YOLOv8'} CPU)`
                  : 'Backend unreachable at http://127.0.0.1:8000'
              }
            >
              <Server className="w-3 h-3" />
              <span className="w-1.5 h-1.5 rounded-full animate-pulse bg-current" />
              <span className="text-[11px]">
                {backendOnline === true ? 'Backend Connected' : backendOnline === false ? 'Offline' : 'Checking'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
