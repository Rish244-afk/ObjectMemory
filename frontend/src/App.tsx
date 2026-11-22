import React, { useState, useEffect } from 'react';
import type { WorkspaceSummary, WorkspaceResponse } from './types/api';
import { api } from './api/client';
import { Navbar } from './components/Navbar';
import type { ActiveTab } from './components/Navbar';
import { CreateWorkspaceModal } from './components/CreateWorkspaceModal';
import { DashboardView } from './views/DashboardView';
import { WorkspaceOverviewView } from './views/WorkspaceOverviewView';
import { SceneComparisonView } from './views/SceneComparisonView';
import { EntityHistoryView } from './views/EntityHistoryView';
import { TimelineView } from './views/TimelineView';
import { DiagnosticsView } from './views/DiagnosticsView';
import { SpatialMemoryExplorerView } from './views/SpatialMemoryExplorerView';
import { EvaluationLabView } from './views/EvaluationLabView';

const LOCAL_STORAGE_WS_KEY = 'objectmemory_active_workspace_id';

export const App: React.FC = () => {
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [currentWorkspace, setCurrentWorkspace] = useState<WorkspaceSummary | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');
  const [loading, setLoading] = useState(true);
  const [createModalOpen, setCreateModalOpen] = useState(false);

  const fetchWorkspaces = async (keepActiveId?: string | null) => {
    setLoading(true);
    try {
      const list = await api.listWorkspaces();
      setWorkspaces(list);

      const targetId = keepActiveId !== undefined ? keepActiveId : localStorage.getItem(LOCAL_STORAGE_WS_KEY);
      if (targetId) {
        const found = list.find((w) => w.workspace_id === targetId);
        if (found) {
          setCurrentWorkspace(found);
          localStorage.setItem(LOCAL_STORAGE_WS_KEY, found.workspace_id);
        } else {
          setCurrentWorkspace(null);
          localStorage.removeItem(LOCAL_STORAGE_WS_KEY);
        }
      }
    } catch (err: any) {
      console.error('Failed to load workspaces:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkspaces();
  }, []);

  const handleSelectWorkspace = (ws: WorkspaceSummary | null) => {
    setCurrentWorkspace(ws);
    if (ws) {
      localStorage.setItem(LOCAL_STORAGE_WS_KEY, ws.workspace_id);
      setActiveTab('overview');
    } else {
      localStorage.removeItem(LOCAL_STORAGE_WS_KEY);
    }
  };

  const handleWorkspaceCreated = (res: WorkspaceResponse) => {
    fetchWorkspaces(res.workspace.workspace_id);
  };

  const handleRefreshActiveWorkspace = async () => {
    if (!currentWorkspace) return;
    try {
      const resp = await api.getWorkspace(currentWorkspace.workspace_id);
      setCurrentWorkspace(resp.workspace);
      setWorkspaces((prev) =>
        prev.map((w) => (w.workspace_id === resp.workspace.workspace_id ? resp.workspace : w))
      );
    } catch (err) {
      console.error('Failed to refresh active workspace:', err);
    }
  };

  return (
    <div className="min-h-screen bg-[#090B10] text-[#F8FAFC] flex flex-col font-sans">
      <Navbar
        currentWorkspace={currentWorkspace}
        workspaces={workspaces}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onSelectWorkspace={handleSelectWorkspace}
        onOpenCreateModal={() => setCreateModalOpen(true)}
      />

      <main className="flex-1">
        {!currentWorkspace ? (
          <DashboardView
            workspaces={workspaces}
            loading={loading}
            onSelectWorkspace={handleSelectWorkspace}
            onOpenCreateModal={() => setCreateModalOpen(true)}
            onRefresh={() => fetchWorkspaces(null)}
          />
        ) : (
          <>
            {activeTab === 'overview' && (
              <WorkspaceOverviewView
                workspace={currentWorkspace}
                onRefreshWorkspace={handleRefreshActiveWorkspace}
                onNavigateToComparison={() => setActiveTab('comparison')}
              />
            )}
            {activeTab === 'comparison' && <SceneComparisonView workspace={currentWorkspace} />}
            {activeTab === 'spatial' && <SpatialMemoryExplorerView workspace={currentWorkspace} />}
            {activeTab === 'entities' && <EntityHistoryView workspace={currentWorkspace} />}
            {activeTab === 'timeline' && <TimelineView workspace={currentWorkspace} />}
            {activeTab === 'diagnostics' && <DiagnosticsView workspace={currentWorkspace} />}
            {activeTab === 'evaluation' && <EvaluationLabView workspace={currentWorkspace} />}
          </>
        )}
      </main>

      <footer className="border-t border-neutral-900 bg-neutral-950/80 py-4 px-6 text-center text-xs font-mono text-neutral-600">
        ObjectMemory Studio v1.0.0 • Long-Term Visual Memory for Detecting Persistent Environmental Changes
      </footer>

      <CreateWorkspaceModal
        isOpen={createModalOpen}
        onClose={() => setCreateModalOpen(false)}
        onCreated={handleWorkspaceCreated}
      />
    </div>
  );
};

export default App;
