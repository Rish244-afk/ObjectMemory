import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error in UI component:', error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 max-w-xl mx-auto my-12 bg-neutral-900 border border-rose-900/60 rounded-xl text-center shadow-xl">
          <div className="w-12 h-12 rounded-full bg-rose-950/70 border border-rose-800 text-rose-400 flex items-center justify-center mx-auto mb-4">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h2 className="text-base font-bold text-neutral-100 mb-2">
            {this.props.fallbackTitle || 'Component Rendering Exception'}
          </h2>
          <p className="text-xs text-neutral-400 font-mono mb-4 break-words bg-neutral-950 p-3 rounded border border-neutral-800 text-left">
            {this.state.error?.message || 'Unknown error occurred'}
          </p>
          <button
            type="button"
            onClick={this.handleReset}
            className="inline-flex items-center gap-2 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 rounded-lg text-xs font-medium transition"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Reload View
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
