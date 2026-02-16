"use client";

import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (this.state.error) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4">
          <p className="text-sm font-medium text-destructive">Something went wrong</p>
          <p className="mt-1 text-xs text-muted-foreground">{this.state.error.message}</p>
          <button
            type="button"
            className="mt-3 text-xs font-medium text-primary underline underline-offset-2 hover:no-underline"
            onClick={() => this.setState({ error: null })}
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
