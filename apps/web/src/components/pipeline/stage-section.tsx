"use client";

import { Skeleton } from "@/components/ui/skeleton";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { AlertCircle, ChevronDown } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { StageHeader } from "./stage-header";

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes}m ${secs}s`;
}

export function StageSection({
  id,
  number,
  title,
  status,
  context,
  children,
  defaultCollapsed = false,
  elapsedMs,
  hasGate = false,
  loadingHint,
}: {
  id?: string;
  number: string;
  title: string;
  status: StageRunStatus;
  context?: string;
  children?: ReactNode;
  defaultCollapsed?: boolean;
  elapsedMs?: number;
  hasGate?: boolean;
  loadingHint?: string;
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  // Expand when transitioning to completed
  useEffect(() => {
    if (status === "completed") {
      setCollapsed(false);
    }
  }, [status]);

  const isCollapsible = status === "completed";

  return (
    <section
      id={id}
      className="animate-fade-in-up scroll-mt-28 rounded-lg border bg-card p-4 shadow-sm sm:p-6"
    >
      <div
        className={isCollapsible ? "flex items-center cursor-pointer" : ""}
        onClick={isCollapsible ? () => setCollapsed((c) => !c) : undefined}
      >
        <div className="flex-1">
          <StageHeader number={number} title={title} status={status} hasGate={hasGate} context={context} />
        </div>
        {isCollapsible && (
          <div className="flex items-center gap-2">
            {elapsedMs !== undefined && (
              <span className="text-xs text-muted-foreground/60 font-mono">
                {formatElapsed(elapsedMs)}
              </span>
            )}
            <ChevronDown
              className={`h-5 w-5 shrink-0 text-muted-foreground transition-transform duration-200 ${collapsed ? "-rotate-90" : ""}`}
            />
          </div>
        )}
      </div>
      {status === "running" && (
        <div className="animate-fade-in mt-4 space-y-3">
          {loadingHint && (
            <p className="text-sm text-muted-foreground">{loadingHint}</p>
          )}
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-32 w-full" />
        </div>
      )}
      {status === "completed" && !collapsed && (
        <div className="animate-fade-in mt-4 space-y-4">
          {children}
        </div>
      )}
      {status === "failed" && (
        <div className="animate-fade-in mt-4 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-3">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
          <div className="text-sm">
            <p className="font-medium text-destructive">Stage failed</p>
            <p className="mt-0.5 text-muted-foreground">
              Check pipeline logs for details. This may be a transient error.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
