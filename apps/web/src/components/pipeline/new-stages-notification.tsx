"use client";

import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import type { StageMeta } from "@causal-ssm/api-types";
import { ArrowDown } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

function getBelowViewport(progress: PipelineProgress): StageMeta[] {
  return STAGES.filter((s) => {
    const status = progress.stages[s.id];
    if (status !== "completed" && status !== "failed") return false;
    const el = document.getElementById(s.id);
    if (!el) return true; // not yet rendered, treat as below
    return el.getBoundingClientRect().top >= window.innerHeight;
  });
}

/**
 * Fixed bottom notification showing how many completed stages
 * are currently below the viewport. Count updates live as the
 * user scrolls.
 */
export function NewStagesNotification({
  progress,
}: {
  progress: PipelineProgress;
}) {
  const [belowStages, setBelowStages] = useState<StageMeta[]>([]);

  // Recompute on progress changes
  useEffect(() => {
    setBelowStages(getBelowViewport(progress));
  }, [progress]);

  // Recompute on scroll
  useEffect(() => {
    const handler = () => setBelowStages(getBelowViewport(progress));
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, [progress]);

  const scrollToNext = useCallback(() => {
    // Scroll to the first stage below the viewport
    if (belowStages.length === 0) return;
    const el = document.getElementById(belowStages[0].id);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [belowStages]);

  if (belowStages.length === 0) return null;

  const next = belowStages[0];
  const label =
    belowStages.length === 1
      ? `Stage ${next.number}: ${next.label} completed`
      : `${belowStages.length} stages completed below`;

  return (
    <button
      type="button"
      onClick={scrollToNext}
      className="animate-fade-in-up fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-2 rounded-full border bg-background px-4 py-2.5 shadow-lg transition-colors hover:bg-secondary"
    >
      <ArrowDown className="h-4 w-4" />
      <span className="text-sm font-medium">{label}</span>
    </button>
  );
}
