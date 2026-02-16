"use client";

import { BackToTop } from "@/components/back-to-top";
import { Skeleton } from "@/components/ui/skeleton";
import { useKeyboardNav } from "@/lib/hooks/use-keyboard-nav";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { Loader2 } from "lucide-react";
import { useMemo } from "react";
import { ActiveStageIndicator } from "./active-stage-indicator";
import { CompletionSummary } from "./completion-summary";
import { PipelineProgressBar } from "./progress-bar";
import { StageSectionRouter } from "./stage-section-router";

export function AnalysisFeed({
  runId,
  progress,
}: {
  runId: string;
  progress: PipelineProgress | undefined;
}) {
  if (!progress) {
    return (
      <div
        className="flex flex-col items-center justify-center gap-4 py-20"
        role="status"
        aria-live="polite"
      >
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <div className="text-center space-y-1">
          <p className="text-sm font-medium text-muted-foreground">
            Waiting for pipeline to start...
          </p>
          <p className="text-xs text-muted-foreground/60">
            This usually takes a few seconds
          </p>
        </div>
        <div className="w-full max-w-md space-y-3 mt-4">
          <Skeleton className="h-4 w-3/4 mx-auto" />
          <Skeleton className="h-4 w-1/2 mx-auto" />
        </div>
      </div>
    );
  }

  const visibleStages = STAGES.filter((s) => progress.stages[s.id] !== "pending");
  const visibleStageIds = useMemo(() => visibleStages.map((s) => s.id), [visibleStages]);
  useKeyboardNav(visibleStageIds);

  return (
    <div>
      <PipelineProgressBar progress={progress} />
      <div className="max-w-4xl mx-auto space-y-4 px-4 py-6 sm:space-y-6 sm:px-6">
        {visibleStages.map((stage) => (
          <StageSectionRouter
            key={stage.id}
            stage={stage}
            runId={runId}
            status={progress.stages[stage.id]}
            timing={progress.timings[stage.id]}
          />
        ))}
        {!progress.isComplete && <ActiveStageIndicator stageId={progress.currentStage} />}
        {progress.isComplete && <CompletionSummary runId={runId} />}
      </div>
      <BackToTop />
    </div>
  );
}
