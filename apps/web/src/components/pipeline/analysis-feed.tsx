"use client";

import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { ActiveStageIndicator } from "./active-stage-indicator";
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
      <div className="flex items-center justify-center py-20 text-muted-foreground">
        Waiting for pipeline to start...
      </div>
    );
  }

  const visibleStages = STAGES.filter((s) => progress.stages[s.id] !== "pending");

  return (
    <div>
      <PipelineProgressBar progress={progress} />
      <div className="max-w-4xl mx-auto space-y-6 p-6">
        {visibleStages.map((stage) => (
          <StageSectionRouter
            key={stage.id}
            stage={stage}
            runId={runId}
            status={progress.stages[stage.id]}
          />
        ))}
        <ActiveStageIndicator stageId={progress.currentStage} />
      </div>
    </div>
  );
}
