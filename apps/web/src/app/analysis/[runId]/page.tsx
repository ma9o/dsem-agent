"use client";

import { AnalysisFeed } from "@/components/pipeline/analysis-feed";
import { usePipelineStatus } from "@/lib/hooks/use-pipeline-status";
import { useRunEvents } from "@/lib/hooks/use-run-events";
import { use } from "react";

export default function AnalysisPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = use(params);

  useRunEvents(runId);
  const progress = usePipelineStatus(runId);

  return <AnalysisFeed runId={runId} progress={progress} />;
}
