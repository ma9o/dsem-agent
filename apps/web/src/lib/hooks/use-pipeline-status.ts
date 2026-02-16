"use client";

import { useQueryClient } from "@tanstack/react-query";
import type { PipelineProgress } from "./use-run-events";

export function usePipelineStatus(runId: string | null): PipelineProgress | undefined {
  const queryClient = useQueryClient();
  return queryClient.getQueryData<PipelineProgress>(["pipeline", runId, "status"]);
}
