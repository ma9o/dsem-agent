"use client";

import type { StageId } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";

export type StageRunStatus = "pending" | "running" | "completed" | "failed";

export interface PipelineProgress {
  stages: Record<StageId, StageRunStatus>;
  currentStage: StageId | null;
  isComplete: boolean;
  isFailed: boolean;
}

function initialProgress(): PipelineProgress {
  const stages = {} as Record<StageId, StageRunStatus>;
  for (const s of STAGES) stages[s.id] = "pending";
  return { stages, currentStage: null, isComplete: false, isFailed: false };
}

export function useRunEvents(runId: string | null) {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);

  const updateStage = useCallback(
    (stageId: StageId, status: StageRunStatus) => {
      queryClient.setQueryData<PipelineProgress>(["pipeline", runId, "status"], (old) => {
        const prev = old ?? initialProgress();
        const stages = { ...prev.stages, [stageId]: status };
        const completedAll = STAGES.every((s) => stages[s.id] === "completed");
        const anyFailed = STAGES.some((s) => stages[s.id] === "failed");
        return {
          stages,
          currentStage: status === "running" ? stageId : prev.currentStage,
          isComplete: completedAll,
          isFailed: anyFailed,
        };
      });
    },
    [queryClient, runId],
  );

  useEffect(() => {
    if (!runId) return;

    // Initialize progress
    queryClient.setQueryData(["pipeline", runId, "status"], initialProgress());

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents({
        onStageStart: (id) => updateStage(id, "running"),
        onStageComplete: (id) => {
          updateStage(id, "completed");
          // Trigger data fetch
          queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", id] });
        },
      });
      return cleanup;
    }

    // Real WebSocket to Prefect
    const wsUrl = "ws://localhost:4200/api/events/out";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.resource?.["prefect.flow-run.id"] !== runId) return;

        const taskName = data.resource?.["prefect.task-run.name"];
        if (!taskName) return;

        const stage = STAGES.find((s) => taskName.startsWith(s.prefectTaskName));
        if (!stage) return;

        if (data.event === "prefect.task-run.Running") {
          updateStage(stage.id, "running");
        } else if (data.event === "prefect.task-run.Completed") {
          updateStage(stage.id, "completed");
          queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", stage.id] });
        } else if (data.event === "prefect.task-run.Failed") {
          updateStage(stage.id, "failed");
        }
      } catch {
        // Ignore parse errors
      }
    };

    ws.onclose = () => {
      // Reconnect after delay
      setTimeout(() => {
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
      }, 3000);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [runId, queryClient, updateStage]);
}
