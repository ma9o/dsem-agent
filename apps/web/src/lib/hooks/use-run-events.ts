"use client";

import type { StageId, StageStatus } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";

export type StageRunStatus = Exclude<StageStatus, "blocked">;

export interface StageTiming {
  startedAt: number;
  completedAt?: number;
}

export interface PipelineProgress {
  stages: Record<StageId, StageRunStatus>;
  timings: Partial<Record<StageId, StageTiming>>;
  gateFailures: Partial<Record<StageId, boolean>>;
  gateOverrides: Partial<Record<StageId, boolean>>;
  currentStage: StageId | null;
  isComplete: boolean;
  isFailed: boolean;
}

function initialProgress(): PipelineProgress {
  const stages = {} as Record<StageId, StageRunStatus>;
  for (const s of STAGES) stages[s.id] = "pending";
  return { stages, timings: {}, gateFailures: {}, gateOverrides: {}, currentStage: null, isComplete: false, isFailed: false };
}

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_DELAY_MS = 1000;

export function useRunEvents(runId: string | null) {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const unmounted = useRef(false);

  const updateStage = useCallback(
    (stageId: StageId, status: StageRunStatus, eventTime?: number) => {
      queryClient.setQueryData<PipelineProgress>(["pipeline", runId, "status"], (old) => {
        const prev = old ?? initialProgress();
        const stages = { ...prev.stages, [stageId]: status };
        const completedAll = STAGES.every((s) => stages[s.id] === "completed");
        const anyFailed = STAGES.some((s) => stages[s.id] === "failed");

        // Use server event timestamp if available, otherwise fall back to client time
        const ts = eventTime ?? Date.now();
        const timings = { ...prev.timings };
        if (status === "running") {
          timings[stageId] = { startedAt: ts };
        } else if ((status === "completed" || status === "failed") && timings[stageId]) {
          timings[stageId] = { ...timings[stageId]!, completedAt: ts };
        }

        return {
          stages,
          timings,
          gateFailures: prev.gateFailures,
          gateOverrides: prev.gateOverrides,
          currentStage: status === "running" ? stageId : prev.currentStage,
          isComplete: completedAll,
          isFailed: anyFailed,
        };
      });
    },
    [queryClient, runId],
  );

  const connect = useCallback(() => {
    if (!runId || unmounted.current) return;

    const wsUrl = "ws://localhost:4200/api/events/out";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.resource?.["prefect.flow-run.id"] !== runId) return;

        const taskName = data.resource?.["prefect.task-run.name"];
        if (!taskName) return;

        const stage = STAGES.find((s) => taskName.startsWith(s.prefectTaskName));
        if (!stage) return;

        // Prefer server-side event timestamp over client-side Date.now()
        const eventTime = data.occurred ? new Date(data.occurred).getTime() : undefined;

        if (data.event === "prefect.task-run.Running") {
          updateStage(stage.id, "running", eventTime);
        } else if (data.event === "prefect.task-run.Completed") {
          updateStage(stage.id, "completed", eventTime);
          queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", stage.id] });
        } else if (data.event === "prefect.task-run.Failed") {
          updateStage(stage.id, "failed", eventTime);
        }
      } catch {
        // Ignore parse errors
      }
    };

    ws.onclose = () => {
      if (unmounted.current) return;

      // Check if pipeline is already complete — no need to reconnect
      const progress = queryClient.getQueryData<PipelineProgress>(["pipeline", runId, "status"]);
      if (progress?.isComplete || progress?.isFailed) return;

      // Exponential backoff reconnection
      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = BASE_DELAY_MS * 2 ** reconnectAttempts.current;
        reconnectAttempts.current++;
        setTimeout(() => {
          if (!unmounted.current) connect();
        }, delay);
      }
    };

    ws.onerror = () => {
      // Error will trigger onclose, which handles reconnection
    };
  }, [runId, queryClient, updateStage]);

  useEffect(() => {
    if (!runId) return;
    unmounted.current = false;

    // Initialize progress
    queryClient.setQueryData(["pipeline", runId, "status"], initialProgress());

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents({
        onStageStart: (id) => updateStage(id, "running"),
        onStageComplete: (id) => {
          updateStage(id, "completed");
          queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", id] });
        },
      });
      return cleanup;
    }

    connect();

    return () => {
      unmounted.current = true;
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [runId, queryClient, updateStage, connect]);
}
