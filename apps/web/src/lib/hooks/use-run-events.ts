"use client";

import type { StageId, StageStatus } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import ReconnectingWebSocket from "reconnecting-websocket";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";

export type StageRunStatus = Exclude<StageStatus, "blocked">;

export interface StageTiming {
  startedAt: number;
  completedAt?: number;
}

export interface WorkerProgress {
  completed: number;
  running: number;
  failed: number;
  total: number;
}

export interface PipelineProgress {
  stages: Record<StageId, StageRunStatus>;
  timings: Partial<Record<StageId, StageTiming>>;
  gateFailures: Partial<Record<StageId, boolean>>;
  gateOverrides: Partial<Record<StageId, boolean>>;
  workerProgress: WorkerProgress | null;
  currentStage: StageId | null;
  isComplete: boolean;
  isFailed: boolean;
}

function initialProgress(): PipelineProgress {
  const stages = {} as Record<StageId, StageRunStatus>;
  for (const s of STAGES) stages[s.id] = "pending";
  return { stages, timings: {}, gateFailures: {}, gateOverrides: {}, workerProgress: null, currentStage: null, isComplete: false, isFailed: false };
}

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_DELAY_MS = 1000;

function computeWorkerProgress(workers: Map<string, StageRunStatus>): WorkerProgress {
  let completed = 0, running = 0, failed = 0;
  for (const state of workers.values()) {
    if (state === "completed") completed++;
    else if (state === "running") running++;
    else if (state === "failed") failed++;
  }
  return { completed, running, failed, total: workers.size };
}

export function useRunEvents(runId: string | null) {
  const queryClient = useQueryClient();
  const wsRef = useRef<ReconnectingWebSocket | null>(null);
  // Track individual worker task-run states for fan-out stages
  const fanOutWorkers = useRef(new Map<string, StageRunStatus>());

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
          workerProgress: prev.workerProgress,
          currentStage: status === "running" ? stageId : prev.currentStage,
          isComplete: completedAll,
          isFailed: anyFailed,
        };
      });
    },
    [queryClient, runId],
  );

  // Hydrate initial state from Prefect REST API (snapshot),
  // then subscribe to WebSocket for live updates (deltas).
  // Prefect's WebSocket does NOT replay historical events.
  useEffect(() => {
    if (!runId) return;

    queryClient.setQueryData(["pipeline", runId, "status"], initialProgress());
    fanOutWorkers.current.clear();

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

    // Abort controller ensures stale fetches (from React Strict Mode double-invocation)
    // don't apply after this effect instance is cleaned up.
    const ac = new AbortController();

    // 1. Snapshot: hydrate current state from Prefect REST API.
    //    The WebSocket only delivers future events — this fills in history.
    fetch(`/api/stages/${runId}`, { signal: ac.signal })
      .then((res) => (res.ok ? res.json() : null))
      .then((snapshot: Record<string, { status: string; startedAt?: string; completedAt?: string; workerProgress?: WorkerProgress }> | null) => {
        if (!snapshot) return;
        for (const [stageId, info] of Object.entries(snapshot)) {
          const status = info.status as StageRunStatus;
          const eventTime = info.completedAt
            ? new Date(info.completedAt).getTime()
            : info.startedAt
              ? new Date(info.startedAt).getTime()
              : undefined;
          updateStage(stageId as StageId, status, eventTime);
          if (status === "completed") {
            queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", stageId] });
          }
          // Hydrate worker progress from snapshot for fan-out stages
          if (info.workerProgress) {
            queryClient.setQueryData<PipelineProgress>(["pipeline", runId, "status"], (old) => {
              if (!old) return old;
              return { ...old, workerProgress: info.workerProgress! };
            });
          }
        }
      })
      .catch(() => {});

    // 2. Stream: subscribe to WebSocket for live events going forward
    const wsUrl = "ws://localhost:4200/api/events/out";
    const ws = new ReconnectingWebSocket(wsUrl, [], {
      maxRetries: MAX_RECONNECT_ATTEMPTS,
      minReconnectionDelay: BASE_DELAY_MS,
      maxReconnectionDelay: BASE_DELAY_MS * 2 ** MAX_RECONNECT_ATTEMPTS,
      reconnectionDelayGrowFactor: 2,
    });
    wsRef.current = ws;

    ws.onopen = () => {
      // Prefect's /api/events/out requires a filter message before it streams events.
      // `occurred.since` controls backfill depth; `occurred.until` MUST extend into
      // the future — without it, Prefect only sends backfill and stops streaming.
      const since = new Date(Date.now() - 15 * 60 * 1000).toISOString(); // 15 min ago
      const until = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(); // +24h
      ws.send(
        JSON.stringify({
          type: "filter",
          filter: {
            occurred: { since, until },
            event: { prefix: ["prefect.task-run."] },
            related: {
              resources_in_roles: [[`prefect.flow-run.${runId}`, "flow-run"]],
            },
          },
        }),
      );
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data);

        // Prefect wraps events in { type: "event", event: { ... } }
        const data = msg.type === "event" ? msg.event : msg;

        const taskName = data.resource?.["prefect.resource.name"];
        if (!taskName) return;

        const stage = STAGES.find((s) => taskName.startsWith(s.prefectTaskName));
        if (!stage) return;

        // Prefer server-side event timestamp over client-side Date.now()
        const eventTime = data.occurred ? new Date(data.occurred).getTime() : undefined;
        const isCompleted =
          data.event === "prefect.task-run.Completed" ||
          data.resource?.["prefect.state-type"] === "COMPLETED";

        // Fan-out stages: track individual worker task runs
        if (stage.isFanOut) {
          const taskRunId = data.resource?.["prefect.resource.id"];
          if (!taskRunId) return;

          const workers = fanOutWorkers.current;
          if (data.event === "prefect.task-run.Running") {
            workers.set(taskRunId, "running");
            // Mark stage as running on first worker
            updateStage(stage.id, "running", eventTime);
          } else if (isCompleted) {
            workers.set(taskRunId, "completed");
          } else if (data.event === "prefect.task-run.Failed") {
            workers.set(taskRunId, "failed");
          } else {
            // Pending or other — register the worker if not seen
            if (!workers.has(taskRunId)) workers.set(taskRunId, "pending");
          }

          const wp = computeWorkerProgress(workers);
          queryClient.setQueryData<PipelineProgress>(["pipeline", runId, "status"], (old) => {
            if (!old) return old;
            return { ...old, workerProgress: wp };
          });

          // Only mark stage completed when ALL workers are done
          const allDone = wp.completed + wp.failed === wp.total && wp.total > 0;
          if (allDone) {
            updateStage(stage.id, wp.failed > 0 ? "failed" : "completed", eventTime);
            queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", stage.id] });
          }
        } else {
          // Non-fan-out stages: single task per stage
          if (data.event === "prefect.task-run.Running") {
            updateStage(stage.id, "running", eventTime);
          } else if (isCompleted) {
            updateStage(stage.id, "completed", eventTime);
            queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", stage.id] });
          } else if (data.event === "prefect.task-run.Failed") {
            updateStage(stage.id, "failed", eventTime);
          }
        }

        // Close permanently if pipeline is done — no need to reconnect
        const progress = queryClient.getQueryData<PipelineProgress>(["pipeline", runId, "status"]);
        if (progress?.isComplete || progress?.isFailed) {
          ws.close();
        }
      } catch {
        // Ignore parse errors
      }
    };

    return () => {
      ac.abort();
      ws.close();
      wsRef.current = null;
    };
  }, [runId, queryClient, updateStage]);
}
