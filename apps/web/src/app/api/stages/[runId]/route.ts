import { STAGES } from "@causal-ssm/api-types";
import { NextResponse } from "next/server";

const PREFECT_API = "http://localhost:4200/api";

/**
 * GET /api/stages/{runId}
 *
 * Queries Prefect for task-run states belonging to a flow run and maps them
 * to pipeline stage statuses. Used to hydrate the UI with the current
 * snapshot — the WebSocket only delivers future events.
 *
 * For fan-out stages (e.g. stage-2 workers), queries all task runs and
 * returns aggregate worker progress counts.
 */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string }> },
) {
  const { runId } = await params;

  const stages: Record<
    string,
    {
      status: string;
      startedAt?: string;
      completedAt?: string;
      workerProgress?: { completed: number; running: number; failed: number; total: number };
    }
  > = {};

  // Query each stage's task run in parallel
  const results = await Promise.allSettled(
    STAGES.map(async (stage) => {
      const res = await fetch(`${PREFECT_API}/task_runs/filter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          flow_runs: { id: { any_: [runId] } },
          task_runs: { name: { like_: `${stage.prefectTaskName}%` } },
          // Fan-out stages have many tasks; others have one
          limit: stage.isFanOut ? 1000 : 1,
          sort: "EXPECTED_START_TIME_DESC",
        }),
      });
      if (!res.ok) return null;
      const runs: Array<{
        name: string;
        state: { type: string; name: string };
        start_time: string | null;
        end_time: string | null;
      }> = await res.json();

      if (runs.length === 0) return null;

      const hasRunning = runs.some((r) => r.state.type === "RUNNING");
      const hasFailed = runs.some((r) => r.state.type === "FAILED");
      const allCompleted = runs.every((r) => r.state.type === "COMPLETED");

      let status: string;
      if (hasFailed) status = "failed";
      else if (allCompleted) status = "completed";
      else if (hasRunning) status = "running";
      else status = "running"; // Some pending = still in progress

      const earliest = runs.reduce((a, b) =>
        (a.start_time ?? "") < (b.start_time ?? "") ? a : b,
      );
      const latest = runs.reduce((a, b) =>
        (a.end_time ?? "") > (b.end_time ?? "") ? a : b,
      );

      // For fan-out stages, compute worker progress counts
      let workerProgress: { completed: number; running: number; failed: number; total: number } | undefined;
      if (stage.isFanOut) {
        let completed = 0, running = 0, failed = 0;
        for (const r of runs) {
          if (r.state.type === "COMPLETED") completed++;
          else if (r.state.type === "RUNNING") running++;
          else if (r.state.type === "FAILED") failed++;
        }
        workerProgress = { completed, running, failed, total: runs.length };
      }

      return {
        stageId: stage.id,
        status,
        startedAt: earliest.start_time ?? undefined,
        completedAt: status === "completed" ? (latest.end_time ?? undefined) : undefined,
        workerProgress,
      };
    }),
  );

  for (const r of results) {
    if (r.status === "fulfilled" && r.value) {
      stages[r.value.stageId] = {
        status: r.value.status,
        startedAt: r.value.startedAt,
        completedAt: r.value.completedAt,
        workerProgress: r.value.workerProgress,
      };
    }
  }

  return NextResponse.json(stages);
}
