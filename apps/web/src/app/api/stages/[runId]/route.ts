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
 * Queries each stage's task name individually to avoid stage-2's hundreds
 * of parallel populate_indicators sub-tasks filling the default 200-row page.
 */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string }> },
) {
  const { runId } = await params;

  const stages: Record<string, { status: string; startedAt?: string; completedAt?: string }> = {};

  // Query each stage's task run in parallel
  const results = await Promise.allSettled(
    STAGES.map(async (stage) => {
      const res = await fetch(`${PREFECT_API}/task_runs/filter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          flow_runs: { id: { any_: [runId] } },
          task_runs: { name: { like_: `${stage.prefectTaskName}%` } },
          limit: 1,
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

      // For stages with multiple sub-tasks (e.g. populate_indicators),
      // aggregate: running if any running, completed only if all completed
      const hasRunning = runs.some((r) => r.state.type === "RUNNING");
      const hasFailed = runs.some((r) => r.state.type === "FAILED");
      const allCompleted = runs.every((r) => r.state.type === "COMPLETED");

      let status: string;
      if (hasFailed) status = "failed";
      else if (hasRunning) status = "running";
      else if (allCompleted) status = "completed";
      else return null;

      const earliest = runs.reduce((a, b) =>
        (a.start_time ?? "") < (b.start_time ?? "") ? a : b,
      );
      const latest = runs.reduce((a, b) =>
        (a.end_time ?? "") > (b.end_time ?? "") ? a : b,
      );

      return {
        stageId: stage.id,
        status,
        startedAt: earliest.start_time ?? undefined,
        completedAt: status === "completed" ? (latest.end_time ?? undefined) : undefined,
      };
    }),
  );

  for (const r of results) {
    if (r.status === "fulfilled" && r.value) {
      stages[r.value.stageId] = {
        status: r.value.status,
        startedAt: r.value.startedAt,
        completedAt: r.value.completedAt,
      };
    }
  }

  return NextResponse.json(stages);
}
