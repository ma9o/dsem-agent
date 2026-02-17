import { Tooltip } from "@/components/ui/tooltip";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { Check, Loader2, X } from "lucide-react";

export function PipelineProgressBar({ progress }: { progress: PipelineProgress | undefined }) {
  if (!progress) return null;

  const completed = STAGES.filter((s) => progress.stages[s.id] === "completed").length;

  return (
    <div className="sticky top-12 z-40 bg-background/80 backdrop-blur-sm border-b px-3 py-2.5 sm:px-4 sm:py-3">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-medium text-muted-foreground">
            {completed}/{STAGES.length} stages
          </span>
          {progress.isComplete && (
            <span className="animate-fade-in text-xs font-medium text-emerald-600">
              Complete
            </span>
          )}
          {progress.isFailed && (
            <span className="text-xs font-medium text-destructive">Failed</span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {STAGES.map((stage) => {
            const status = progress.stages[stage.id];
            const isClickable = status !== "pending";

            return (
              <Tooltip
                key={stage.id}
                content={
                  <div className="flex items-center gap-1.5 text-xs whitespace-nowrap">
                    {status === "completed" && <Check className="h-3 w-3 text-emerald-500" />}
                    {status === "running" && <Loader2 className="h-3 w-3 animate-spin" />}
                    {status === "failed" && <X className="h-3 w-3 text-destructive" />}
                    <span>
                      {stage.number}. {stage.label}
                    </span>
                  </div>
                }
              >
                <button
                  type="button"
                  disabled={!isClickable}
                  className="group relative flex-1"
                  onClick={() => {
                    if (!isClickable) return;
                    document
                      .getElementById(stage.id)
                      ?.scrollIntoView({ behavior: "smooth", block: "start" });
                  }}
                >
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      status === "completed"
                        ? "bg-emerald-500"
                        : status === "running"
                          ? "bg-primary animate-pulse-subtle"
                          : status === "failed"
                            ? "bg-destructive"
                            : "bg-secondary"
                    } ${isClickable ? "group-hover:opacity-80 cursor-pointer" : "cursor-default"}`}
                  />
                </button>
              </Tooltip>
            );
          })}
        </div>
      </div>
    </div>
  );
}
