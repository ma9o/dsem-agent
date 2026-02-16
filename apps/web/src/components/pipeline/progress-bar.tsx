import { Progress } from "@/components/ui/progress";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";

export function PipelineProgressBar({ progress }: { progress: PipelineProgress | undefined }) {
  if (!progress) return null;

  const completed = STAGES.filter((s) => progress.stages[s.id] === "completed").length;
  const pct = (completed / STAGES.length) * 100;

  return (
    <div className="sticky top-0 z-40 bg-background/80 backdrop-blur-sm border-b px-4 py-2">
      <div className="flex items-center gap-3 max-w-4xl mx-auto">
        <span className="text-xs font-medium text-muted-foreground whitespace-nowrap">
          {completed}/{STAGES.length} stages
        </span>
        <Progress value={pct} className="flex-1" />
        {progress.isComplete && (
          <span className="text-xs font-medium text-emerald-600">Complete</span>
        )}
        {progress.isFailed && <span className="text-xs font-medium text-destructive">Failed</span>}
      </div>
    </div>
  );
}
