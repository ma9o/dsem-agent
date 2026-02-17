import { STAGES } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import { Loader2 } from "lucide-react";

export function ActiveStageIndicator({ stageId }: { stageId: StageId | null }) {
  if (!stageId) return null;
  const stage = STAGES.find((s) => s.id === stageId);
  if (!stage) return null;

  return (
    <div className="animate-fade-in-up flex items-center gap-2 rounded-lg border border-dashed border-muted-foreground/30 px-4 py-3 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>
        Running Stage {stage.number}: {stage.label}...
      </span>
    </div>
  );
}
