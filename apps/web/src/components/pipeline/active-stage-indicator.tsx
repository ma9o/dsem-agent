import { STAGES } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import { Loader2 } from "lucide-react";

export function ActiveStageIndicator({ stageId }: { stageId: StageId | null }) {
  if (!stageId) return null;
  const stage = STAGES.find((s) => s.id === stageId);
  if (!stage) return null;

  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>
        Running Stage {stage.number}: {stage.label}...
      </span>
    </div>
  );
}
