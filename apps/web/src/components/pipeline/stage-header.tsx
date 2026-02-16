import { Badge } from "@/components/ui/badge";
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils/cn";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { ShieldCheck } from "lucide-react";

const statusVariant: Record<
  StageRunStatus,
  "default" | "secondary" | "success" | "destructive" | "warning"
> = {
  pending: "secondary",
  running: "default",
  completed: "success",
  failed: "destructive",
};

const statusLabel: Record<StageRunStatus, string> = {
  pending: "Pending",
  running: "Running...",
  completed: "Complete",
  failed: "Failed",
};

export function StageHeader({
  number,
  title,
  status,
  hasGate = false,
}: {
  number: string;
  title: string;
  status: StageRunStatus;
  hasGate?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold transition-colors",
          status === "completed"
            ? "bg-emerald-500 text-white"
            : status === "failed"
              ? "bg-destructive text-white"
              : status === "running"
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground",
        )}
      >
        {number}
      </div>
      <h2 className="text-base font-semibold sm:text-lg">{title}</h2>
      {hasGate && (
        <Tooltip content="This stage can halt the pipeline if checks fail">
          <ShieldCheck className="h-4 w-4 text-amber-500" />
        </Tooltip>
      )}
      <Badge variant={statusVariant[status]} className="ml-auto shrink-0">
        {statusLabel[status]}
      </Badge>
    </div>
  );
}
