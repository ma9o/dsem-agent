import { Badge } from "@/components/ui/badge";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";

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
}: {
  number: string;
  title: string;
  status: StageRunStatus;
}) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-bold">
        {number}
      </div>
      <h2 className="text-lg font-semibold">{title}</h2>
      <Badge variant={statusVariant[status]}>{statusLabel[status]}</Badge>
    </div>
  );
}
