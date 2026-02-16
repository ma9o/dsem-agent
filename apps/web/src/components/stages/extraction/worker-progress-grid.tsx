import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils/cn";
import type { WorkerStatus } from "@causal-ssm/api-types";
import { Cpu } from "lucide-react";

const statusVariant: Record<
  WorkerStatus["status"],
  "default" | "secondary" | "success" | "destructive"
> = {
  pending: "secondary",
  running: "default",
  completed: "success",
  failed: "destructive",
};

export function WorkerProgressGrid({ workers }: { workers: WorkerStatus[] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {workers.map((w) => (
        <Card
          key={w.worker_id}
          className={cn(
            w.status === "failed" && "border-red-200 dark:border-red-800",
            w.status === "completed" && "border-emerald-200 dark:border-emerald-800",
          )}
        >
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
                Worker {w.worker_id}
              </span>
              <Badge variant={statusVariant[w.status]}>{w.status}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm text-muted-foreground">
            <div className="flex justify-between">
              <span>Extractions</span>
              <span className="font-medium text-foreground">{w.n_extractions}</span>
            </div>
            <div className="flex justify-between">
              <span>Chunk size</span>
              <span className="font-medium text-foreground">{w.chunk_size}</span>
            </div>
            {w.error && <p className="mt-2 text-xs text-destructive">{w.error}</p>}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
