import { Annotation } from "@/components/ui/custom/annotation";
import { Skeleton } from "@/components/ui/skeleton";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { ReactNode } from "react";
import { StageHeader } from "./stage-header";

export function StageSection({
  number,
  title,
  status,
  context,
  children,
}: {
  number: string;
  title: string;
  status: StageRunStatus;
  context?: string;
  children?: ReactNode;
}) {
  return (
    <section className="rounded-lg border bg-card p-6 shadow-sm">
      <StageHeader number={number} title={title} status={status} />
      {status === "running" && (
        <div className="mt-4 space-y-3">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-32 w-full" />
        </div>
      )}
      {status === "completed" && (
        <div className="mt-4 space-y-4">
          {children}
          {context && <Annotation content={context} />}
        </div>
      )}
      {status === "failed" && (
        <div className="mt-4 text-sm text-destructive">
          Stage failed. Check pipeline logs for details.
        </div>
      )}
    </section>
  );
}
