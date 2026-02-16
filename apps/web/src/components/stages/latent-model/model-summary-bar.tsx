import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils/cn";
import { CheckCircle2, XCircle } from "lucide-react";

interface ModelSummaryBarProps {
  nConstructs: number;
  nEdges: number;
  outcomeName: string;
  exogenousCount: number;
  graphProperties: {
    is_acyclic: boolean;
    has_single_outcome: boolean;
  };
}

function StatPill({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center gap-1.5 rounded-full border bg-muted/50 px-3 py-1 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}

function CheckPill({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div
      className={cn(
        "flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm",
        ok
          ? "border-emerald-200 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950"
          : "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950",
      )}
    >
      {ok ? (
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
      ) : (
        <XCircle className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />
      )}
      <span>{label}</span>
    </div>
  );
}

export function ModelSummaryBar({
  nConstructs,
  nEdges,
  outcomeName,
  exogenousCount,
  graphProperties,
}: ModelSummaryBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <StatPill label="Constructs" value={nConstructs} />
      <StatPill label="Edges" value={nEdges} />
      <StatPill label="Exogenous" value={exogenousCount} />

      <div className="flex items-center gap-1.5 rounded-full border bg-muted/50 px-3 py-1 text-sm">
        <span className="text-muted-foreground">Outcome</span>
        <Badge variant="warning" className="ml-0.5">
          {outcomeName}
        </Badge>
      </div>

      <CheckPill label="Acyclic" ok={graphProperties.is_acyclic} />
      <CheckPill label="Single outcome" ok={graphProperties.has_single_outcome} />
    </div>
  );
}
