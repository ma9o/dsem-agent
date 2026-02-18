import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Stage2Data } from "@causal-ssm/api-types";
import { CheckCircle2, XCircle } from "lucide-react";

export default function Stage2Content({ data }: { data: Stage2Data }) {
  if (data.workers.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No extraction workers were dispatched. Check if indicators were defined in the previous
        stage.
      </div>
    );
  }

  const succeeded = data.workers.filter((w) => w.status === "completed").length;
  const failed = data.workers.filter((w) => w.status === "failed").length;
  const errors = data.workers.filter((w) => w.status === "failed" && w.error);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 text-sm">
        <span className="flex items-center gap-1.5 text-success-soft-foreground">
          <CheckCircle2 className="h-3.5 w-3.5" />
          {succeeded} succeeded
        </span>
        {failed > 0 && (
          <span className="flex items-center gap-1.5 text-destructive">
            <XCircle className="h-3.5 w-3.5" />
            {failed} failed
          </span>
        )}
        <span className="text-muted-foreground">
          {data.total_extractions.toLocaleString()} extractions
        </span>
      </div>

      {errors.length > 0 && (
        <div className="space-y-1">
          {errors.map((w) => (
            <p key={w.worker_id} className="text-xs text-destructive">
              Worker {w.worker_id}: {w.error}
            </p>
          ))}
        </div>
      )}

      {data.combined_extractions_sample.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Indicator</TableHead>
              <TableHead>Value</TableHead>
              <TableHead>Timestamp</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.combined_extractions_sample.map((ext, i) => (
              <TableRow key={`${ext.indicator}-${ext.timestamp}-${i}`}>
                <TableCell className="font-medium">{ext.indicator}</TableCell>
                <TableCell className="font-mono text-sm">
                  {ext.value === null ? (
                    <span className="text-muted-foreground">null</span>
                  ) : (
                    String(ext.value)
                  )}
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {ext.timestamp ?? "--"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
