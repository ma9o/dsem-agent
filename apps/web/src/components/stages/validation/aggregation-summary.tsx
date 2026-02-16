import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { AggregationSummary as AggregationSummaryType } from "@causal-ssm/api-types";

export function AggregationSummary({
  summary,
}: {
  summary: AggregationSummaryType[];
}) {
  if (summary.length === 0) {
    return <p className="text-sm text-muted-foreground">No aggregation summary available.</p>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Granularity</TableHead>
          <TableHead className="text-right">Indicators</TableHead>
          <TableHead className="text-right">Row Count</TableHead>
          <TableHead>Aggregation Functions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {summary.map((row) => (
          <TableRow key={row.granularity}>
            <TableCell className="font-medium">{row.granularity}</TableCell>
            <TableCell className="text-right">{row.n_indicators}</TableCell>
            <TableCell className="text-right">{row.row_count.toLocaleString()}</TableCell>
            <TableCell>
              <div className="flex flex-wrap gap-1">
                {row.aggregation_functions.map((fn) => (
                  <Badge key={fn} variant="outline" className="text-xs">
                    {fn}
                  </Badge>
                ))}
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
