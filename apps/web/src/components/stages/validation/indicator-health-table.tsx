import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatNumber } from "@/lib/utils/format";
import type { IndicatorHealth } from "@causal-ssm/api-types";

export function IndicatorHealthTable({ rows }: { rows: IndicatorHealth[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Indicator</TableHead>
          <TableHead className="text-right">Obs</TableHead>
          <TableHead className="text-right">Variance</TableHead>
          <TableHead className="text-right">Time Coverage</TableHead>
          <TableHead className="text-right">Max Gap</TableHead>
          <TableHead className="text-right">Dtype Violations</TableHead>
          <TableHead className="text-right">Dup %</TableHead>
          <TableHead>Arith. Seq.</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((h) => (
          <TableRow key={h.indicator}>
            <TableCell className="font-medium">{h.indicator}</TableCell>
            <TableCell className="text-right">{h.n_obs.toLocaleString()}</TableCell>
            <TableCell className="text-right">
              {h.variance !== null ? formatNumber(h.variance) : "--"}
            </TableCell>
            <TableCell className="text-right">
              {h.time_coverage_ratio !== null ? formatNumber(h.time_coverage_ratio) : "--"}
            </TableCell>
            <TableCell className="text-right">
              {h.max_gap_ratio !== null ? formatNumber(h.max_gap_ratio) : "--"}
            </TableCell>
            <TableCell className="text-right">
              {h.dtype_violations > 0 ? (
                <Badge variant="destructive">{h.dtype_violations}</Badge>
              ) : (
                "0"
              )}
            </TableCell>
            <TableCell className="text-right">{formatNumber(h.duplicate_pct)}</TableCell>
            <TableCell>
              {h.arithmetic_sequence_detected ? (
                <Badge variant="warning">detected</Badge>
              ) : (
                <span className="text-muted-foreground">none</span>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
