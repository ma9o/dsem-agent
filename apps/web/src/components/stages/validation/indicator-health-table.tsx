import { Badge } from "@/components/ui/badge";
import { StatTooltip } from "@/components/ui/stat-tooltip";
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
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              Obs
              <StatTooltip explanation="Number of non-null observations after extraction. More observations generally yield more reliable estimates." />
            </span>
          </TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              Variance
              <StatTooltip explanation="Sample variance of the indicator values. Near-zero variance means the series is effectively constant and carries no information." />
            </span>
          </TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              Time Coverage
              <StatTooltip explanation="Fraction of the requested time range that has data. Values close to 1.0 indicate good temporal coverage." />
            </span>
          </TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              Max Gap
              <StatTooltip explanation="Longest consecutive gap without data as a fraction of the total time range. Large values indicate periods where the indicator is missing." />
            </span>
          </TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              Dtype Violations
              <StatTooltip explanation="Number of values that could not be converted to the expected numeric type. Non-zero counts suggest data quality issues at the source." />
            </span>
          </TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              Dup %
              <StatTooltip explanation="Percentage of duplicate timestamp-value pairs. High duplication may indicate redundant data or extraction errors." />
            </span>
          </TableHead>
          <TableHead>
            <span className="inline-flex items-center gap-1">
              Arith. Seq.
              <StatTooltip explanation="Whether the values form an arithmetic sequence (constant step between consecutive observations). Detected sequences often indicate synthetic or interpolated data rather than real measurements." />
            </span>
          </TableHead>
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
