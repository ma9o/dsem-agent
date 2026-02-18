import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper } from "@tanstack/react-table";
import { formatNumber } from "@/lib/utils/format";
import type { IndicatorHealth } from "@causal-ssm/api-types";

const col = createColumnHelper<IndicatorHealth>();

const columns = [
  col.accessor("indicator", {
    header: "Indicator",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("n_obs", {
    header: () => (
      <HeaderWithTooltip
        label="Obs"
        tooltip="Number of non-null observations after extraction. More observations generally yield more reliable estimates."
      />
    ),
    cell: (info) => info.getValue().toLocaleString(),
    meta: { align: "right" },
  }),
  col.accessor("variance", {
    header: () => (
      <HeaderWithTooltip
        label="Variance"
        tooltip="Sample variance of the indicator values. Near-zero variance means the series is effectively constant and carries no information."
      />
    ),
    cell: (info) => {
      const v = info.getValue();
      return v !== null ? formatNumber(v) : "--";
    },
    meta: { align: "right" },
  }),
  col.accessor("time_coverage_ratio", {
    header: () => (
      <HeaderWithTooltip
        label="Time Coverage"
        tooltip="Fraction of the requested time range that has data. Values close to 1.0 indicate good temporal coverage."
      />
    ),
    cell: (info) => {
      const v = info.getValue();
      return v !== null ? formatNumber(v) : "--";
    },
    meta: { align: "right" },
  }),
  col.accessor("max_gap_ratio", {
    header: () => (
      <HeaderWithTooltip
        label="Max Gap"
        tooltip="Longest consecutive gap without data as a fraction of the total time range. Large values indicate periods where the indicator is missing."
      />
    ),
    cell: (info) => {
      const v = info.getValue();
      return v !== null ? formatNumber(v) : "--";
    },
    meta: { align: "right" },
  }),
  col.accessor("dtype_violations", {
    header: () => (
      <HeaderWithTooltip
        label="Dtype Violations"
        tooltip="Number of values that could not be converted to the expected numeric type. Non-zero counts suggest data quality issues at the source."
      />
    ),
    cell: (info) => {
      const v = info.getValue();
      return v > 0 ? <Badge variant="destructive">{v}</Badge> : "0";
    },
    meta: { align: "right" },
  }),
  col.accessor("duplicate_pct", {
    header: () => (
      <HeaderWithTooltip
        label="Dup %"
        tooltip="Percentage of duplicate timestamp-value pairs. High duplication may indicate redundant data or extraction errors."
      />
    ),
    cell: (info) => formatNumber(info.getValue()),
    meta: { align: "right" },
  }),
  col.accessor("arithmetic_sequence_detected", {
    header: () => (
      <HeaderWithTooltip
        label="Arith. Seq."
        tooltip="Whether the values form an arithmetic sequence (constant step between consecutive observations). Detected sequences often indicate synthetic or interpolated data rather than real measurements."
      />
    ),
    cell: (info) =>
      info.getValue() ? (
        <Badge variant="warning">detected</Badge>
      ) : (
        <span className="text-muted-foreground">none</span>
      ),
  }),
];

export function IndicatorHealthTable({ rows }: { rows: IndicatorHealth[] }) {
  return <InfoTable columns={columns} data={rows} />;
}
