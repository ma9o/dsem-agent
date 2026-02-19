import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import { diagnosisBadgeVariant, diagnosisLabel } from "@/lib/constants/charts";
import { formatNumber } from "@/lib/utils/format";
import type { PowerScalingResult } from "@causal-ssm/api-types";



const col = createColumnHelper<PowerScalingResult>();

const baseColumns = [
  col.accessor("parameter", {
    header: "Parameter",
    cell: (info) => (
      <span className="font-medium font-mono">{info.getValue()}</span>
    ),
  }),
  col.accessor("diagnosis", {
    header: () => (
      <HeaderWithTooltip
        label="Diagnosis"
        tooltip="'Well Identified' = data-driven. 'Prior Dominated' = prior choice matters too much. 'Prior-Data Conflict' = prior and data disagree."
      />
    ),
    cell: (info) => (
      <Badge variant={diagnosisBadgeVariant[info.getValue()] ?? "secondary"}>
        {diagnosisLabel[info.getValue()] ?? info.getValue()}
      </Badge>
    ),
  }),
  col.accessor("prior_sensitivity", {
    header: () => (
      <HeaderWithTooltip
        label="Prior Sens."
        tooltip="How much the posterior changes when the prior is scaled. Low values (<0.05) are good."
      />
    ),
    cell: (info) => formatNumber(info.getValue()),
    meta: { align: "right" as const, mono: true },
  }),
  col.accessor("likelihood_sensitivity", {
    header: () => (
      <HeaderWithTooltip
        label="Lik. Sens."
        tooltip="How much the posterior changes when the likelihood is scaled. High values indicate the data is informative."
      />
    ),
    cell: (info) => formatNumber(info.getValue()),
    meta: { align: "right" as const, mono: true },
  }),
];

const psisColumn = col.accessor("psis_k_hat", {
  header: () => (
    <HeaderWithTooltip
      label="PSIS k"
      tooltip="Pareto k diagnostic for importance sampling reliability. Values > 0.7 indicate unreliable sensitivity estimates."
    />
  ),
  cell: (info) => {
    const v = info.getValue();
    if (v == null) return "—";
    return (
      <Badge
        variant={v > 0.7 ? "destructive" : v > 0.5 ? "warning" : "success"}
      >
        {formatNumber(v, 2)}
      </Badge>
    );
  },
  meta: { align: "right" as const, mono: true },
});

export function PowerScalingTable({ results }: { results: PowerScalingResult[] }) {
  const hasPsis = results.some((p) => p.psis_k_hat != null);

  const columns = useMemo<ColumnDef<PowerScalingResult, unknown>[]>(
    () => (hasPsis ? [...baseColumns, psisColumn] : baseColumns),
    [hasPsis],
  );

  return <InfoTable columns={columns} data={results} />;
}
