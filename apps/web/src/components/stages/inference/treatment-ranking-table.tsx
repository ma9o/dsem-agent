import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper } from "@tanstack/react-table";
import { formatNumber, formatPercent } from "@/lib/utils/format";
import type { TreatmentEffect } from "@causal-ssm/api-types";
import { AlertTriangle, Check, X } from "lucide-react";

const col = createColumnHelper<TreatmentEffect>();

const columns = [
  col.accessor("treatment", {
    header: "Treatment",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("beta_hat", {
    header: () => (
      <HeaderWithTooltip
        label={"\u03B2\u0302"}
        tooltip="Posterior mean of the causal effect. Positive values indicate the treatment increases the outcome."
      />
    ),
    cell: (info) => formatNumber(info.getValue()),
    meta: { align: "right", mono: true },
  }),
  col.accessor("se", {
    header: () => (
      <HeaderWithTooltip
        label="SE"
        tooltip="Standard error of the posterior estimate. Smaller values indicate more precise estimates."
      />
    ),
    cell: (info) => formatNumber(info.getValue()),
    meta: { align: "right", mono: true },
  }),
  col.display({
    id: "ci",
    header: () => (
      <HeaderWithTooltip
        label="95% CI"
        tooltip="95% credible interval. The true effect lies within this range with 95% posterior probability."
      />
    ),
    cell: ({ row }) => (
      <span>
        [{formatNumber(row.original.ci_lower)}, {formatNumber(row.original.ci_upper)}]
      </span>
    ),
    meta: { align: "right", mono: true },
  }),
  col.accessor("p_positive", {
    header: () => (
      <HeaderWithTooltip
        label={`P(\u03B2>0)`}
        tooltip="Posterior probability that the effect is positive. Values near 1 or 0 indicate strong directional evidence."
      />
    ),
    cell: (info) => formatPercent(info.getValue()),
    meta: { align: "right", mono: true },
  }),
  col.accessor("identifiable", {
    header: () => (
      <HeaderWithTooltip
        label="Identifiable"
        tooltip="Whether the causal effect can be uniquely determined from the observational data given the DAG structure."
      />
    ),
    cell: (info) =>
      info.getValue() ? (
        <Check className="h-4 w-4 text-success" />
      ) : (
        <X className="h-4 w-4 text-destructive" />
      ),
  }),
  col.accessor("sensitivity_flag", {
    header: () => (
      <HeaderWithTooltip
        label="Sensitivity"
        tooltip="Robustness to unobserved confounding. 'Sensitive' means the estimate may change substantially if assumptions are violated."
      />
    ),
    cell: (info) =>
      info.getValue() ? (
        <Badge variant="warning">
          <AlertTriangle className="mr-1 h-3 w-3" />
          Sensitive
        </Badge>
      ) : (
        <Badge variant="success">Robust</Badge>
      ),
  }),
];

export function TreatmentRankingTable({ results }: { results: TreatmentEffect[] }) {
  const sorted = useMemo(
    () => [...results].sort((a, b) => Math.abs(b.beta_hat) - Math.abs(a.beta_hat)),
    [results],
  );

  return <InfoTable columns={columns} data={sorted} />;
}
