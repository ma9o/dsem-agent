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
import { formatNumber, formatPercent } from "@/lib/utils/format";
import type { TreatmentEffect } from "@causal-ssm/api-types";
import { AlertTriangle, Check, X } from "lucide-react";

interface TreatmentRankingTableProps {
  results: TreatmentEffect[];
}

export function TreatmentRankingTable({ results }: TreatmentRankingTableProps) {
  const sorted = [...results].sort((a, b) => Math.abs(b.beta_hat) - Math.abs(a.beta_hat));

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Treatment</TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              {"\u03B2\u0302"}
              <StatTooltip explanation="Posterior mean of the causal effect. Positive values indicate the treatment increases the outcome." />
            </span>
          </TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              SE
              <StatTooltip explanation="Standard error of the posterior estimate. Smaller values indicate more precise estimates." />
            </span>
          </TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              95% CI
              <StatTooltip explanation="95% credible interval. The true effect lies within this range with 95% posterior probability." />
            </span>
          </TableHead>
          <TableHead className="text-right">
            <span className="inline-flex items-center gap-1">
              P({"\u03B2"}&gt;0)
              <StatTooltip explanation="Posterior probability that the effect is positive. Values near 1 or 0 indicate strong directional evidence." />
            </span>
          </TableHead>
          <TableHead>
            <span className="inline-flex items-center gap-1">
              Identifiable
              <StatTooltip explanation="Whether the causal effect can be uniquely determined from the observational data given the DAG structure." />
            </span>
          </TableHead>
          <TableHead>
            <span className="inline-flex items-center gap-1">
              Sensitivity
              <StatTooltip explanation="Robustness to unobserved confounding. 'Sensitive' means the estimate may change substantially if assumptions are violated." />
            </span>
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((r) => (
          <TableRow key={r.treatment}>
            <TableCell className="font-medium">{r.treatment}</TableCell>
            <TableCell className="text-right font-mono text-sm">
              {formatNumber(r.beta_hat)}
            </TableCell>
            <TableCell className="text-right font-mono text-sm">{formatNumber(r.se)}</TableCell>
            <TableCell className="text-right font-mono text-sm">
              [{formatNumber(r.ci_lower)}, {formatNumber(r.ci_upper)}]
            </TableCell>
            <TableCell className="text-right font-mono text-sm">
              {formatPercent(r.p_positive)}
            </TableCell>
            <TableCell>
              {r.identifiable ? (
                <Check className="h-4 w-4 text-emerald-600" />
              ) : (
                <X className="h-4 w-4 text-destructive" />
              )}
            </TableCell>
            <TableCell>
              {r.sensitivity_flag ? (
                <Badge variant="warning">
                  <AlertTriangle className="mr-1 h-3 w-3" />
                  Sensitive
                </Badge>
              ) : (
                <Badge variant="success">Robust</Badge>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
