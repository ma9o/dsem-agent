import { Badge } from "@/components/ui/badge";
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
          <TableHead className="text-right">{"\u03B2\u0302"}</TableHead>
          <TableHead className="text-right">SE</TableHead>
          <TableHead className="text-right">95% CI</TableHead>
          <TableHead className="text-right">P({"\u03B2"}&gt;0)</TableHead>
          <TableHead>Identifiable</TableHead>
          <TableHead>Sensitivity</TableHead>
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
