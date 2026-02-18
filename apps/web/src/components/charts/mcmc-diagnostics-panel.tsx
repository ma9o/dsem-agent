"use client";

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
import type { MCMCDiagnostics } from "@causal-ssm/api-types";
import { AlertTriangle } from "lucide-react";

interface MCMCDiagnosticsPanelProps {
  diagnostics: MCMCDiagnostics;
}

function rhatBadge(value: number | number[]) {
  const v = Array.isArray(value) ? Math.max(...value) : value;
  if (v < 1.01) return <Badge variant="success">{formatNumber(v, 3)}</Badge>;
  if (v < 1.1) return <Badge variant="warning">{formatNumber(v, 3)}</Badge>;
  return <Badge variant="destructive">{formatNumber(v, 3)}</Badge>;
}

function essBadge(value: number | number[], nSamples: number | null) {
  const v = Array.isArray(value) ? Math.min(...value) : value;
  const total = nSamples ?? 1000;
  const ratio = v / total;
  if (ratio > 0.5) return <span className="font-mono text-sm">{formatNumber(v, 0)}</span>;
  if (ratio > 0.1)
    return <span className="font-mono text-sm text-warning">{formatNumber(v, 0)}</span>;
  return <span className="font-mono text-sm text-destructive">{formatNumber(v, 0)}</span>;
}

export function MCMCDiagnosticsPanel({ diagnostics }: MCMCDiagnosticsPanelProps) {
  const hasDivergences = diagnostics.num_divergences > 0;

  return (
    <div className="space-y-3">
      {/* Sampler-level summary */}
      <div className="flex flex-wrap gap-2">
        {diagnostics.num_chains != null && (
          <Badge variant="secondary">{diagnostics.num_chains} chains</Badge>
        )}
        {diagnostics.num_samples != null && (
          <Badge variant="secondary">{diagnostics.num_samples.toLocaleString()} samples</Badge>
        )}
        <Badge variant={hasDivergences ? "destructive" : "success"}>
          {hasDivergences && <AlertTriangle className="mr-1 h-3 w-3" />}
          {diagnostics.num_divergences} divergence{diagnostics.num_divergences !== 1 && "s"}
          {hasDivergences && ` (${formatNumber(diagnostics.divergence_rate * 100, 1)}%)`}
        </Badge>
        <Badge variant="secondary">
          tree depth: {formatNumber(diagnostics.tree_depth_mean, 1)} avg, {diagnostics.tree_depth_max} max
        </Badge>
        <Badge variant="secondary">
          accept: {formatNumber(diagnostics.accept_prob_mean * 100, 1)}%
        </Badge>
      </div>

      {/* Per-parameter table */}
      {diagnostics.per_parameter.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Parameter</TableHead>
              <TableHead className="text-right">
                <span className="inline-flex items-center gap-1">
                  R-hat
                  <StatTooltip explanation="Potential scale reduction factor. Values near 1.0 indicate convergence. Worry above 1.01." />
                </span>
              </TableHead>
              <TableHead className="text-right">
                <span className="inline-flex items-center gap-1">
                  ESS (bulk)
                  <StatTooltip explanation="Effective sample size for bulk of the distribution. Higher is better. Worry if < 100 per chain." />
                </span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {diagnostics.per_parameter.map((p) => (
              <TableRow key={p.parameter}>
                <TableCell className="font-medium font-mono text-sm">{p.parameter}</TableCell>
                <TableCell className="text-right">{rhatBadge(p.r_hat)}</TableCell>
                <TableCell className="text-right">
                  {essBadge(p.ess_bulk, diagnostics.num_samples)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
