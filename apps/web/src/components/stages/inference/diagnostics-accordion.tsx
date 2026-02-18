"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
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
import type {
  MCMCDiagnostics,
  PPCResult,
  PowerScalingResult,
  SVIDiagnostics,
} from "@causal-ssm/api-types";
import { Check, X } from "lucide-react";
import { ELBOLossChart } from "@/components/charts/elbo-loss-chart";
import { MCMCDiagnosticsPanel } from "@/components/charts/mcmc-diagnostics-panel";
import { PPCRibbonChart } from "@/components/charts/ppc-ribbon-chart";
import { PPCTestStatChart } from "@/components/charts/ppc-test-stat-chart";
import { PowerScalingScatter } from "@/components/charts/power-scaling-scatter";

interface DiagnosticsAccordionProps {
  powerScaling: PowerScalingResult[];
  ppc: PPCResult;
  mcmcDiagnostics?: MCMCDiagnostics | null;
  sviDiagnostics?: SVIDiagnostics | null;
}

const diagnosisBadgeVariant: Record<string, "success" | "warning" | "destructive"> = {
  well_identified: "success",
  prior_dominated: "warning",
  prior_data_conflict: "destructive",
};

const diagnosisLabel: Record<string, string> = {
  well_identified: "Well Identified",
  prior_dominated: "Prior Dominated",
  prior_data_conflict: "Prior-Data Conflict",
};

export function DiagnosticsAccordion({
  powerScaling,
  ppc,
  mcmcDiagnostics,
  sviDiagnostics,
}: DiagnosticsAccordionProps) {
  // Determine which sections to open by default
  const defaultOpen = ["power-scaling", "ppc"];
  if (mcmcDiagnostics) defaultOpen.push("mcmc");
  if (sviDiagnostics) defaultOpen.push("svi");
  if (ppc.overlays?.length) defaultOpen.push("ppc-overlays");
  if (ppc.test_stats?.length) defaultOpen.push("ppc-stats");

  // Deduplicate overlay variables for grouped display
  const overlayVars = ppc.overlays ?? [];
  const testStatsByVar = new Map<string, typeof ppc.test_stats>();
  for (const ts of ppc.test_stats ?? []) {
    const existing = testStatsByVar.get(ts.variable) ?? [];
    existing.push(ts);
    testStatsByVar.set(ts.variable, existing);
  }

  return (
    <Accordion defaultOpen={defaultOpen}>
      {/* MCMC Convergence Diagnostics */}
      {mcmcDiagnostics && (
        <AccordionItem value="mcmc">
          <AccordionTrigger value="mcmc" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              MCMC Convergence
              <StatTooltip explanation="Chain mixing and convergence diagnostics: R-hat, effective sample size, divergences, and tree depth from NUTS/HMC." />
            </span>
            <Badge
              variant={mcmcDiagnostics.num_divergences === 0 ? "success" : "destructive"}
              className="ml-2"
            >
              {mcmcDiagnostics.num_divergences === 0 ? "Converged" : `${mcmcDiagnostics.num_divergences} divergences`}
            </Badge>
          </AccordionTrigger>
          <AccordionContent value="mcmc">
            <MCMCDiagnosticsPanel diagnostics={mcmcDiagnostics} />
          </AccordionContent>
        </AccordionItem>
      )}

      {/* SVI ELBO Loss Curve */}
      {sviDiagnostics && (
        <AccordionItem value="svi">
          <AccordionTrigger value="svi" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              ELBO Convergence
              <StatTooltip explanation="Evidence Lower Bound loss over SVI optimization steps. Should decrease and plateau." />
            </span>
          </AccordionTrigger>
          <AccordionContent value="svi">
            <ELBOLossChart diagnostics={sviDiagnostics} />
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Power Scaling Section */}
      <AccordionItem value="power-scaling">
        <AccordionTrigger value="power-scaling" className="text-sm">
          <span className="inline-flex items-center gap-1.5">
            Power Scaling Diagnostics
            <StatTooltip explanation="Tests whether posteriors are driven by data (good) or priors (concerning). Scales the likelihood and prior to detect sensitivity." />
          </span>
          <Badge
            variant={
              powerScaling.every((p) => p.diagnosis === "well_identified") ? "success" : "warning"
            }
            className="ml-2"
          >
            {powerScaling.filter((p) => p.diagnosis === "well_identified").length}/
            {powerScaling.length} OK
          </Badge>
        </AccordionTrigger>
        <AccordionContent value="power-scaling">
          <div className="space-y-4">
            {powerScaling.length >= 2 && <PowerScalingScatter results={powerScaling} />}
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Parameter</TableHead>
                  <TableHead>
                    <span className="inline-flex items-center gap-1">
                      Diagnosis
                      <StatTooltip explanation="'Well Identified' = data-driven. 'Prior Dominated' = prior choice matters too much. 'Prior-Data Conflict' = prior and data disagree." />
                    </span>
                  </TableHead>
                  <TableHead className="text-right">
                    <span className="inline-flex items-center gap-1">
                      Prior Sens.
                      <StatTooltip explanation="How much the posterior changes when the prior is scaled. Low values (<0.05) are good." />
                    </span>
                  </TableHead>
                  <TableHead className="text-right">
                    <span className="inline-flex items-center gap-1">
                      Lik. Sens.
                      <StatTooltip explanation="How much the posterior changes when the likelihood is scaled. High values indicate the data is informative." />
                    </span>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {powerScaling.map((ps) => (
                  <TableRow key={ps.parameter}>
                    <TableCell className="font-medium font-mono text-sm">{ps.parameter}</TableCell>
                    <TableCell>
                      <Badge variant={diagnosisBadgeVariant[ps.diagnosis] ?? "secondary"}>
                        {diagnosisLabel[ps.diagnosis] ?? ps.diagnosis}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-mono text-sm">
                      {formatNumber(ps.prior_sensitivity)}
                    </TableCell>
                    <TableCell className="text-right font-mono text-sm">
                      {formatNumber(ps.likelihood_sensitivity)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </AccordionContent>
      </AccordionItem>

      {/* PPC Warnings Table */}
      <AccordionItem value="ppc">
        <AccordionTrigger value="ppc" className="text-sm">
          <span className="inline-flex items-center gap-1.5">
            Posterior Predictive Checks
            <StatTooltip explanation="Simulates data from the fitted model and compares to observed data. Failures suggest model misspecification." />
          </span>
          <Badge variant={ppc.overall_passed ? "success" : "destructive"} className="ml-2">
            {ppc.overall_passed ? "Passed" : "Failed"}
          </Badge>
        </AccordionTrigger>
        <AccordionContent value="ppc">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Variable</TableHead>
                <TableHead>Check</TableHead>
                <TableHead className="text-right">Value</TableHead>
                <TableHead>Message</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ppc.per_variable_warnings.map((w, i) => (
                <TableRow
                  key={`ppc-${
                    // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                    i
                  }`}
                >
                  <TableCell className="font-medium">{w.variable}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{w.check_type}</Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm">
                    {w.value != null ? formatNumber(w.value) : "—"}
                  </TableCell>
                  <TableCell className="max-w-sm text-sm text-muted-foreground">
                    {w.message}
                  </TableCell>
                  <TableCell>
                    {w.passed ? (
                      <Check className="h-4 w-4 text-success" />
                    ) : (
                      <X className="h-4 w-4 text-destructive" />
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </AccordionContent>
      </AccordionItem>

      {/* PPC Ribbon Overlays (Gabry's ppc_dens_overlay) */}
      {overlayVars.length > 0 && (
        <AccordionItem value="ppc-overlays">
          <AccordionTrigger value="ppc-overlays" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              PPC Overlay Plots
              <StatTooltip explanation="Observed data (solid line) vs posterior predictive quantile bands. The shaded regions show where the model expects data to fall." />
            </span>
            <Badge variant="outline" className="ml-2">
              {overlayVars.length} variable{overlayVars.length !== 1 && "s"}
            </Badge>
          </AccordionTrigger>
          <AccordionContent value="ppc-overlays">
            <div className="space-y-6">
              {overlayVars.map((ov) => (
                <div key={ov.variable}>
                  <h4 className="mb-2 text-sm font-medium">{ov.variable}</h4>
                  <PPCRibbonChart overlay={ov} />
                </div>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* PPC Test Statistics (Gabry's ppc_stat) */}
      {testStatsByVar.size > 0 && (
        <AccordionItem value="ppc-stats">
          <AccordionTrigger value="ppc-stats" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              PPC Test Statistics
              <StatTooltip explanation="Distribution of test statistics (mean, sd, min, max) across posterior predictive draws. The vertical line shows the observed value. Extreme p-values suggest misspecification." />
            </span>
          </AccordionTrigger>
          <AccordionContent value="ppc-stats">
            <div className="space-y-6">
              {Array.from(testStatsByVar.entries()).map(([variable, stats]) => (
                <div key={variable}>
                  <h4 className="mb-3 text-sm font-medium">{variable}</h4>
                  <div className="grid gap-4 sm:grid-cols-2">
                    {stats.map((s) => (
                      <PPCTestStatChart key={`${s.variable}-${s.stat_name}`} stat={s} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}
    </Accordion>
  );
}
