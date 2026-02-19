"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import type {
  LOODiagnostics,
  MCMCDiagnostics,
  PPCResult,
  PosteriorMarginal,
  PosteriorPair,
  PowerScalingResult,
  SVIDiagnostics,
} from "@causal-ssm/api-types";
import { ELBOLossChart } from "@/components/charts/elbo-loss-chart";
import { EnergyChart } from "@/components/charts/energy-chart";
import { LOOPITChart } from "@/components/charts/loo-pit-chart";
import { MCMCDiagnosticsPanel } from "@/components/charts/mcmc-diagnostics-panel";
import { ParetoKChart } from "@/components/charts/pareto-k-chart";
import { PPCRibbonChart } from "@/components/charts/ppc-ribbon-chart";
import { PPCTestStatChart } from "@/components/charts/ppc-test-stat-chart";
import { PosteriorDensityChart } from "@/components/charts/posterior-density-chart";
import { PosteriorPairsChart } from "@/components/charts/posterior-pairs-chart";
import { PowerScalingScatter } from "@/components/charts/power-scaling-scatter";
import { RankHistogram } from "@/components/charts/rank-histogram";
import { TracePlot } from "@/components/charts/trace-plot";
import { PowerScalingTable } from "./power-scaling-table";
import { PPCWarningsTable } from "./ppc-warnings-table";

interface DiagnosticsAccordionProps {
  powerScaling: PowerScalingResult[];
  ppc: PPCResult;
  mcmcDiagnostics?: MCMCDiagnostics | null;
  sviDiagnostics?: SVIDiagnostics | null;
  looDiagnostics?: LOODiagnostics | null;
  posteriorMarginals?: PosteriorMarginal[] | null;
  posteriorPairs?: PosteriorPair[] | null;
}

export function DiagnosticsAccordion({
  powerScaling,
  ppc,
  mcmcDiagnostics,
  sviDiagnostics,
  looDiagnostics,
  posteriorMarginals,
  posteriorPairs,
}: DiagnosticsAccordionProps) {
  const hasTraces = mcmcDiagnostics?.trace_data && mcmcDiagnostics.trace_data.length > 0;
  const hasRankHists = mcmcDiagnostics?.rank_histograms && mcmcDiagnostics.rank_histograms.length > 0;
  const hasEnergy = mcmcDiagnostics?.energy != null;
  const hasMarginals = posteriorMarginals && posteriorMarginals.length > 0;
  const hasPairs = posteriorPairs && posteriorPairs.length > 0;

  const overlayVars = ppc.overlays ?? [];
  const testStatsByVar = new Map<string, typeof ppc.test_stats>();
  for (const ts of ppc.test_stats ?? []) {
    const existing = testStatsByVar.get(ts.variable) ?? [];
    existing.push(ts);
    testStatsByVar.set(ts.variable, existing);
  }

  // Build paired trace + rank histogram data per parameter
  const mcmcParamPairs = (() => {
    if (!hasTraces && !hasRankHists) return [];
    const traceByParam = new Map(
      (mcmcDiagnostics?.trace_data ?? []).map((t) => [t.parameter, t]),
    );
    const rankByParam = new Map(
      (mcmcDiagnostics?.rank_histograms ?? []).map((h) => [h.parameter, h]),
    );
    const params = new Set([...traceByParam.keys(), ...rankByParam.keys()]);
    return Array.from(params).map((param) => ({
      parameter: param,
      trace: traceByParam.get(param),
      rank: rankByParam.get(param),
    }));
  })();

  // All sections default open
  const defaultOpen = ["mcmc", "svi", "loo", "posteriors", "power-scaling", "ppc"];

  return (
    <Accordion defaultOpen={defaultOpen}>
      {/* ── MCMC Diagnostics (convergence + energy + traces + rank histograms) ── */}
      {mcmcDiagnostics && (
        <AccordionItem value="mcmc">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              MCMC Diagnostics
              <StatTooltip explanation="Chain convergence (R-hat, ESS, MCSE), energy diagnostics, trace plots, and rank histograms for NUTS/HMC sampling." />
              <Badge
                variant={mcmcDiagnostics.num_divergences === 0 ? "success" : "destructive"}
              >
                {mcmcDiagnostics.num_divergences === 0 ? "Converged" : `${mcmcDiagnostics.num_divergences} divergences`}
              </Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4">
              <MCMCDiagnosticsPanel diagnostics={mcmcDiagnostics} />
              {hasEnergy && <EnergyChart energy={mcmcDiagnostics.energy!} />}

              {/* Paired trace + rank histogram per parameter */}
              {mcmcParamPairs.length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Per-parameter traces & rank histograms
                  </h4>
                  {mcmcParamPairs.map(({ parameter, trace, rank }) => (
                    <div key={parameter} className="grid gap-4 sm:grid-cols-2">
                      {trace && <TracePlot trace={trace} />}
                      {rank && <RankHistogram histogram={rank} />}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── ELBO Convergence (SVI only) ── */}
      {sviDiagnostics && (
        <AccordionItem value="svi">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              ELBO Convergence
              <StatTooltip explanation="Evidence Lower Bound loss over SVI optimization steps. Should decrease and plateau." />
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <ELBOLossChart diagnostics={sviDiagnostics} />
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── LOO Cross-Validation (PIT + Pareto-K side by side) ── */}
      {looDiagnostics && (
        <AccordionItem value="loo">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              LOO Cross-Validation
              <StatTooltip explanation="Leave-one-out cross-validation via Pareto-smoothed importance sampling. Assesses predictive accuracy and identifies influential observations." />
              <Badge
                variant={
                  looDiagnostics.n_bad_k != null && looDiagnostics.n_bad_k === 0
                    ? "success"
                    : "warning"
                }
              >
                ELPD = {formatNumber(looDiagnostics.elpd_loo, 1)}
              </Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge variant="outline">ELPD = {formatNumber(looDiagnostics.elpd_loo, 1)}</Badge>
                <Badge variant="outline">p_loo = {formatNumber(looDiagnostics.p_loo, 1)}</Badge>
                <Badge variant="outline">SE = {formatNumber(looDiagnostics.se, 1)}</Badge>
                {looDiagnostics.n_bad_k != null && (
                  <Badge variant={looDiagnostics.n_bad_k === 0 ? "success" : "destructive"}>
                    {looDiagnostics.n_bad_k === 0 ? "All Pareto k OK" : `${looDiagnostics.n_bad_k} bad Pareto k`}
                  </Badge>
                )}
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                {looDiagnostics.loo_pit && <LOOPITChart loo={looDiagnostics} />}
                {looDiagnostics.pareto_k && <ParetoKChart loo={looDiagnostics} />}
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── Posterior Exploration (marginals + pairs) ── */}
      {(hasMarginals || hasPairs) && (
        <AccordionItem value="posteriors">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              Posterior Exploration
              <StatTooltip explanation="Marginal posterior densities with 94% HDI, and pairwise scatter plots revealing parameter correlations and identifiability issues." />
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4">
              {hasMarginals && (
                <div>
                  <h4 className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Marginal distributions
                  </h4>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {posteriorMarginals!.map((m) => (
                      <PosteriorDensityChart key={m.parameter} marginal={m} />
                    ))}
                  </div>
                </div>
              )}
              {hasPairs && (
                <div>
                  <h4 className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Pairwise correlations
                  </h4>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {posteriorPairs!.map((p) => (
                      <PosteriorPairsChart key={`${p.param_x}-${p.param_y}`} pair={p} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── Power Scaling (scatter + table side by side) ── */}
      <AccordionItem value="power-scaling">
        <AccordionTrigger className="text-sm">
          <span className="inline-flex items-center gap-1.5 flex-wrap">
            Power Scaling Diagnostics
            <StatTooltip explanation="Tests whether posteriors are driven by data (good) or priors (concerning). Scales the likelihood and prior to detect sensitivity." />
            <Badge
              variant={
                powerScaling.every((p) => p.diagnosis === "well_identified") ? "success" : "warning"
              }
            >
              {powerScaling.filter((p) => p.diagnosis === "well_identified").length}/
              {powerScaling.length} OK
            </Badge>
          </span>
        </AccordionTrigger>
        <AccordionContent>
          {powerScaling.length >= 2 ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <PowerScalingScatter results={powerScaling} />
              <PowerScalingTable results={powerScaling} />
            </div>
          ) : (
            <PowerScalingTable results={powerScaling} />
          )}
        </AccordionContent>
      </AccordionItem>

      {/* ── Posterior Predictive Checks (warnings + overlays + test stats) ── */}
      <AccordionItem value="ppc">
        <AccordionTrigger className="text-sm">
          <span className="inline-flex items-center gap-1.5 flex-wrap">
            Posterior Predictive Checks
            <StatTooltip explanation="Simulates data from the fitted model and compares to observed data. Includes per-variable warnings, overlay plots, and test statistics." />
            <Badge variant={ppc.overall_passed ? "success" : "destructive"}>
              {ppc.overall_passed ? "Passed" : "Failed"}
            </Badge>
          </span>
        </AccordionTrigger>
        <AccordionContent>
          <div className="space-y-6">
            <PPCWarningsTable warnings={ppc.per_variable_warnings} />

            {overlayVars.length > 0 && (
              <div>
                <h4 className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Overlay plots
                </h4>
                <div className="space-y-6">
                  {overlayVars.map((ov) => (
                    <div key={ov.variable}>
                      <h5 className="mb-2 text-sm font-medium">{ov.variable}</h5>
                      <PPCRibbonChart overlay={ov} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {testStatsByVar.size > 0 && (
              <div>
                <h4 className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Test statistics
                </h4>
                <div className="space-y-6">
                  {Array.from(testStatsByVar.entries()).map(([variable, stats]) => (
                    <div key={variable}>
                      <h5 className="mb-3 text-sm font-medium">{variable}</h5>
                      <div className="grid gap-4 sm:grid-cols-2">
                        {stats.map((s) => (
                          <PPCTestStatChart key={`${s.variable}-${s.stat_name}`} stat={s} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
