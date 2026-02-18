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
  // Determine which sections to open by default
  const defaultOpen = ["power-scaling", "ppc"];
  if (mcmcDiagnostics) defaultOpen.push("mcmc");
  if (sviDiagnostics) defaultOpen.push("svi");
  if (ppc.overlays?.length) defaultOpen.push("ppc-overlays");
  if (ppc.test_stats?.length) defaultOpen.push("ppc-stats");
  if (looDiagnostics) defaultOpen.push("loo");

  // Deduplicate overlay variables for grouped display
  const overlayVars = ppc.overlays ?? [];
  const testStatsByVar = new Map<string, typeof ppc.test_stats>();
  for (const ts of ppc.test_stats ?? []) {
    const existing = testStatsByVar.get(ts.variable) ?? [];
    existing.push(ts);
    testStatsByVar.set(ts.variable, existing);
  }

  const hasTraces = mcmcDiagnostics?.trace_data && mcmcDiagnostics.trace_data.length > 0;
  const hasRankHists = mcmcDiagnostics?.rank_histograms && mcmcDiagnostics.rank_histograms.length > 0;
  const hasEnergy = mcmcDiagnostics?.energy != null;
  const hasMarginals = posteriorMarginals && posteriorMarginals.length > 0;
  const hasPairs = posteriorPairs && posteriorPairs.length > 0;

  return (
    <Accordion defaultOpen={defaultOpen}>
      {/* MCMC Convergence Diagnostics */}
      {mcmcDiagnostics && (
        <AccordionItem value="mcmc">
          <AccordionTrigger value="mcmc" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              MCMC Convergence
              <StatTooltip explanation="Chain mixing and convergence diagnostics: R-hat, effective sample size (bulk + tail), MCSE, divergences, and tree depth from NUTS/HMC." />
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

      {/* NUTS Energy Diagnostics (Betancourt 2017) */}
      {hasEnergy && (
        <AccordionItem value="energy">
          <AccordionTrigger value="energy" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              Energy Diagnostics
              <StatTooltip explanation="NUTS energy diagnostics (Betancourt 2017). Compares marginal energy E and transition dE distributions. Large discrepancy or BFMI < 0.3 indicates the sampler struggles to explore the target geometry." />
            </span>
            <Badge
              variant={Math.min(...mcmcDiagnostics!.energy!.bfmi) >= 0.3 ? "success" : "destructive"}
              className="ml-2"
            >
              BFMI {Math.min(...mcmcDiagnostics!.energy!.bfmi) >= 0.3 ? "OK" : "Low"}
            </Badge>
          </AccordionTrigger>
          <AccordionContent value="energy">
            <EnergyChart energy={mcmcDiagnostics!.energy!} />
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Trace Plots */}
      {hasTraces && (
        <AccordionItem value="traces">
          <AccordionTrigger value="traces" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              Trace Plots
              <StatTooltip explanation="MCMC chain trajectories over sampling iterations. Well-mixed chains should look like 'hairy caterpillars' with no trends or stuck regions." />
            </span>
            <Badge variant="outline" className="ml-2">
              {mcmcDiagnostics!.trace_data!.length} params
            </Badge>
          </AccordionTrigger>
          <AccordionContent value="traces">
            <div className="grid gap-4 sm:grid-cols-2">
              {mcmcDiagnostics!.trace_data!.map((trace) => (
                <TracePlot key={trace.parameter} trace={trace} />
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Rank Histograms */}
      {hasRankHists && (
        <AccordionItem value="rank-hists">
          <AccordionTrigger value="rank-hists" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              Rank Histograms
              <StatTooltip explanation="Rank histograms (Vehtari et al. 2021). Samples are ranked across all chains and binned per chain. Uniform histograms indicate good mixing." />
            </span>
          </AccordionTrigger>
          <AccordionContent value="rank-hists">
            <div className="grid gap-4 sm:grid-cols-2">
              {mcmcDiagnostics!.rank_histograms!.map((hist) => (
                <RankHistogram key={hist.parameter} histogram={hist} />
              ))}
            </div>
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

      {/* LOO-CV Diagnostics */}
      {looDiagnostics && (
        <AccordionItem value="loo">
          <AccordionTrigger value="loo" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              LOO Cross-Validation
              <StatTooltip explanation="Leave-one-out cross-validation via Pareto-smoothed importance sampling. Assesses predictive accuracy and identifies influential observations." />
            </span>
            <Badge
              variant={
                looDiagnostics.n_bad_k != null && looDiagnostics.n_bad_k === 0
                  ? "success"
                  : "warning"
              }
              className="ml-2"
            >
              ELPD = {formatNumber(looDiagnostics.elpd_loo, 1)}
            </Badge>
          </AccordionTrigger>
          <AccordionContent value="loo">
            <div className="space-y-4">
              {looDiagnostics.loo_pit && <LOOPITChart loo={looDiagnostics} />}
              {looDiagnostics.pareto_k && <ParetoKChart loo={looDiagnostics} />}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Posterior Marginals */}
      {hasMarginals && (
        <AccordionItem value="posteriors">
          <AccordionTrigger value="posteriors" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              Posterior Distributions
              <StatTooltip explanation="Marginal posterior density for each parameter with 94% HDI. Vertical line shows the posterior mean." />
            </span>
            <Badge variant="outline" className="ml-2">
              {posteriorMarginals!.length} params
            </Badge>
          </AccordionTrigger>
          <AccordionContent value="posteriors">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {posteriorMarginals!.map((m) => (
                <PosteriorDensityChart key={m.parameter} marginal={m} />
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Posterior Pairs */}
      {hasPairs && (
        <AccordionItem value="pairs">
          <AccordionTrigger value="pairs" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              Posterior Pairs
              <StatTooltip explanation="Pairwise scatter plots of posterior samples. Reveals correlations and potential identifiability issues between parameters." />
            </span>
            <Badge variant="outline" className="ml-2">
              {posteriorPairs!.length} pairs
            </Badge>
          </AccordionTrigger>
          <AccordionContent value="pairs">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {posteriorPairs!.map((p) => (
                <PosteriorPairsChart key={`${p.param_x}-${p.param_y}`} pair={p} />
              ))}
            </div>
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
            <PowerScalingTable results={powerScaling} />
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
          <PPCWarningsTable warnings={ppc.per_variable_warnings} />
        </AccordionContent>
      </AccordionItem>

      {/* PPC Ribbon Overlays (Gabry's ppc_dens_overlay) */}
      {overlayVars.length > 0 && (
        <AccordionItem value="ppc-overlays">
          <AccordionTrigger value="ppc-overlays" className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              PPC Overlay Plots
              <StatTooltip explanation="Observed data (solid line) vs posterior predictive quantile bands and individual y_rep draws. The shaded regions show where the model expects data to fall." />
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
