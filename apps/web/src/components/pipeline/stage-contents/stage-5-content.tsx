import { DiagnosticsAccordion } from "@/components/stages/inference/diagnostics-accordion";
import { ForestPlotPanel } from "@/components/stages/inference/forest-plot-panel";
import { TreatmentRankingTable } from "@/components/stages/inference/treatment-ranking-table";
import type { Stage5Data } from "@causal-ssm/api-types";

export default function Stage5Content({ data }: { data: Stage5Data }) {
  if (data.intervention_results.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No treatment effects were estimated. This may happen if no treatments passed
        identification checks.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <TreatmentRankingTable results={data.intervention_results} />
      <ForestPlotPanel results={data.intervention_results} />
      <DiagnosticsAccordion
        powerScaling={data.power_scaling}
        ppc={data.ppc}
        mcmcDiagnostics={data.mcmc_diagnostics}
        sviDiagnostics={data.svi_diagnostics}
        looDiagnostics={data.loo_diagnostics}
        posteriorMarginals={data.posterior_marginals}
        posteriorPairs={data.posterior_pairs}
      />
      <div className="rounded-lg bg-muted p-3 text-xs text-muted-foreground">
        Inference: {data.inference_metadata.method} | {data.inference_metadata.n_samples} samples |{" "}
        {data.inference_metadata.duration_seconds.toFixed(1)}s
      </div>
    </div>
  );
}
