import { MeasurementTable } from "@/components/stages/model-spec/measurement-table";
import { ParameterTable } from "@/components/stages/model-spec/parameter-table";
import { PriorTable } from "@/components/stages/model-spec/prior-table";
import { RetryIndicator } from "@/components/stages/model-spec/retry-indicator";
import { SSMEquationDisplay } from "@/components/stages/model-spec/ssm-equation-display";
import type { Extraction } from "@causal-ssm/api-types";
import type { Stage4Data } from "@causal-ssm/api-types";

export default function Stage4Content({
  data,
  extractions,
}: {
  data: Stage4Data;
  extractions?: Extraction[];
}) {
  return (
    <div className="space-y-4">
      <SSMEquationDisplay
        likelihoods={data.model_spec.likelihoods}
        parameters={data.model_spec.parameters}
        priors={data.priors}
      />
      {extractions && extractions.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Measurement Model</h3>
          <MeasurementTable
            likelihoods={data.model_spec.likelihoods}
            extractions={extractions}
            priorPredictiveSamples={data.prior_predictive_samples}
          />
        </div>
      )}
      <ParameterTable parameters={data.model_spec.parameters} />
      {data.priors.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Prior Distributions</h3>
          <PriorTable priors={data.priors} />
        </div>
      )}
      {data.validation_retries && data.validation_retries.length > 0 && (
        <RetryIndicator retries={data.validation_retries} />
      )}
    </div>
  );
}
