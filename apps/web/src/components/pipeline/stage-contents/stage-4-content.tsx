import { LikelihoodTable } from "@/components/stages/model-spec/likelihood-table";
import { ParameterTable } from "@/components/stages/model-spec/parameter-table";
import { PriorCard } from "@/components/stages/model-spec/prior-card";
import { RetryIndicator } from "@/components/stages/model-spec/retry-indicator";
import { SSMEquationDisplay } from "@/components/stages/model-spec/ssm-equation-display";
import type { Stage4Data } from "@causal-ssm/api-types";

export default function Stage4Content({ data }: { data: Stage4Data }) {
  return (
    <div className="space-y-4">
      {data.ssm_equations.length > 0 && <SSMEquationDisplay equations={data.ssm_equations} />}
      <LikelihoodTable likelihoods={data.model_spec.likelihoods} />
      <ParameterTable parameters={data.model_spec.parameters} />
      {data.priors.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Prior Distributions</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {data.priors.map((p) => (
              <PriorCard key={p.parameter} prior={p} />
            ))}
          </div>
        </div>
      )}
      {data.validation_retries.length > 0 && <RetryIndicator retries={data.validation_retries} />}
      {data.model_spec.reasoning && (
        <div className="rounded-lg bg-muted p-4 text-sm">
          <p className="font-medium mb-1">Model Reasoning</p>
          <p className="text-muted-foreground">{data.model_spec.reasoning}</p>
        </div>
      )}
    </div>
  );
}
