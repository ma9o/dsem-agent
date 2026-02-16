"use client";

import { CausalDag } from "@/components/dag/causal-dag";
import { IdentifiabilityPanel } from "@/components/stages/measurement/identifiability-panel";
import { IndicatorTable } from "@/components/stages/measurement/indicator-table";
import { HardGateAlert } from "@/components/ui/custom/hard-gate-alert";
import type { Stage1bData } from "@causal-ssm/api-types";

export default function Stage1bContent({ data }: { data: Stage1bData }) {
  const spec = data.causal_spec;
  const nonId = spec.identifiability?.non_identifiable_treatments ?? {};
  const hasNonIdentifiable = Object.keys(nonId).length > 0;

  return (
    <div className="space-y-4">
      <div className="h-[400px] rounded-lg border">
        <CausalDag
          constructs={spec.latent.constructs}
          edges={spec.latent.edges}
          indicators={spec.measurement.indicators}
        />
      </div>
      <IndicatorTable indicators={spec.measurement.indicators} />
      {spec.identifiability && <IdentifiabilityPanel identifiability={spec.identifiability} />}
      {hasNonIdentifiable && (
        <HardGateAlert
          title="Non-Identifiable Treatment Effects Removed"
          explanation={`${Object.keys(nonId).length} treatment(s) have non-identifiable causal effects and were filtered out: ${Object.keys(nonId).join(", ")}.`}
          suggestion="Add instrumental variables or remove unobserved confounders to enable identification."
        />
      )}
    </div>
  );
}
