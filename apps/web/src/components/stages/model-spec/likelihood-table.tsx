import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper } from "@tanstack/react-table";
import type { LikelihoodSpec } from "@causal-ssm/api-types";

const col = createColumnHelper<LikelihoodSpec>();

const columns = [
  col.accessor("variable", {
    header: "Variable",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("distribution", {
    header: () => (
      <HeaderWithTooltip
        label="Distribution"
        tooltip="The probability distribution family chosen by the pipeline for this observed variable (e.g., Gaussian for continuous, Poisson for counts, Bernoulli for binary)."
      />
    ),
    cell: (info) => <Badge variant="outline">{info.getValue()}</Badge>,
  }),
  col.accessor("link", {
    header: () => (
      <HeaderWithTooltip
        label="Link"
        tooltip="The link function mapping the latent linear predictor to the distribution's natural parameter (e.g., identity for Gaussian, log for Poisson, logit for Bernoulli)."
      />
    ),
    cell: (info) => <Badge variant="secondary">{info.getValue()}</Badge>,
  }),
  col.accessor("reasoning", {
    header: "Reasoning",
    cell: (info) => (
      <span className="max-w-sm text-muted-foreground">{info.getValue()}</span>
    ),
  }),
];

export function LikelihoodTable({ likelihoods }: { likelihoods: LikelihoodSpec[] }) {
  return <InfoTable columns={columns} data={likelihoods} />;
}
