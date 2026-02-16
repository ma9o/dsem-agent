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
import type { LikelihoodSpec } from "@causal-ssm/api-types";

interface LikelihoodTableProps {
  likelihoods: LikelihoodSpec[];
}

export function LikelihoodTable({ likelihoods }: LikelihoodTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Variable</TableHead>
          <TableHead>
            <span className="inline-flex items-center gap-1">
              Distribution
              <StatTooltip explanation="The probability distribution family chosen by the pipeline for this observed variable (e.g., Gaussian for continuous, Poisson for counts, Bernoulli for binary)." />
            </span>
          </TableHead>
          <TableHead>
            <span className="inline-flex items-center gap-1">
              Link
              <StatTooltip explanation="The link function mapping the latent linear predictor to the distribution's natural parameter (e.g., identity for Gaussian, log for Poisson, logit for Bernoulli)." />
            </span>
          </TableHead>
          <TableHead>Reasoning</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {likelihoods.map((lik) => (
          <TableRow key={lik.variable}>
            <TableCell className="font-medium">{lik.variable}</TableCell>
            <TableCell>
              <Badge variant="outline">{lik.distribution}</Badge>
            </TableCell>
            <TableCell>
              <Badge variant="secondary">{lik.link}</Badge>
            </TableCell>
            <TableCell className="max-w-sm text-sm text-muted-foreground">
              {lik.reasoning}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
