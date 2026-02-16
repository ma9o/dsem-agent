import { Badge } from "@/components/ui/badge";
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
          <TableHead>Distribution</TableHead>
          <TableHead>Link Function</TableHead>
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
