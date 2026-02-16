import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { CausalEdge } from "@causal-ssm/api-types";

interface EdgeListProps {
  edges: CausalEdge[];
}

export function EdgeList({ edges }: EdgeListProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Cause</TableHead>
          <TableHead>Effect</TableHead>
          <TableHead>Timing</TableHead>
          <TableHead>Description</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {edges.map((edge) => (
          <TableRow key={`${edge.cause}->${edge.effect}`}>
            <TableCell className="font-medium">{edge.cause}</TableCell>
            <TableCell className="font-medium">{edge.effect}</TableCell>
            <TableCell>
              <Badge variant={edge.lagged ? "default" : "secondary"}>
                {edge.lagged ? "Lagged" : "Contemporaneous"}
              </Badge>
            </TableCell>
            <TableCell className="max-w-xs text-sm text-muted-foreground">
              {edge.description}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
