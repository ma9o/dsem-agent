import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ParameterSpec } from "@causal-ssm/api-types";

interface ParameterTableProps {
  parameters: ParameterSpec[];
}

export function ParameterTable({ parameters }: ParameterTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Role</TableHead>
          <TableHead>Constraint</TableHead>
          <TableHead>Description</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {parameters.map((param) => (
          <TableRow key={param.name}>
            <TableCell className="font-medium font-mono text-sm">{param.name}</TableCell>
            <TableCell>
              <Badge variant="outline">{param.role}</Badge>
            </TableCell>
            <TableCell>
              <Badge variant="secondary">{param.constraint}</Badge>
            </TableCell>
            <TableCell className="max-w-sm text-sm text-muted-foreground">
              {param.description}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
