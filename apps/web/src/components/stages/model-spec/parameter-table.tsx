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
import type { ParameterSpec } from "@causal-ssm/api-types";

interface ParameterTableProps {
  parameters: ParameterSpec[];
}

export function ParameterTable({ parameters }: ParameterTableProps) {
  const hasSearchContext = parameters.some((p) => p.search_context);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Role</TableHead>
          <TableHead>Constraint</TableHead>
          <TableHead>Description</TableHead>
          {hasSearchContext && (
            <TableHead>
              <span className="inline-flex items-center gap-1">
                Search Context
                <StatTooltip explanation="The search query used by the pipeline to find prior literature for this parameter's effect size." />
              </span>
            </TableHead>
          )}
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
            {hasSearchContext && (
              <TableCell className="max-w-xs text-xs text-muted-foreground italic">
                {param.search_context || "--"}
              </TableCell>
            )}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
