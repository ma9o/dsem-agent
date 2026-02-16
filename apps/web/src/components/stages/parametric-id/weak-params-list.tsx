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
import { formatNumber } from "@/lib/utils/format";
import type { ParameterClassification, ParameterIdentification } from "@causal-ssm/api-types";

interface WeakParamsListProps {
  params: ParameterIdentification[];
}

const classificationVariant: Record<
  ParameterClassification,
  "success" | "warning" | "destructive"
> = {
  structurally_identified: "success",
  boundary: "warning",
  weak: "destructive",
};

const classificationLabel: Record<ParameterClassification, string> = {
  structurally_identified: "Identified",
  boundary: "Boundary",
  weak: "Weak",
};

export function WeakParamsList({ params }: WeakParamsListProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Parameter</TableHead>
          <TableHead>
            <span className="inline-flex items-center gap-1">
              Classification
              <StatTooltip explanation="Structurally identified: uniquely determined by the model. Boundary: near the identification boundary. Weak: poorly identified — posterior dominated by the prior." />
            </span>
          </TableHead>
          <TableHead>
            <span className="inline-flex items-center gap-1">
              Contraction
              <StatTooltip explanation="Prior-to-posterior contraction ratio. Values near 1 mean the data strongly informs the parameter; values near 0 mean the posterior is dominated by the prior (weak identification)." />
            </span>
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {params.map((param) => (
          <TableRow key={param.name}>
            <TableCell className="font-medium font-mono text-sm">{param.name}</TableCell>
            <TableCell>
              <Badge variant={classificationVariant[param.classification]}>
                {classificationLabel[param.classification]}
              </Badge>
            </TableCell>
            <TableCell className="text-sm text-muted-foreground">
              {param.contraction_ratio !== null ? formatNumber(param.contraction_ratio) : "--"}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
