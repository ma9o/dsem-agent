import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper } from "@tanstack/react-table";
import { formatNumber } from "@/lib/utils/format";
import type { ParameterClassification, ParameterIdentification } from "@causal-ssm/api-types";

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

const col = createColumnHelper<ParameterIdentification>();

const columns = [
  col.accessor("name", {
    header: "Parameter",
    cell: (info) => (
      <span className="font-medium font-mono">{info.getValue()}</span>
    ),
  }),
  col.accessor("classification", {
    header: () => (
      <HeaderWithTooltip
        label="Classification"
        tooltip="Structurally identified: uniquely determined by the model. Boundary: near the identification boundary. Weak: poorly identified — posterior dominated by the prior."
      />
    ),
    cell: (info) => (
      <Badge variant={classificationVariant[info.getValue()]}>
        {classificationLabel[info.getValue()]}
      </Badge>
    ),
  }),
  col.accessor("contraction_ratio", {
    header: () => (
      <HeaderWithTooltip
        label="Contraction"
        tooltip="Prior-to-posterior contraction ratio. Values near 1 mean the data strongly informs the parameter; values near 0 mean the posterior is dominated by the prior (weak identification)."
      />
    ),
    cell: (info) => {
      const v = info.getValue();
      return (
        <span className="text-muted-foreground">
          {v !== null ? formatNumber(v) : "--"}
        </span>
      );
    },
  }),
];

export function WeakParamsList({ params }: { params: ParameterIdentification[] }) {
  return <InfoTable columns={columns} data={params} />;
}
