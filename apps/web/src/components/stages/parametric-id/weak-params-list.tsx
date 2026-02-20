"use client";

import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import { formatNumber } from "@/lib/utils/format";
import type { ParameterClassification, ParameterIdentification } from "@causal-ssm/api-types";
import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

const classificationVariant: Record<
  ParameterClassification,
  "success" | "warning" | "destructive"
> = {
  identified: "success",
  practically_unidentifiable: "warning",
  structurally_unidentifiable: "destructive",
};

const classificationLabel: Record<ParameterClassification, string> = {
  identified: "Identified",
  practically_unidentifiable: "Practically Unidentifiable",
  structurally_unidentifiable: "Structurally Unidentifiable",
};

const classificationStroke: Record<ParameterClassification, string> = {
  identified: "var(--success)",
  practically_unidentifiable: "var(--warning)",
  structurally_unidentifiable: "var(--destructive)",
};

function ProfileSparkline({
  param,
  threshold,
}: {
  param: ParameterIdentification;
  threshold: number;
}) {
  if (!param.profile_x || !param.profile_ll) return <span className="text-muted-foreground">--</span>;

  const data = param.profile_x.map((x, i) => ({
    x,
    ll: param.profile_ll![i],
  }));

  const stroke = classificationStroke[param.classification];

  return (
    <div className="h-14 w-32">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 2, left: 0, bottom: 0 }}>
          <XAxis dataKey="x" hide />
          <YAxis hide domain={["dataMin", "auto"]} />
          <ReferenceLine
            y={-threshold}
            stroke="var(--muted-foreground)"
            strokeDasharray="3 3"
            strokeWidth={1}
          />
          <Line
            type="monotone"
            dataKey="ll"
            stroke={stroke}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const col = createColumnHelper<ParameterIdentification>();

function buildColumns(threshold: number) {
  return [
    col.accessor("name", {
      header: "Parameter",
      cell: (info) => (
        <span className="font-medium font-mono">{info.getValue()}</span>
      ),
    }),
    col.display({
      id: "profile",
      header: () => (
        <HeaderWithTooltip
          label="Profile Likelihood"
          tooltip="Profile likelihood curve: shows how log-likelihood changes as each parameter is varied with others optimized out. The dashed line is the chi-squared threshold — if the curve stays above it, the parameter is unidentifiable."
        />
      ),
      cell: (info) => (
        <ProfileSparkline param={info.row.original} threshold={threshold} />
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
            {v != null ? formatNumber(v) : "--"}
          </span>
        );
      },
    }),
  ];
}

export function WeakParamsList({
  params,
  threshold,
}: {
  params: ParameterIdentification[];
  threshold?: number | null;
}) {
  const columns = buildColumns(threshold ?? 1.92);
  return <InfoTable columns={columns as ColumnDef<ParameterIdentification, unknown>[]} data={params} />;
}
