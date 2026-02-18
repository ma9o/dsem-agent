import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import type { ParameterSpec } from "@causal-ssm/api-types";

const col = createColumnHelper<ParameterSpec>();

const baseColumns = [
  col.accessor("name", {
    header: "Name",
    cell: (info) => (
      <span className="font-medium font-mono">{info.getValue()}</span>
    ),
  }),
  col.accessor("role", {
    header: "Role",
    cell: (info) => <Badge variant="outline">{info.getValue()}</Badge>,
  }),
  col.accessor("constraint", {
    header: "Constraint",
    cell: (info) => <Badge variant="secondary">{info.getValue()}</Badge>,
  }),
  col.accessor("description", {
    header: "Description",
    cell: (info) => (
      <span className="max-w-sm text-muted-foreground">{info.getValue()}</span>
    ),
  }),
];

const searchContextColumn = col.accessor("search_context", {
  header: () => (
    <HeaderWithTooltip
      label="Search Context"
      tooltip="The search query used by the pipeline to find prior literature for this parameter's effect size."
    />
  ),
  cell: (info) => (
    <span className="max-w-xs text-xs text-muted-foreground italic">
      {info.getValue() || "--"}
    </span>
  ),
});

export function ParameterTable({ parameters }: { parameters: ParameterSpec[] }) {
  const hasSearchContext = parameters.some((p) => p.search_context);

  const columns = useMemo<ColumnDef<ParameterSpec, unknown>[]>(
    () =>
      hasSearchContext
        ? [...baseColumns, searchContextColumn]
        : baseColumns,
    [hasSearchContext],
  );

  return <InfoTable columns={columns} data={parameters} />;
}
