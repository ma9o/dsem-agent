import { useMemo } from "react";
import { IssueBadge } from "@/components/ui/custom/issue-badge";
import { InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import type { ValidationIssue } from "@causal-ssm/api-types";

const col = createColumnHelper<ValidationIssue>();

const columns = [
  col.accessor("indicator", {
    header: "Indicator",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("issue_type", {
    header: "Issue Type",
    cell: (info) => <span className="font-mono text-xs">{info.getValue()}</span>,
  }),
  col.accessor("severity", {
    header: "Severity",
    cell: (info) => <IssueBadge severity={info.getValue()} />,
  }),
  col.accessor("message", {
    header: "Message",
    cell: (info) => (
      <span className="max-w-md text-muted-foreground">{info.getValue()}</span>
    ),
  }),
];

export function IssueTable({ issues }: { issues: ValidationIssue[] }) {
  const sorted = useMemo(() => {
    const order: Record<string, number> = { error: 0, warning: 1, info: 2 };
    return [...issues].sort(
      (a, b) => (order[a.severity] ?? 3) - (order[b.severity] ?? 3),
    );
  }, [issues]);

  if (sorted.length === 0) {
    return <p className="text-sm text-muted-foreground">No issues found.</p>;
  }

  return <InfoTable columns={columns as ColumnDef<ValidationIssue, unknown>[]} data={sorted} />;
}
