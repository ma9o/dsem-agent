import { IssueBadge } from "@/components/ui/custom/issue-badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ValidationIssue } from "@causal-ssm/api-types";

export function IssueTable({ issues }: { issues: ValidationIssue[] }) {
  const sorted = [...issues].sort((a, b) => {
    const order: Record<string, number> = { error: 0, warning: 1, info: 2 };
    return (order[a.severity] ?? 3) - (order[b.severity] ?? 3);
  });

  if (sorted.length === 0) {
    return <p className="text-sm text-muted-foreground">No issues found.</p>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Indicator</TableHead>
          <TableHead>Issue Type</TableHead>
          <TableHead>Severity</TableHead>
          <TableHead>Message</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((issue, i) => (
          <TableRow key={`${issue.indicator}-${issue.issue_type}-${i}`}>
            <TableCell className="font-medium">{issue.indicator}</TableCell>
            <TableCell className="font-mono text-xs">{issue.issue_type}</TableCell>
            <TableCell>
              <IssueBadge severity={issue.severity} />
            </TableCell>
            <TableCell className="max-w-md text-sm text-muted-foreground">
              {issue.message}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
