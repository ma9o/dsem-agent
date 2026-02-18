import { Badge } from "@/components/ui/badge";
import { InfoTable } from "@/components/ui/info-table";
import { createColumnHelper } from "@tanstack/react-table";
import { formatNumber } from "@/lib/utils/format";
import type { PPCWarning } from "@causal-ssm/api-types";
import { Check, X } from "lucide-react";

const col = createColumnHelper<PPCWarning>();

const columns = [
  col.accessor("variable", {
    header: "Variable",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("check_type", {
    header: "Check",
    cell: (info) => <Badge variant="outline">{info.getValue()}</Badge>,
  }),
  col.accessor("value", {
    header: "Value",
    cell: (info) => {
      const v = info.getValue();
      return v != null ? formatNumber(v) : "—";
    },
    meta: { align: "right", mono: true },
  }),
  col.accessor("message", {
    header: "Message",
    cell: (info) => (
      <span className="max-w-sm text-muted-foreground">{info.getValue()}</span>
    ),
  }),
  col.accessor("passed", {
    header: "Status",
    cell: (info) =>
      info.getValue() ? (
        <Check className="h-4 w-4 text-success" />
      ) : (
        <X className="h-4 w-4 text-destructive" />
      ),
  }),
];

export function PPCWarningsTable({ warnings }: { warnings: PPCWarning[] }) {
  return <InfoTable columns={columns} data={warnings} />;
}
