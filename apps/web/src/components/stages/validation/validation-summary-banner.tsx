import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils/cn";
import type { ValidationReport, ValidationSeverity } from "@causal-ssm/api-types";
import { AlertTriangle, CheckCircle, Info, XCircle } from "lucide-react";

const severityIcon: Record<ValidationSeverity, typeof XCircle> = {
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
};

export function ValidationSummaryBanner({ report }: { report: ValidationReport }) {
  const counts: Record<ValidationSeverity, number> = { error: 0, warning: 0, info: 0 };
  for (const issue of report.issues) {
    counts[issue.severity]++;
  }

  return (
    <div
      className={cn(
        "flex items-center justify-between rounded-lg border p-4",
        report.is_valid
          ? "border-emerald-200 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950"
          : "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950",
      )}
    >
      <div className="flex items-center gap-3">
        {report.is_valid ? (
          <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
        ) : (
          <XCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
        )}
        <span className="font-medium">
          {report.is_valid ? "Validation Passed" : "Validation Failed"}
        </span>
      </div>

      <div className="flex items-center gap-3">
        {(["error", "warning", "info"] as const).map((severity) => {
          if (counts[severity] === 0) return null;
          const Icon = severityIcon[severity];
          return (
            <Badge
              key={severity}
              variant={
                severity === "error"
                  ? "destructive"
                  : severity === "warning"
                    ? "warning"
                    : "secondary"
              }
              className="gap-1"
            >
              <Icon className="h-3 w-3" />
              {counts[severity]}
            </Badge>
          );
        })}
      </div>
    </div>
  );
}
