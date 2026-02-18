import { IndicatorHealthTable } from "@/components/stages/validation/indicator-health-table";
import { IssueTable } from "@/components/stages/validation/issue-table";
import { ValidationSummaryBanner } from "@/components/stages/validation/validation-summary-banner";
import { HardGateAlert } from "@/components/ui/custom/hard-gate-alert";
import type { Stage3Data } from "@causal-ssm/api-types";

export default function Stage3Content({ data }: { data: Stage3Data }) {
  const report = data.validation_report;

  return (
    <div className="space-y-4">
      <ValidationSummaryBanner report={report} />
      {!report.is_valid && (
        <HardGateAlert
          title="Validation Failed — Pipeline Halted"
          explanation="The extracted data failed validation checks. No usable data remains after filtering."
          suggestion="Check data source quality, add more indicators, or broaden measurement criteria."
        />
      )}
      {report.issues.length > 0 && <IssueTable issues={report.issues} />}
      {report.per_indicator_health.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Indicator Health</h3>
          <IndicatorHealthTable rows={report.per_indicator_health} />
        </div>
      )}
    </div>
  );
}
