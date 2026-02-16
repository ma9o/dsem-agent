import { AggregationSummary } from "@/components/stages/validation/aggregation-summary";
import { IndicatorHealthCard } from "@/components/stages/validation/indicator-health-card";
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
          <div className="grid gap-3 sm:grid-cols-2">
            {report.per_indicator_health.map((h) => (
              <IndicatorHealthCard key={h.indicator} health={h} />
            ))}
          </div>
        </div>
      )}
      {report.aggregation_summary.length > 0 && (
        <AggregationSummary summary={report.aggregation_summary} />
      )}
    </div>
  );
}
