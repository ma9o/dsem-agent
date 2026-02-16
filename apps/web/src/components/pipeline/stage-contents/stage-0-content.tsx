import { ActivityTimeline } from "@/components/stages/preprocess/activity-timeline";
import { DataSummaryStats } from "@/components/stages/preprocess/data-summary-stats";
import type { Stage0Data } from "@causal-ssm/api-types";

export default function Stage0Content({ data }: { data: Stage0Data }) {
  return (
    <div className="space-y-4">
      <DataSummaryStats
        nRecords={data.n_records}
        dateRange={data.date_range}
        activityTypeCounts={data.activity_type_counts}
      />
      {data.lines.length > 0 && <ActivityTimeline lines={data.lines.slice(0, 100)} />}
    </div>
  );
}
