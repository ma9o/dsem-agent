import { ExtractionPreview } from "@/components/stages/extraction/extraction-preview";
import { WorkerProgressGrid } from "@/components/stages/extraction/worker-progress-grid";
import type { Stage2Data } from "@causal-ssm/api-types";

export default function Stage2Content({ data }: { data: Stage2Data }) {
  if (data.workers.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No extraction workers were dispatched. Check if indicators were defined in the previous
        stage.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <WorkerProgressGrid workers={data.workers} />
      <ExtractionPreview
        extractions={data.combined_extractions_sample}
        totalExtractions={data.total_extractions}
        perIndicatorCounts={data.per_indicator_counts}
      />
    </div>
  );
}
