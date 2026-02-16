import { ExtractionPreview } from "@/components/stages/extraction/extraction-preview";
import { WorkerProgressGrid } from "@/components/stages/extraction/worker-progress-grid";
import type { Stage2Data } from "@causal-ssm/api-types";

export default function Stage2Content({ data }: { data: Stage2Data }) {
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
