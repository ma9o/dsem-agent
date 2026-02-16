import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Calendar, Database } from "lucide-react";

interface DataSummaryStatsProps {
  nRecords: number;
  dateRange: { start: string; end: string };
  activityTypeCounts: Record<string, number>;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function DataSummaryStats({
  nRecords,
  dateRange,
  activityTypeCounts,
}: DataSummaryStatsProps) {
  const sortedTypes = Object.entries(activityTypeCounts).sort(([, a], [, b]) => b - a);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Data Summary</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <Database className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium">{nRecords.toLocaleString()}</span>
            <span className="text-muted-foreground">records</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">
              {formatDate(dateRange.start)} &ndash; {formatDate(dateRange.end)}
            </span>
          </div>
        </div>

        <div>
          <p className="mb-2 text-sm font-medium text-muted-foreground">Activity types</p>
          <div className="flex flex-wrap gap-2">
            {sortedTypes.map(([type, count]) => (
              <Badge key={type} variant="secondary">
                {type}
                <span className="ml-1.5 text-muted-foreground">{count.toLocaleString()}</span>
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
