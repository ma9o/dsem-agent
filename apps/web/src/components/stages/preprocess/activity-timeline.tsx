import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ActivityTimelineProps {
  lines: string[];
}

export function ActivityTimeline({ lines }: ActivityTimelineProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Activity Timeline</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-64 overflow-y-auto rounded-md border bg-muted/30 p-3">
          {lines.map((line, i) => (
            <div
              key={`line-${
                // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                i
              }`}
              className="border-b border-muted py-1 text-xs font-mono text-muted-foreground last:border-0"
            >
              {line}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
