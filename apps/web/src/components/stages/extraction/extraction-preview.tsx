import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Extraction } from "@causal-ssm/api-types";
import { Database } from "lucide-react";

export function ExtractionPreview({
  extractions,
  totalExtractions,
  perIndicatorCounts,
}: {
  extractions: Extraction[];
  totalExtractions: number;
  perIndicatorCounts: Record<string, number>;
}) {
  const indicatorEntries = Object.entries(perIndicatorCounts).sort(([, a], [, b]) => b - a);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Total Extractions:</span>
          <Badge variant="default">{totalExtractions.toLocaleString()}</Badge>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Per-Indicator Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {indicatorEntries.map(([indicator, count]) => (
              <Badge key={indicator} variant="outline" className="gap-1">
                {indicator}
                <span className="ml-1 font-mono text-xs text-muted-foreground">{count}</span>
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      {extractions.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Sample Extractions ({extractions.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Indicator</TableHead>
                  <TableHead>Value</TableHead>
                  <TableHead>Timestamp</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {extractions.map((ext, i) => (
                  <TableRow key={`${ext.indicator}-${ext.timestamp}-${i}`}>
                    <TableCell className="font-medium">{ext.indicator}</TableCell>
                    <TableCell className="font-mono text-sm">
                      {ext.value === null ? (
                        <span className="text-muted-foreground">null</span>
                      ) : (
                        String(ext.value)
                      )}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {ext.timestamp ?? "--"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
