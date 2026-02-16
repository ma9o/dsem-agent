import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber } from "@/lib/utils/format";
import type { IndicatorHealth } from "@causal-ssm/api-types";
import { Activity } from "lucide-react";

function StatRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{children}</span>
    </div>
  );
}

export function IndicatorHealthCard({ health }: { health: IndicatorHealth }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Activity className="h-4 w-4 text-muted-foreground" />
          {health.indicator}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <StatRow label="Observations">{health.n_obs.toLocaleString()}</StatRow>
        <StatRow label="Variance">
          {health.variance !== null ? formatNumber(health.variance) : "--"}
        </StatRow>
        <StatRow label="Time Coverage">
          {health.time_coverage_ratio !== null ? formatNumber(health.time_coverage_ratio) : "--"}
        </StatRow>
        <StatRow label="Max Gap Ratio">
          {health.max_gap_ratio !== null ? formatNumber(health.max_gap_ratio) : "--"}
        </StatRow>
        <StatRow label="Dtype Violations">
          {health.dtype_violations > 0 ? (
            <Badge variant="destructive">{health.dtype_violations}</Badge>
          ) : (
            <span>0</span>
          )}
        </StatRow>
        <StatRow label="Duplicate %">{formatNumber(health.duplicate_pct)}</StatRow>
        <StatRow label="Arithmetic Sequence">
          {health.arithmetic_sequence_detected ? (
            <Badge variant="warning">detected</Badge>
          ) : (
            <span className="text-muted-foreground">none</span>
          )}
        </StatRow>
      </CardContent>
    </Card>
  );
}
