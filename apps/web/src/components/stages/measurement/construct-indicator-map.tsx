import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Construct, Indicator } from "@causal-ssm/api-types";
import { Layers } from "lucide-react";

export function ConstructIndicatorMap({
  constructs,
  indicators,
}: {
  constructs: Construct[];
  indicators: Indicator[];
}) {
  const indicatorsByConstruct = new Map<string, Indicator[]>();
  for (const ind of indicators) {
    const existing = indicatorsByConstruct.get(ind.construct_name) ?? [];
    existing.push(ind);
    indicatorsByConstruct.set(ind.construct_name, existing);
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {constructs.map((construct) => {
        const mapped = indicatorsByConstruct.get(construct.name) ?? [];
        return (
          <Card key={construct.name}>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Layers className="h-4 w-4 text-muted-foreground" />
                {construct.name}
              </CardTitle>
              <div className="flex flex-wrap gap-1">
                <Badge variant="outline">{construct.role}</Badge>
                {construct.is_outcome && <Badge variant="default">outcome</Badge>}
                <Badge variant="secondary">{construct.temporal_status}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              {mapped.length > 0 ? (
                <ul className="space-y-1.5">
                  {mapped.map((ind) => (
                    <li key={ind.name} className="flex items-center justify-between text-sm">
                      <span>{ind.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {ind.measurement_dtype}
                      </Badge>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">No indicators mapped</p>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
