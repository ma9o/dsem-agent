import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import type { IdentifiabilityStatus } from "@causal-ssm/api-types";
import { XCircle } from "lucide-react";

export function IdentifiabilityPanel({
  identifiability,
}: {
  identifiability: IdentifiabilityStatus;
}) {
  const nonIdentifiable = Object.entries(identifiability.non_identifiable_treatments);

  if (nonIdentifiable.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="flex items-center gap-2 text-sm font-medium">
        <XCircle className="h-4 w-4 text-destructive" />
        Non-Identifiable Treatments
        <StatTooltip explanation="Treatments with unobserved confounders that prevent unique identification of causal effects from the data." />
      </h3>
      <div className="grid gap-3 sm:grid-cols-2">
        {nonIdentifiable.map(([name, status]) => (
          <Card key={name} className="border-destructive/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">{name}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex flex-wrap items-center gap-1">
                <span className="text-muted-foreground">Confounders:</span>
                {status.confounders.map((c) => (
                  <Badge key={c} variant="destructive">
                    {c}
                  </Badge>
                ))}
              </div>
              {status.notes && <p className="text-muted-foreground">{status.notes}</p>}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
