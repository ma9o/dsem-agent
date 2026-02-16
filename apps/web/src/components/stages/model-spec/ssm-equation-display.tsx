import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";

interface SsmEquationDisplayProps {
  equations: string[];
}

export function SSMEquationDisplay({ equations }: SsmEquationDisplayProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="inline-flex items-center gap-1.5 text-base">
          SSM Equations
          <StatTooltip explanation="State-space model equations describing the continuous-time dynamics. η(t) is the latent state vector, A encodes autoregressive and causal effects, and y(t) = Λη(t) + ε(t) maps latents to observed indicators." />
        </CardTitle>
      </CardHeader>
      <CardContent>
        <pre className="overflow-x-auto rounded-md border bg-muted/30 p-4 text-sm font-mono leading-relaxed">
          {equations.map((eq, i) => (
            <div
              key={`eq-${
                // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                i
              }`}
            >
              {eq}
            </div>
          ))}
        </pre>
      </CardContent>
    </Card>
  );
}
