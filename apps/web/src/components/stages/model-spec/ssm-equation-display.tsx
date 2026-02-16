import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface SsmEquationDisplayProps {
  equations: string[];
}

export function SSMEquationDisplay({ equations }: SsmEquationDisplayProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">SSM Equations</CardTitle>
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
