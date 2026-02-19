import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Lock } from "lucide-react";

export function HardGateAlert({
  title,
  explanation,
  suggestion,
  children,
}: {
  title: string;
  explanation: string;
  suggestion?: string;
  children?: React.ReactNode;
}) {
  return (
    <Alert variant="destructive" className="border-2">
      <Lock className="h-5 w-5" />
      <AlertTitle className="text-base font-semibold">{title}</AlertTitle>
      <AlertDescription className="mt-2 space-y-2">
        <p>{explanation}</p>
        {children}
        {suggestion && <p className="font-medium text-sm">Suggestion: {suggestion}</p>}
      </AlertDescription>
    </Alert>
  );
}
