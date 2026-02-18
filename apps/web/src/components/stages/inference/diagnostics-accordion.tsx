"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatNumber } from "@/lib/utils/format";
import type { PPCResult, PowerScalingResult } from "@causal-ssm/api-types";
import { Check, X } from "lucide-react";

interface DiagnosticsAccordionProps {
  powerScaling: PowerScalingResult[];
  ppc: PPCResult;
}

const diagnosisBadgeVariant: Record<string, "success" | "warning" | "destructive"> = {
  well_identified: "success",
  prior_dominated: "warning",
  prior_data_conflict: "destructive",
};

const diagnosisLabel: Record<string, string> = {
  well_identified: "Well Identified",
  prior_dominated: "Prior Dominated",
  prior_data_conflict: "Prior-Data Conflict",
};

export function DiagnosticsAccordion({ powerScaling, ppc }: DiagnosticsAccordionProps) {
  return (
    <Accordion>
      {/* Power Scaling Section */}
      <AccordionItem value="power-scaling">
        <AccordionTrigger value="power-scaling" className="text-sm">
          <span className="inline-flex items-center gap-1.5">
            Power Scaling Diagnostics
            <StatTooltip explanation="Tests whether posteriors are driven by data (good) or priors (concerning). Scales the likelihood and prior to detect sensitivity." />
          </span>
          <Badge
            variant={
              powerScaling.every((p) => p.diagnosis === "well_identified") ? "success" : "warning"
            }
            className="ml-2"
          >
            {powerScaling.filter((p) => p.diagnosis === "well_identified").length}/
            {powerScaling.length} OK
          </Badge>
        </AccordionTrigger>
        <AccordionContent value="power-scaling">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Parameter</TableHead>
                <TableHead>
                  <span className="inline-flex items-center gap-1">
                    Diagnosis
                    <StatTooltip explanation="'Well Identified' = data-driven. 'Prior Dominated' = prior choice matters too much. 'Prior-Data Conflict' = prior and data disagree." />
                  </span>
                </TableHead>
                <TableHead className="text-right">
                  <span className="inline-flex items-center gap-1">
                    Prior Sens.
                    <StatTooltip explanation="How much the posterior changes when the prior is scaled. Low values (<0.05) are good." />
                  </span>
                </TableHead>
                <TableHead className="text-right">
                  <span className="inline-flex items-center gap-1">
                    Lik. Sens.
                    <StatTooltip explanation="How much the posterior changes when the likelihood is scaled. High values indicate the data is informative." />
                  </span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {powerScaling.map((ps) => (
                <TableRow key={ps.parameter}>
                  <TableCell className="font-medium font-mono text-sm">{ps.parameter}</TableCell>
                  <TableCell>
                    <Badge variant={diagnosisBadgeVariant[ps.diagnosis] ?? "secondary"}>
                      {diagnosisLabel[ps.diagnosis] ?? ps.diagnosis}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm">
                    {formatNumber(ps.prior_sensitivity)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm">
                    {formatNumber(ps.likelihood_sensitivity)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </AccordionContent>
      </AccordionItem>

      {/* PPC Section */}
      <AccordionItem value="ppc">
        <AccordionTrigger value="ppc" className="text-sm">
          <span className="inline-flex items-center gap-1.5">
            Posterior Predictive Checks
            <StatTooltip explanation="Simulates data from the fitted model and compares to observed data. Failures suggest model misspecification." />
          </span>
          <Badge variant={ppc.overall_passed ? "success" : "destructive"} className="ml-2">
            {ppc.overall_passed ? "Passed" : "Failed"}
          </Badge>
        </AccordionTrigger>
        <AccordionContent value="ppc">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Variable</TableHead>
                <TableHead>Check</TableHead>
                <TableHead>Message</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ppc.per_variable_warnings.map((w, i) => (
                <TableRow
                  key={`ppc-${
                    // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                    i
                  }`}
                >
                  <TableCell className="font-medium">{w.variable}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{w.check_type}</Badge>
                  </TableCell>
                  <TableCell className="max-w-sm text-sm text-muted-foreground">
                    {w.message}
                  </TableCell>
                  <TableCell>
                    {w.passed ? (
                      <Check className="h-4 w-4 text-success" />
                    ) : (
                      <X className="h-4 w-4 text-destructive" />
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
