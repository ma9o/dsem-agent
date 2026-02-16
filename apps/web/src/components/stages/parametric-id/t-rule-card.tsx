"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import type { TRuleResult } from "@causal-ssm/api-types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface TRuleCardProps {
  tRule: TRuleResult;
}

export function TRuleCard({ tRule }: TRuleCardProps) {
  const chartData = [
    { name: "Free Params", value: tRule.n_free_params, fill: "hsl(var(--primary))" },
    { name: "Moments", value: tRule.n_moments, fill: "hsl(var(--muted-foreground))" },
  ];

  const paramCountEntries = Object.entries(tRule.param_counts);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="inline-flex items-center gap-1.5 text-base">
            T-Rule Check
            <StatTooltip explanation="The t-rule checks if there are enough moment conditions (data constraints) to uniquely identify all free parameters. Requires moments >= free params." />
          </CardTitle>
          <Badge variant={tRule.satisfies ? "success" : "destructive"}>
            {tRule.satisfies ? "Pass" : "Fail"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Bar chart comparing free params vs moments */}
        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
              <RechartsTooltip />
              <Bar dataKey="value" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center gap-4 text-sm">
          <span>
            <span className="font-medium">Free params:</span>{" "}
            <span className="text-muted-foreground">{tRule.n_free_params}</span>
          </span>
          <span>
            <span className="font-medium">Moments:</span>{" "}
            <span className="text-muted-foreground">{tRule.n_moments}</span>
          </span>
        </div>

        {/* Param counts breakdown */}
        {paramCountEntries.length > 0 && (
          <div>
            <p className="mb-2 text-sm font-medium text-muted-foreground">Parameter counts</p>
            <div className="flex flex-wrap gap-2">
              {paramCountEntries.map(([key, count]) => (
                <Badge key={key} variant="secondary">
                  {key}: {count}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
