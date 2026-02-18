import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Indicator } from "@causal-ssm/api-types";

function groupByConstruct(indicators: Indicator[]): Map<string, Indicator[]> {
  const grouped = new Map<string, Indicator[]>();
  for (const ind of indicators) {
    const list = grouped.get(ind.construct_name) ?? [];
    list.push(ind);
    grouped.set(ind.construct_name, list);
  }
  return grouped;
}

export function IndicatorTable({ indicators }: { indicators: Indicator[] }) {
  const grouped = groupByConstruct(indicators);

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Dtype</TableHead>
            <TableHead>Aggregation</TableHead>
            <TableHead>How to Measure</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {[...grouped.entries()].map(([construct, inds]) => (
            <>
              <TableRow key={`header-${construct}`} className="bg-muted/50 hover:bg-muted/50">
                <TableCell colSpan={4} className="py-2">
                  <span className="text-sm font-semibold">{construct}</span>
                  <span className="ml-2 text-xs text-muted-foreground">
                    {inds.length} indicator{inds.length !== 1 && "s"}
                  </span>
                </TableCell>
              </TableRow>
              {inds.map((ind) => (
                <TableRow key={ind.name}>
                  <TableCell className="pl-6 font-medium">{ind.name}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{ind.measurement_dtype}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary">{ind.aggregation}</Badge>
                  </TableCell>
                  <TableCell className="max-w-xs text-sm text-muted-foreground">
                    {ind.how_to_measure}
                  </TableCell>
                </TableRow>
              ))}
            </>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
