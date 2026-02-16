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

export function IndicatorTable({ indicators }: { indicators: Indicator[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Construct</TableHead>
          <TableHead>Dtype</TableHead>
          <TableHead>Granularity</TableHead>
          <TableHead>Aggregation</TableHead>
          <TableHead>How to Measure</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {indicators.map((ind) => (
          <TableRow key={ind.name}>
            <TableCell className="font-medium">{ind.name}</TableCell>
            <TableCell>{ind.construct_name}</TableCell>
            <TableCell>
              <Badge variant="outline">{ind.measurement_dtype}</Badge>
            </TableCell>
            <TableCell>
              <Badge variant="secondary">{ind.measurement_granularity}</Badge>
            </TableCell>
            <TableCell>
              <Badge variant="secondary">{ind.aggregation}</Badge>
            </TableCell>
            <TableCell className="max-w-xs text-sm text-muted-foreground">
              {ind.how_to_measure}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
