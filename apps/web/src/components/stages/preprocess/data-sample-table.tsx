import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface DataSampleTableProps {
  sample: Array<Record<string, string | null>>;
}

export function DataSampleTable({ sample }: DataSampleTableProps) {
  if (sample.length === 0) return null;

  // Discover columns from keys present in the data, preserving insertion order.
  // Drop columns that are null/empty in every row.
  const allKeys = Object.keys(sample[0]);
  const columns = allKeys.filter((key) => sample.some((row) => row[key] != null));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Data Sample</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-64 overflow-y-auto rounded-md border">
          <Table>
            <TableHeader className="sticky top-0 bg-background">
              <TableRow>
                {columns.map((col) => (
                  <TableHead key={col} className="text-xs capitalize">
                    {col.replace(/_/g, " ")}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {sample.map((row, i) => (
                <TableRow
                  key={`sample-${
                    // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                    i
                  }`}
                >
                  {columns.map((col) => (
                    <TableCell
                      key={col}
                      className="py-2 text-xs text-muted-foreground"
                    >
                      {row[col] ?? ""}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
