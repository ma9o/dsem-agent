import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatDate } from "@/lib/utils/format";

interface DataSampleTableProps {
  sample: Array<{ timestamp: string; content: string }>;
}

export function DataSampleTable({ sample }: DataSampleTableProps) {
  if (sample.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Data Sample</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[160px]">Timestamp</TableHead>
              <TableHead>Content</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sample.map((entry, i) => (
              <TableRow
                key={`sample-${
                  // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                  i
                }`}
              >
                <TableCell className="whitespace-nowrap font-mono text-xs text-muted-foreground">
                  {formatDate(entry.timestamp)}
                </TableCell>
                <TableCell className="text-sm">{entry.content}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
