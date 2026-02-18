import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface DynamicTableProps {
  rows: Array<Record<string, string | null>>;
  maxHeight?: string;
}

export function DynamicTable({ rows, maxHeight = "max-h-52" }: DynamicTableProps) {
  if (rows.length === 0) return null;

  const allKeys = Object.keys(rows[0]);
  const columns = allKeys.filter((key) => rows.some((row) => row[key] != null));

  return (
    <div className={`${maxHeight} overflow-y-auto rounded-md border`}>
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
          {rows.map((row, i) => (
            <TableRow
              key={`row-${
                // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                i
              }`}
            >
              {columns.map((col) => (
                <TableCell
                  key={col}
                  className="py-1 text-xs text-muted-foreground"
                >
                  {row[col] ?? ""}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
