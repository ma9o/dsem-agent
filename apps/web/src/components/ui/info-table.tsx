"use client";

import { Fragment, type ReactNode, useMemo, useState } from "react";
import {
  type ColumnDef,
  type SortingState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ChevronDown, ChevronUp, ChevronsUpDown } from "lucide-react";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils/cn";
import { useTableKeyboardNav } from "./use-table-keyboard-nav";

// ---------- Column meta typing ----------
declare module "@tanstack/react-table" {
  // biome-ignore lint/correctness/noUnusedVariables: augmenting module
  interface ColumnMeta<TData, TValue> {
    align?: "left" | "center" | "right";
    mono?: boolean;
  }
}

// ---------- HeaderWithTooltip helper ----------
export function HeaderWithTooltip({
  label,
  tooltip,
}: {
  label: string;
  tooltip: string;
}) {
  return (
    <span className="inline-flex items-center gap-1">
      {label}
      <StatTooltip explanation={tooltip} />
    </span>
  );
}

// ---------- InfoTable ----------
interface InfoTableProps<TData> {
  columns: ColumnDef<TData, unknown>[];
  data: TData[];
  sorting?: boolean;
  maxHeight?: string;
  groupBy?: (row: TData) => string;
  renderGroupHeader?: (groupKey: string, rows: TData[]) => ReactNode;
}

export function InfoTable<TData>({
  columns,
  data,
  sorting: enableSorting = false,
  maxHeight,
  groupBy,
  renderGroupHeader,
}: InfoTableProps<TData>) {
  const [sortingState, setSortingState] = useState<SortingState>([]);

  const table = useReactTable({
    data,
    columns,
    state: enableSorting ? { sorting: sortingState } : undefined,
    onSortingChange: enableSorting ? setSortingState : undefined,
    getCoreRowModel: getCoreRowModel(),
    ...(enableSorting && { getSortedRowModel: getSortedRowModel() }),
  });

  const rows = table.getRowModel().rows;
  const { focusedRowIndex, containerProps } = useTableKeyboardNav(rows.length);

  // Grouped rendering
  const groups = useMemo(() => {
    if (!groupBy) return null;
    const map = new Map<string, typeof rows>();
    for (const row of rows) {
      const key = groupBy(row.original);
      const list = map.get(key) ?? [];
      list.push(row);
      map.set(key, list);
    }
    return map;
  }, [groupBy, rows]);

  const wrapperClass = maxHeight
    ? `${maxHeight} overflow-y-auto rounded-md border`
    : undefined;

  return (
    <div className={wrapperClass} {...containerProps}>
      <Table>
        <TableHeader className={maxHeight ? "sticky top-0 bg-background z-10" : undefined}>
          {table.getHeaderGroups().map((hg) => (
            <TableRow key={hg.id}>
              {hg.headers.map((header) => {
                const meta = header.column.columnDef.meta;
                const alignClass =
                  meta?.align === "right"
                    ? "text-right"
                    : meta?.align === "center"
                      ? "text-center"
                      : "text-left";

                return (
                  <TableHead
                    key={header.id}
                    className={cn(alignClass)}
                    onClick={
                      enableSorting && header.column.getCanSort()
                        ? header.column.getToggleSortingHandler()
                        : undefined
                    }
                    style={
                      enableSorting && header.column.getCanSort()
                        ? { cursor: "pointer", userSelect: "none" }
                        : undefined
                    }
                  >
                    <span className="inline-flex items-center gap-1">
                      {header.isPlaceholder
                        ? null
                        : flexRender(header.column.columnDef.header, header.getContext())}
                      {enableSorting &&
                        header.column.getCanSort() &&
                        (header.column.getIsSorted() === "asc" ? (
                          <ChevronUp className="h-3.5 w-3.5" />
                        ) : header.column.getIsSorted() === "desc" ? (
                          <ChevronDown className="h-3.5 w-3.5" />
                        ) : (
                          <ChevronsUpDown className="h-3 w-3 text-muted-foreground/50" />
                        ))}
                    </span>
                  </TableHead>
                );
              })}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {groups
            ? [...groups.entries()].map(([groupKey, groupRows]) => (
                <Fragment key={groupKey}>
                  {renderGroupHeader && (
                    <TableRow className="bg-muted/50 hover:bg-muted/50">
                      <TableCell colSpan={columns.length} className="py-2">
                        {renderGroupHeader(groupKey, groupRows.map((r) => r.original))}
                      </TableCell>
                    </TableRow>
                  )}
                  {groupRows.map((row) => renderRow(row, columns, focusedRowIndex))}
                </Fragment>
              ))
            : rows.map((row) => renderRow(row, columns, focusedRowIndex))}
        </TableBody>
      </Table>
    </div>
  );
}

function renderRow<TData>(
  row: ReturnType<ReturnType<typeof useReactTable<TData>>["getRowModel"]>["rows"][number],
  columns: ColumnDef<TData, unknown>[],
  focusedRowIndex: number | null,
) {
  return (
    <TableRow
      key={row.id}
      className={cn(focusedRowIndex === row.index && "ring-2 ring-ring ring-inset")}
    >
      {row.getVisibleCells().map((cell) => {
        const meta = cell.column.columnDef.meta;
        const alignClass =
          meta?.align === "right"
            ? "text-right"
            : meta?.align === "center"
              ? "text-center"
              : undefined;

        return (
          <TableCell
            key={cell.id}
            className={cn(
              alignClass,
              meta?.mono && "font-mono",
            )}
          >
            {flexRender(cell.column.columnDef.cell, cell.getContext())}
          </TableCell>
        );
      })}
    </TableRow>
  );
}
