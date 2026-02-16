"use client";

import { isMockMode } from "@/lib/api/mock-provider";
import Link from "next/link";
import { ThemeToggle } from "./theme-toggle";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-sm">
      <div className="mx-auto flex h-12 max-w-4xl items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="text-sm font-semibold tracking-tight hover:opacity-80 transition-opacity"
          >
            Causal Inference Pipeline
          </Link>
          {isMockMode() && (
            <span className="rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-400">
              Mock Mode
            </span>
          )}
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
}
