"use client";

import { isMockMode } from "@/lib/api/mock-provider";
import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-sm">
      <div className="mx-auto flex h-12 max-w-4xl items-center justify-between px-4">
        <Link
          href="/"
          className="text-sm font-semibold tracking-tight hover:opacity-80 transition-opacity"
        >
          Causal Inference Pipeline
        </Link>
        {isMockMode() && (
          <span className="rounded-full border border-warning/50 bg-warning-soft px-2 py-0.5 text-[10px] font-medium text-warning-soft-foreground">
            Mock Mode
          </span>
        )}
      </div>
    </header>
  );
}
