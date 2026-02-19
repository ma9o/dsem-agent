/**
 * Build a histogram from numeric values.
 *
 * Returns bin center, count, and optional edges for each bin.
 */
export function buildHistogram(
  values: number[],
  nBins = 20,
): Array<{ binCenter: number; count: number; binStart: number; binEnd: number }> {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / nBins;

  const bins = Array.from({ length: nBins }, (_, i) => ({
    binCenter: Math.round((min + (i + 0.5) * binWidth) * 100) / 100,
    count: 0,
    binStart: min + i * binWidth,
    binEnd: min + (i + 1) * binWidth,
  }));

  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), nBins - 1);
    bins[idx].count++;
  }
  return bins;
}
