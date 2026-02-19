/** Evaluate PDF for chart rendering — visualization only, not analytical logic. */

type DistParams = Record<string, number>;

function normalPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

function halfNormalPdf(x: number, sigma: number): number {
  if (x < 0) return 0;
  return (2 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * (x / sigma) ** 2);
}

/** Stirling approximation for ln(Gamma(a)). */
function lnGamma(a: number): number {
  return 0.5 * Math.log((2 * Math.PI) / a) + a * (Math.log(a + 1 / (12 * a - 1 / (10 * a))) - 1);
}

function gammaPdf(x: number, alpha: number, beta: number): number {
  if (x <= 0) return 0;
  const logPdf = alpha * Math.log(beta) + (alpha - 1) * Math.log(x) - beta * x - lnGamma(alpha);
  return Math.exp(logPdf);
}

function betaPdf(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1) return 0;
  const logB = lnGamma(alpha) + lnGamma(beta) - lnGamma(alpha + beta);
  return Math.exp((alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logB);
}

export function evaluatePdf(
  distribution: string,
  params: DistParams,
  nPoints = 200,
): Array<{ x: number; y: number }> {
  const dist = distribution.toLowerCase();
  let xMin = -4;
  let xMax = 4;

  // Set range based on distribution
  if (dist === "normal" || dist === "gaussian") {
    const mu = params.mu ?? params.loc ?? 0;
    const sigma = params.sigma ?? params.scale ?? 1;
    xMin = mu - 4 * sigma;
    xMax = mu + 4 * sigma;
  } else if (dist === "halfnormal" || dist === "half_normal") {
    xMin = 0;
    xMax = (params.sigma ?? params.scale ?? 1) * 4;
  } else if (dist === "gamma") {
    xMin = 0;
    xMax = ((params.alpha ?? params.concentration ?? 2) / (params.beta ?? params.rate ?? 1)) * 3;
  } else if (dist === "beta") {
    xMin = 0.001;
    xMax = 0.999;
  } else if (dist === "uniform") {
    xMin = params.low ?? 0;
    xMax = params.high ?? 1;
  }

  const step = (xMax - xMin) / nPoints;
  const points: Array<{ x: number; y: number }> = [];

  for (let i = 0; i <= nPoints; i++) {
    const x = xMin + i * step;
    let y = 0;

    if (dist === "normal" || dist === "gaussian") {
      y = normalPdf(x, params.mu ?? params.loc ?? 0, params.sigma ?? params.scale ?? 1);
    } else if (dist === "halfnormal" || dist === "half_normal") {
      y = halfNormalPdf(x, params.sigma ?? params.scale ?? 1);
    } else if (dist === "gamma") {
      y = gammaPdf(x, params.alpha ?? params.concentration ?? 2, params.beta ?? params.rate ?? 1);
    } else if (dist === "beta") {
      y = betaPdf(x, params.alpha ?? params.a ?? 2, params.beta ?? params.b ?? 2);
    } else if (dist === "uniform") {
      y =
        x >= (params.low ?? 0) && x <= (params.high ?? 1)
          ? 1 / ((params.high ?? 1) - (params.low ?? 0))
          : 0;
    }

    points.push({ x: Math.round(x * 1000) / 1000, y });
  }

  return points;
}
