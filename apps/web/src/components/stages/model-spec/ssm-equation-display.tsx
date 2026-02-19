import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import type { LikelihoodSpec, ParameterSpec, PriorProposal } from "@causal-ssm/api-types";
import katex from "katex";
import { ArrowDown, BookOpen } from "lucide-react";

const FUNCTIONAL_SPEC_URL =
  "https://github.com/ma9o/causal-ssm-agent/blob/master/apps/data-pipeline/docs/modeling/functional_spec.md#15-parameter-roles-and-constraints";

interface SsmEquationDisplayProps {
  likelihoods: LikelihoodSpec[];
  parameters: ParameterSpec[];
  priors: PriorProposal[];
}

/** Render a LaTeX string to an HTML string via KaTeX. */
function tex(latex: string, displayMode = true): string {
  return katex.renderToString(latex, { displayMode, throwOnError: false, strict: false });
}

/** Convert snake_case to spaced text for use inside LaTeX \text{}. */
function textify(name: string): string {
  return name.replace(/_/g, " ");
}

/** Build the g⁻¹(·) wrapper for a link function around a linear predictor. */
function linkInverse(link: string, predictor: string): string {
  switch (link) {
    case "identity":
      return predictor;
    case "log":
      return `\\exp(${predictor})`;
    case "logit":
      return `\\sigma(${predictor})`;
    case "probit":
      return `\\Phi(${predictor})`;
    case "cumulative_logit":
      return `\\text{cumlogit}^{-1}(${predictor})`;
    case "softmax":
      return `\\text{softmax}(${predictor})`;
    default:
      return `g^{-1}(${predictor})`;
  }
}

/** Map distribution family enum to LaTeX name. */
function distName(dist: string): string {
  const map: Record<string, string> = {
    gaussian: "\\mathcal{N}",
    student_t: "t_{\\nu}",
    poisson: "\\text{Poisson}",
    gamma: "\\text{Gamma}",
    bernoulli: "\\text{Bernoulli}",
    negative_binomial: "\\text{NegBin}",
    beta: "\\text{Beta}",
    ordered_logistic: "\\text{OrdLogistic}",
    categorical: "\\text{Categorical}",
  };
  return map[dist] ?? `\\text{${dist}}`;
}

/** Build a single observation-model line with per-variable μ subscript. */
function likelihoodLine(lik: LikelihoodSpec): string {
  const v = `\\text{${textify(lik.variable)}}`;
  const mu = linkInverse(lik.link, `\\mu_{${v}}`);
  const d = distName(lik.distribution);

  if (lik.distribution === "gaussian" || lik.distribution === "student_t") {
    return `y_{${v}}(t) &\\sim ${d}(${mu},\\; \\sigma_{${v}}^{2})`;
  }
  if (lik.distribution === "beta") {
    return `y_{${v}}(t) &\\sim ${d}(${mu}\\,\\phi,\\; (1 - ${mu})\\,\\phi)`;
  }
  if (lik.distribution === "negative_binomial") {
    return `y_{${v}}(t) &\\sim ${d}(r,\\; ${mu})`;
  }
  // single-param: Poisson, Bernoulli, etc.
  return `y_{${v}}(t) &\\sim ${d}(${mu})`;
}

/** Parse a parameter name into Greek letter + subscript. */
function paramSymbol(name: string): string {
  const greekMap: Record<string, string> = {
    beta: "\\beta",
    rho: "\\rho",
    sigma: "\\sigma",
    lambda: "\\lambda",
    alpha: "\\alpha",
    gamma: "\\gamma",
    phi: "\\phi",
    tau: "\\tau",
    mu: "\\mu",
    nu: "\\nu",
    kappa: "\\kappa",
    theta: "\\theta",
    omega: "\\omega",
  };

  const parts = name.split("_");
  const greek = greekMap[parts[0]];
  if (greek && parts.length > 1) {
    return `${greek}_{\\text{${parts.slice(1).join(" ")}}}`;
  }
  return `\\text{${textify(name)}}`;
}

/** Map a prior distribution name + params to LaTeX. */
function priorLine(prior: PriorProposal): string {
  const sym = paramSymbol(prior.parameter);
  const vals = Object.values(prior.params).map((v) => String(v));

  const dMap: Record<string, string> = {
    Normal: "\\mathcal{N}",
    HalfNormal: "\\text{HalfNormal}",
    HalfCauchy: "\\text{HalfCauchy}",
    Beta: "\\text{Beta}",
    Gamma: "\\text{Gamma}",
    InverseGamma: "\\text{InvGamma}",
    Uniform: "\\text{Uniform}",
    Exponential: "\\text{Exp}",
    LKJCholesky: "\\text{LKJ}",
    Cauchy: "\\text{Cauchy}",
    LogNormal: "\\text{LogNormal}",
  };

  const d = dMap[prior.distribution] ?? `\\text{${prior.distribution}}`;
  return `${sym} &\\sim ${d}(${vals.join(",\\; ")})`;
}

/** Extract latent state names from AR coefficient parameters. */
function stateNames(parameters: ParameterSpec[]): string[] {
  const ar = parameters.filter((p) => p.role === "ar_coefficient");
  if (ar.length > 0) {
    return ar.map((p) => p.name.split("_").slice(1).join("_"));
  }
  // fallback: residual_sd params
  return parameters
    .filter((p) => p.role === "residual_sd")
    .map((p) => p.name.split("_").slice(1).join("_"));
}

export function SSMEquationDisplay({ likelihoods, parameters, priors }: SsmEquationDisplayProps) {
  const states = stateNames(parameters);

  // --- State dynamics ---
  const transitionLatex = tex(
    String.raw`\begin{aligned}
\eta_i(t) &= \rho_i \, \eta_i(t\!-\!1) + \textstyle\sum_{j \in \mathrm{pa}(i)} \beta_{ji}\, \eta_j(t\!-\!1) + \varepsilon_i(t) \\
\varepsilon_i(t) &\sim \mathcal{N}(0,\, \sigma_i^2)
\end{aligned}`,
  );

  const stateVecLatex =
    states.length > 0
      ? tex(
          `\\boldsymbol{\\eta}(t) = \\begin{bmatrix} ${states.map((s) => `\\eta_{\\text{${textify(s)}}}(t)`).join(" \\\\ ")} \\end{bmatrix}`,
        )
      : null;

  // --- Observation model ---
  const predictorDef =
    likelihoods.length > 0
      ? tex(
          String.raw`\mu_v(t) = \boldsymbol{\lambda}_v^\top \boldsymbol{\eta}(t)`,
        )
      : null;

  const obsLatex =
    likelihoods.length > 0
      ? tex(`\\begin{aligned}\n${likelihoods.map(likelihoodLine).join(" \\\\\n")}\n\\end{aligned}`)
      : null;

  // --- Priors ---
  const priorsLatex =
    priors.length > 0
      ? tex(`\\begin{aligned}\n${priors.map(priorLine).join(" \\\\\n")}\n\\end{aligned}`)
      : null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">SSM Equations</CardTitle>
          <a
            href={FUNCTIONAL_SPEC_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <BookOpen className="h-3.5 w-3.5" />
            Parameter roles &amp; constraints
          </a>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* State dynamics */}
        <section>
          <div className="mb-2 flex items-center justify-between">
            <h4 className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              State Dynamics
              <StatTooltip explanation="Each latent state evolves as a discrete-time AR(1) process: it depends on its own previous value (persistence ρ), causal effects from parent states (β), and Gaussian noise. This is obtained by exact discretization of an underlying continuous-time SDE." />
            </h4>
            <Badge variant="outline">Linear-Gaussian Dynamics</Badge>
          </div>
          <div className="grid gap-3 md:grid-cols-[1fr,auto]">
            {/* Main equations */}
            <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
              <div dangerouslySetInnerHTML={{ __html: transitionLatex }} />
              {stateVecLatex && <div dangerouslySetInnerHTML={{ __html: stateVecLatex }} />}
            </div>
            {/* Discretization aside */}
            <div className="flex flex-col items-center gap-2 rounded-md border border-dashed bg-muted/15 px-4 py-3 text-center md:max-w-56">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Continuous-time
              </span>
              <div
                className="text-xs [&_.katex]:text-xs"
                dangerouslySetInnerHTML={{ __html: tex(
                  String.raw`\mathrm{d}\boldsymbol{\eta} = (\mathbf{A}\boldsymbol{\eta} + \mathbf{c})\,\mathrm{d}t + \mathbf{G}\,\mathrm{d}\mathbf{W}`,
                ) }}
              />
              <div className="flex flex-col items-center gap-0.5 text-muted-foreground">
                <ArrowDown className="h-4 w-4" />
                <span className="text-[10px] leading-tight">
                  exact discretization at&nbsp;&Delta;t
                </span>
              </div>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Discrete-time
              </span>
              <p className="text-[11px] leading-snug text-muted-foreground">
                The pipeline specifies dynamics in continuous time, then discretizes exactly to the
                observation interval to obtain the AR(1) transition used for estimation.
              </p>
            </div>
          </div>
        </section>

        {/* Observation model */}
        {obsLatex && (
          <section>
            <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Observation Model
              <StatTooltip explanation="Maps latent states to observed indicators. Each variable has a distribution family (e.g. Gaussian, Poisson) and a link function (e.g. identity, log, logit) that transforms the linear predictor λᵀη(t) to the distribution's natural parameter." />
            </h4>
            <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
              {predictorDef && <div dangerouslySetInnerHTML={{ __html: predictorDef }} />}
              <div dangerouslySetInnerHTML={{ __html: obsLatex }} />
            </div>
          </section>
        )}

        {/* Priors */}
        {priorsLatex && (
          <section>
            <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Priors
              <StatTooltip explanation="Informative or weakly informative prior distributions for each model parameter, elicited from domain literature. These constrain the posterior and encode existing knowledge about plausible effect sizes, persistence, and variance." />
            </h4>
            <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
              <div dangerouslySetInnerHTML={{ __html: priorsLatex }} />
            </div>
          </section>
        )}
      </CardContent>
    </Card>
  );
}
