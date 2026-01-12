# Scope

This framework models dynamics of time-varying constructs with optional time-invariant covariates and random effects. Time-varying constructs may be **latent** provided they have observed indicators (reflective measurement model). The framework does not support latent state-space models where latent states have no indicators (see Exclusions).

---

## Supported Combinations

| Type | Role | Observability | Temporal | Example | Use |
|------|------|---------------|----------|---------|-----|
| 1 | Exogenous | Observed | Time-varying | Weather, day of week, market index | External time-varying inputs |
| 2 | Exogenous | Observed | Time-invariant | Age, gender, treatment arm | Between-person covariates |
| 4 | Exogenous | Latent | Time-invariant | Person-specific intercept | Random effects for heterogeneity |
| 5 | Endogenous | Observed | Time-varying | Daily mood, sleep quality | Core dynamic system |
| 7* | Endogenous | Latent | Time-varying | Stress (with indicators: HRV, cortisol) | Latent constructs with reflective indicators |

*Type 7 requires observed indicators via a reflective measurement model. See dsem_overview.md for details.

---

## Justifications for Supported Types

### Type 1: Exogenous, Observed, Time-varying

External inputs affecting but not affected by the system. No autoregressive structure needed—we condition on observed values. Autocorrelation in exogenous variables exists but is irrelevant since we are not modeling their causes.

### Type 2: Exogenous, Observed, Time-invariant

Between-person covariates. Can predict level-2 intercepts (e.g., "older participants have lower average sleep quality") or moderate within-person dynamics (e.g., "the stress→sleep coefficient is stronger for high-neuroticism individuals").

### Type 4: Exogenous, Latent, Time-invariant

Random effects capturing stable between-person heterogeneity. Identified by repeated observations within person—no indicators required. Partitions variance into between-person and within-person components. Not a causal node; a variance decomposition device that prevents conflation of between and within effects.

### Type 5: Endogenous, Observed, Time-varying

The core use case. Variables with dynamic structure: autoregressive inertia, cross-lagged effects, contemporaneous correlations. Directly observed, well-identified.

---

## Exclusions and Justifications

The excluded latent types (Types 3, 8, and state-space Type 7) require strong structural assumptions for identification. Type 7 with reflective indicators IS supported—see above.

### Type 3: Exogenous, Latent, Time-varying

An unobserved external shock varying over time. Excluded because identification requires either indicators (making it a factor) or strong structural assumptions. If such a variable is theorized, model it via observed proxy with acknowledged measurement error.

### Type 6: Endogenous, Observed, Time-invariant

A single-occasion outcome. Not a dynamic modeling problem—use standard SEM. Mixing paradigms adds complexity without benefit.

### Type 7 (State-Space): Endogenous, Latent, Time-varying WITHOUT Indicators

A latent state with its own dynamics but NO observed indicators (pure state-space / Kalman filter territory). Excluded because proper specification requires:

- Prior on initial state distribution
- Prior on process variance (latent state evolution noise)
- Prior on measurement variance (observation noise)

These parameters interact non-trivially. Process and measurement variance are notoriously difficult to disentangle without multiple indicators or strong domain-informed priors. An automated framework cannot reliably specify these.

**Note:** Type 7 WITH reflective indicators IS supported—see "Latent Time-Varying Constructs" section in dsem_overview.md. The distinction is:
- **Supported:** Latent "stress" → observed indicators [HRV, cortisol, self-report]. Identified via factor model.
- **Excluded:** Latent "true_mood" with no indicators, identified only through temporal dynamics. Requires state-space machinery.

### Type 8: Endogenous, Latent, Time-invariant

A person-level latent outcome caused by system dynamics. Double identification problem: latent and single observation per person. Not identified without indicators.
