# Modeling Assumptions

This document enumerates the core modeling assumptions underlying the causal-agent framework. Each assumption constrains what can be modeled and has implications for interpretation.

---

## A1. Reflective Measurement Model

**Assumption:** All latent constructs use a reflective (effect-indicator) measurement model, not a formative (causal-indicator) model.

**Definition:** In a reflective model, the latent variable is the common cause of its indicators. Causality flows from construct to indicators:

```
Latent Construct → Indicator₁
                 → Indicator₂
                 → Indicator₃
```

In a formative model, indicators cause the construct (e.g., SES is "formed" by income, education, occupation). We do not support formative measurement.

**Implications:**
- Indicators of the same latent construct should be correlated (they share a common cause)
- Removing an indicator does not change the definition of the construct
- The latent construct exists independently of its specific operationalization
- No causal edges from indicators to their latent construct

**Justification:** Reflective models are standard in psychological and behavioral SEM. They align with classical test theory where observed scores reflect true scores plus error. Formative models require different identification constraints and are conceptually suited to composite indices (HDI, SES) rather than theoretical constructs.

**Reference:** Diamantopoulos, A., & Siguaw, J. A. (2006). Formative versus reflective indicators in organizational measure development. *British Journal of Management*.

---

## A2. Latent Time-Varying Constructs Require Indicators

**Assumption:** Latent time-varying constructs must have at least one observed indicator. Pure state-space models (latent dynamics with no indicators) are not supported.

**Definition:**
- **Supported:** Latent "stress" (time-varying) with indicators [HRV, cortisol, self-report]
- **Not supported:** Latent "true_mood" (time-varying) with no indicators, identified only via temporal dynamics

**Implications:**
- Every latent construct in the structural model must be operationalized into observables
- Identification comes from the measurement model (factor loadings), not from dynamics alone
- We avoid the process-vs-measurement variance disentanglement problem inherent in state-space models

**Justification:** State-space models require specifying priors on initial state distribution, process variance, and measurement variance. These interact non-trivially and are difficult to disentangle without strong domain knowledge. By requiring indicators, we use the well-understood factor-analytic identification strategy instead.

---

## A3. Markov Property for Temporal Dynamics

**Assumption:** All endogenous time-varying constructs follow first-order Markov dynamics. The state at t-1 is a sufficient statistic for all prior history.

**Definition:** For any endogenous variable Y:
```
P(Yₜ | Yₜ₋₁, Yₜ₋₂, ..., Y₁) = P(Yₜ | Yₜ₋₁)
```

**Implications:**
- AR(1) captures the relevant temporal dependence
- Higher-order lags (AR(2), AR(3), etc.) are not modeled
- Cross-lagged effects use lag-1 at the native granularity
- Residual autocorrelation suggests missing cross-lags or unmeasured confounders, not higher-order AR

**Justification:** The Markov property is a parsimony constraint with asymmetric costs. Including unnecessary AR(1) wastes one parameter (coefficient ≈ 0, harmless). Omitting necessary AR(1) biases standard errors and inflates cross-lag estimates (harmful). Default AR(1) is the conservative choice.

---

## A4. Acyclicity Within Time Slice

**Assumption:** Contemporaneous causal relationships (within the same time index) must form a directed acyclic graph (DAG).

**Definition:** If we consider only edges where lag = 0, the resulting graph must have no cycles.

**Implications:**
- Feedback loops must be modeled via lagged edges (across time)
- Contemporaneous relationships represent instantaneous causation or common response to unmodeled causes
- Standard DAG-based identification algorithms apply within each time slice

**Justification:** Cyclic contemporaneous relationships are not identified without additional constraints (instrumental variables, non-Gaussianity). Requiring acyclicity simplifies identification while allowing feedback dynamics through the temporal structure.

---

## A5. Random Effects as Exogenous Between-Person Heterogeneity

**Assumption:** Random effects (latent time-invariant variables) capture stable between-person differences. They are exogenous by construction—no modeled causes within the system.

**Implications:**
- Random effects partition variance into between-person and within-person components
- They prevent conflation of between and within effects (Simpson's paradox)
- They cannot be predicted by other variables in the model

**Justification:** In intensive longitudinal data, ignoring between-person heterogeneity biases within-person effect estimates. Random intercepts are the minimal adjustment for this confound.

---

## A6. Measurement Error Absorbed into Residuals (for Observed Variables)

**Assumption:** For variables treated as observed (not latent), we do not explicitly model measurement error. It is absorbed into residual variance.

**Definition:** For any observed variable X:
```
X_observed = X_true + ε  (where ε is absorbed into model residual)
```

**Implications:**
- Estimated coefficients may be attenuated (biased toward zero) if measurement error is substantial
- "Observed" means data exists, not perfect measurement
- Results should be interpreted as relationships between measured quantities

**Note:** This assumption applies to the final observed indicators. Latent constructs (when used) DO separate measurement error from construct variance via the factor model—that's the point of A1 and A2.

**Justification:** Explicit measurement error modeling for every observed variable requires known reliability coefficients. The framework prioritizes accessibility: users can model relationships with available data and acknowledge measurement limitations in interpretation.

---

## Future Considerations (Not Currently Assumed)

The following are explicitly NOT assumed and may be added in future versions:

- **Non-linear relationships:** Currently all structural effects are linear in parameters
- **Non-Gaussian distributions:** Currently residuals are assumed Gaussian
- **Time-varying parameters:** Currently all causal coefficients are time-invariant
- **Random slopes:** Currently only random intercepts, not person-specific effect sizes
- **Cross-level interactions:** Currently between-person variables do not moderate within-person effects
- **Formative measurement:** Currently only reflective models supported
