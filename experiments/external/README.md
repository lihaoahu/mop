# v3 — Generalization to a published choice-under-uncertainty dataset

This directory holds the external validation required after the referee report:
the model is applied to a published experiment on repeated choice under
uncertainty, rather than only to synthetic data and to our own case study.

## Data

`data/regenwetter2012.csv` — 90 rows (30 participants × 3 gamble sets), 32
columns. Each row reports, for each of the 10 gamble pairs, how many times out
of 45 the participant chose the first gamble, chose the second, or declared
indifference.

- **Study**: Regenwetter, M., & Davis-Stober, C. P. (2012). *Behavioral
  variability of choices versus structural inconsistency of preferences.*
  Psychological Review, 119(2), 408–416. doi:10.1037/a0027372
- **Distribution**: the same data ship with the CRAN package `multinomineq`
  (version 0.2.6) as the dataset `regenwetter2012`. They are copied here so the
  replication package needs no R dependency; please cite the original article
  when using them.
- **Why this dataset**: each pair was judged 45 times, which is what makes it
  possible to separate *intrinsic variability* (preference states that switch
  between judgments) from *structural inconsistency* (residual variability in
  executing a fixed state). It is also the dataset behind the mixture-over-
  orders framework this paper extends.

## Scripts

| File | Purpose |
|---|---|
| `step1_regenwetter.py` | Fits each participant × set unit as a repeated binary comparison under both the binomial and the beta-binomial likelihood, selecting the number of mixture components by BIC, and writes `regenwetter_results.json`. |
| `step2_separability.py` | Simulated separability check: generates data with and without within-state overdispersion and fits both likelihoods, testing whether a locked `rho` inflates the selected number of components. Writes `separability_results.json`. |
| `step3_scaling.py` | Population-level scaling: D decision makers share one set of components, so the observation count grows without adding alternatives. Asks whether more data repairs the confounding or whether the likelihood has to change. Writes `scaling_results.json`. |
| `step5_scalability.py` | Fits a distribution over the $\mathcal{A}!$ transitive orders and the four-component mixture of preferences to the *same* simulated pairwise data, across $\mathcal{A}=4..8$, comparing free parameters and wall-clock cost. Writes `scalability_results.json`. |

```bash
uv run python experiments/external/step1_regenwetter.py
uv run python experiments/external/step1_regenwetter.py --p-max 3 --n-starts 1   # faster
```

## What to look for

The two likelihoods differ in whether the dispersion parameter `rho_p` is free:

- `binomial` fixes `rho_p = 0`, so the variance of a repeated comparison is
  locked at `n·mu·(1-mu)`. Any dispersion beyond that must be absorbed by extra
  mixture components.
- `beta-binomial` lets `rho_p ∈ [1e-5, 0.9]` carry structural inconsistency, so
  the mixture is left to carry intrinsic variability.

Freeing `rho` should therefore move the selected number of components
**downward**. If it did not, the claim that the two sources of variability are
empirically confounded under a locked variance function would not hold on this
data.

Note on unanimous units: when every pair of a unit is decided 45–0, `rho` is not
identified — the likelihood is monotone in `rho` and reaches its maximum at the
boundary. Report `rho` only for units with at least one split decision; the
fitted ranking is unaffected either way. See `PARAMETERISATION.md` §6.

## Separability check (`step2_separability.py`)

Regenerating data under a known regime and fitting both models tests the claim
in both directions:

- regime `binomial` (`rho_gen = 0`): every bit of dispersion is genuine state
  switching, so the mixture should recover `P_true` while `rho` should be
  estimated at zero — a free `rho` must not invent overdispersion.
- regime `beta-binomial` (`rho_gen > 0`): part of the dispersion is within-state
  noise, so a locked-rho fit should **over-select** components, while a free-rho
  fit should recover both `P_true` and `rho_gen`.

Grid: `P_true` ∈ {1, 2, 3} × `rho_gen` ∈ {0, 0.10, 0.25}, 12 replicates per
cell, 6 alternatives (15 pairs) and 45 repetitions per pair. Fifteen pairs are
used rather than ten because ten cannot identify three components together with
a dispersion parameter — with five alternatives the experiment measures the
sample size rather than the modelling question.
