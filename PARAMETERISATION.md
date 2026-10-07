# Parameterisation conventions

Statistical code fails quietly when the same symbol means different things in
different libraries: `scale` is a standard deviation in `scipy.stats.norm` but
a variance in some other toolkits, and "Gamma" is shape-rate in some places and
shape-scale in others. This file records every convention the project depends
on and how each one was verified. Re-run the audit at any time:

```bash
uv run python tests/audit_parameterisation.py
```

It must print `ALL CONVENTIONS VERIFIED` and exit with status 0.

## Summary

| Quantity | Convention in this project | scipy / numpy equivalent | Relation |
|---|---|---|---|
| Judgment transform | `logit(score)`, natural log | `scipy.special.logit` | |
| Structural inconsistency | `sigma2` is a **variance** on the log-odds scale | `norm.logpdf(x, scale=)` takes a **standard deviation** | `scale = sqrt(sigma2)` |
| Vote score | `Phi((u_i - u_j) / sigma_p)`, denominator is the **standard deviation** | `norm.cdf` | reproduces manuscript Table 3 |
| Variance prior | `1/sigma2 ~ Gamma((lambda_s+2)/2, lambda_s*s2bar/2)` | `invgamma.logpdf(sigma2, a=(lambda_s+2)/2, scale=lambda_s*s2bar/2)` | Inverse-Gamma density, **Jacobian included** |
| Utility prior | `u_p \| sigma2_p ~ N(ubar, lambda_u^-1 sigma2_p I)` | `norm.logpdf(u, loc=ubar, scale=sqrt(sigma2/lambda_u))` | |
| Mixture prior | `pi ~ Dirichlet(lambda_pi, ..., lambda_pi)` | `numpy.random.dirichlet` | concentration |
| Repeated binary judgment | `m_k ~ BetaBinomial(n_k, mu_kp, rho_p)` | `betabinom.logpmf(k, n_k, a, b)` | `a = mu*(1-rho)/rho`, `b = (1-mu)*(1-rho)/rho` |
| Overdispersion | `rho` = intra-pair correlation | – | `rho = 1/(a+b+1)`; `Var = n*mu*(1-mu)*[1+(n-1)*rho]` |
| Degenerate `rho = 0` | exactly the binomial | `binom.logpmf(k, n, mu)` | verified to 1e-12 |
| Link / response | `logit` / `expit`, natural log | `scipy.special` | no base-10 anywhere |
| Ties | score `0.5` -> `logit = 0` | – | scores clipped to `[1e-10, 1-1e-10]` |
| Synthetic comparisons | noise added on the **log-odds** scale, `N(0, sigma2)` | – | verified empirically |

## Details

### 1. `sigma2` is a variance, never a standard deviation

`src/mop/inference.py` computes `stds = np.sqrt(sigma2)` before calling
`norm.logpdf(..., scale=stds)`, and `src/mop/decision.py` does the same. This is
the convention every downstream quantity inherits. The audit confirms that
using `scale=sigma2` instead changes the log-likelihood by a visible amount, so
the distinction is worth stating explicitly rather than assuming.

### 2. The variance prior is inverse-Gamma, with the Jacobian

`penalized_log_likelihood` evaluates

```
alpha * log(beta) - gammaln(alpha) - (alpha + 1) * log(sigma2) - beta / sigma2
```

with `alpha = (lambda_s + 2) / 2` and `beta = lambda_s * s2bar / 2`. That is
exactly `scipy.stats.invgamma.logpdf(sigma2, a=alpha, scale=beta)`: the
`(alpha + 1)` exponent comes from the change of variables `tau = 1/sigma2`, so
the prior is stated on `1/sigma2` but the density the code evaluates lives on
`sigma2`. Dropping the Jacobian would give exponent `(alpha - 1)` and would no
longer match. This is consistent with the manuscript wording "a moderately
informative inverse-Gamma prior" (`sections/03_empirical.tex`).

### 3. The M-step is the exact MAP of the stated prior

`inference.py` updates the variance as

```
sigma2 <- (SS + lambda_u * ||u - ubar||^2 + lambda_s * s2bar)
          / (sum_k gamma_k + lambda_s + (nA - 1) + 4)
```

with `SS = sum_k gamma_k * residual_k^2` and `nA - 1` the dimension of the
utility vector. Each term of the denominator is the contribution of one factor
of the posterior that carries `phi_p`:

| term | source |
|---|---|
| `sum_k gamma_k` | the Gaussian likelihood, one factor per observation |
| `nA - 1` | the normalising constant of the utility prior, whose density carries `phi_p^{-(A-1)/2}` |
| `lambda_s + 4` | the inverse-Gamma density, where `4 = 2*(alpha + 1)` with `alpha = (lambda_s + 2)/2` |

The numerator carries the quadratic term of the utility prior.

**Two successive corrections were needed here.**

The first version used `(SS + lambda_s * s2bar) / (sum_k gamma_k + lambda_s)`,
omitting both the inverse-Gamma term and the utility prior; that update was not
a stationary point of the density `penalized_log_likelihood` evaluates.

The second version restored `2*(alpha+1)` as a bare `+4` but still omitted
`nA - 1`. The omission is invisible at large `n`: for the case study the wrong
and right denominators differ by about 0.14%, and refitting the case-study data
moves `pi` in the fourth decimal and leaves the ranking unchanged. It is **not**
small where the component's effective sample size `sum_k gamma_kp` is small —
`K = 1` fits, near-empty components, the external validation (ten observations
per unit), and small-`A`, small-`n` simulation cells.

`tests/audit_parameterisation.py` checks the update against a numerical
maximiser, and now sweeps `A` across the test cases. It is worth recording why
that sweep matters: the first two versions of this audit passed while the code
was wrong, because the audit built its own objective function and reproduced
the same omission. An audit that restates the implementation's assumption
cannot test it. The section now varies `A`, which is the only parameter that
exposes a missing `nA - 1`, and its `closed form` line is the implementation's
formula rather than a restatement of it.

### 4. sigma^2 is weakly identified, so its reported precision is limited

The EM loop stops when successive values of the penalized log-posterior agree
to a relative tolerance. On the cancellation replicate the log-posterior is
flat enough in `sigma2` that a change below 1e-10 still leaves the variance
free to move by 1–5%, while the utility vectors and mixture weights settle to
four decimals. Concretely, loosening the tolerance from 1e-20 to 1e-10 moved
that replicate's two component variances from 0.1005/0.1003 to 0.0988/0.0987
and its single-component variance from 1.4082 to 1.3972, with the priority
vectors unchanged at 1e-4.

Consequences:

- `sigma2` should not be read to four significant figures. On this data the
  second digit is the honest limit; the mixture weights and priority vectors
  are stable to four.
- The tolerance is 1e-10 rather than the earlier 1e-20. That earlier value
  lies below double precision for these magnitudes, so the loop ran to its
  10000-iteration cap instead of stopping: the tolerance was decorative and
  the runtime an order of magnitude larger than necessary. The trajectory was
  checked to confirm the iteration is monotone (the penalized log-posterior
  rises from −896 to a flat −516.7 within 50 iterations), so this is a weak
  direction of the objective rather than oscillation.

### 5. Vote score

Manuscript `eq:vote` writes `Phi((u_pi - u_pj) / sigma_p)` with
`epsilon_p ~ N(0, sigma_p^2)`, so the denominator is a standard deviation, and
`u_pi - u_pj = log(w_pi / w_pj)` because `w` is the normalised exponential of
`u`. Recomputing the vote scores from the published component parameters of
Table 3 gives 0.2116 / 0.2618 / 0.3125 / 0.2140 against the printed
0.212 / 0.262 / 0.313 / 0.214 — a maximum deviation of 0.0005, i.e. rounding.
Using `sigma_p^2` in the denominator instead would miss those values, so the
convention is confirmed rather than assumed.

### 6. Beta-binomial

`src/mop/inference_binomial.py` uses the mean-dispersion parameterisation
`(n, mu, rho)`, which does not exist in scipy. It was validated against
`scipy.stats.betabinom(k, n, a, b)` through `a = mu * c`, `b = (1 - mu) * c`
with `c = (1 - rho) / rho`: the two agree to 1e-12 over 300 random parameter
draws, and the variance identity
`n * mu * (1 - mu) * [1 + (n - 1) * rho]` matches `betabinom.var` to 1e-9.

Two further points about `rho`:

- it is bounded to `[1e-5, 0.9]`. The upper bound is not cosmetic: as
  `rho -> 1` the Beta shapes `a, b -> 0`, and on perfectly separated data
  (every pair decided unanimously) the likelihood is monotone in `rho` and
  maximised at the boundary. `rho` is then not identified for that unit, and
  the fitted ranking is the same either way.
- the gradient uses a floored `digamma` (`_digamma_safe`) because
  `digamma(a)` diverges as `a -> 0`.

### 7. Transforms and ties

All transforms are natural-log based; nothing in `src/mop` uses base 10. A tie
is encoded as the score `0.5`, which maps to `logit = 0` — the same point as a
zero utility difference, which is what the model assumes it means. Scores are
clipped to `[1e-10, 1-1e-10]` before the logit so that the extremes stay finite.
