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
| Aggregate ranking | `sum_p pi_p * log(w_pi / w_pA)` | – | expected utility; utilities are the logs of the normalised priority vector, taken against the benchmark alternative |
| Variance prior | `sigma2 ~ InvGamma((lambda_s-2)/2, lambda_s*s2bar/2)` | `invgamma.logpdf(sigma2, a=(lambda_s-2)/2, scale=lambda_s*s2bar/2)` | density in `sigma2`; the shape exponent is `(alpha+1)` |
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

with `alpha = (lambda_s - 2) / 2` and `beta = lambda_s * s2bar / 2`. That is
exactly `scipy.stats.invgamma.logpdf(sigma2, a=alpha, scale=beta)`, which is how
the manuscript states the prior, as `phi_p ~ InvGamma((lambda_s-2)/2,
lambda_s*s2bar/2)` in `eq:prior1` of `sections/02_model.tex`. The same
distribution can be written as a Gamma on `1/sigma2`, but only if the density is
then read on `sigma2` with the Jacobian the change of variables contributes;
dropping that Jacobian would give exponent `(alpha - 1)` instead of
`(alpha + 1)` and would no longer match. Writing the prior directly on
`sigma2`, as the manuscript now does, avoids the question.

### 3. The M-step is the exact MAP of the stated prior

`inference.py` updates the variance as

```
sigma2 <- (SS + lambda_u * ||u - ubar||^2 + lambda_s * s2bar)
          / (sum_k gamma_k + lambda_s + (nA - 1))
```

with `SS = sum_k gamma_k * residual_k^2` and `nA - 1` the dimension of the
utility vector. Each term of the denominator is the contribution of one factor
of the posterior that carries `phi_p`:

| term | source |
|---|---|
| `sum_k gamma_k` | the Gaussian likelihood, one factor per observation |
| `nA - 1` | the normalising constant of the utility prior, whose density carries `phi_p^{-(A-1)/2}` |
| `lambda_s` | the Gamma density on `1/phi_p`, where `lambda_s = 2*(alpha + 1)` with `alpha = (lambda_s - 2)/2` |

The numerator carries the quadratic term of the utility prior.

**This section needed three corrections, and the history is worth keeping.**

The first version used `(SS + lambda_s * s2bar) / (sum_k gamma_k + lambda_s)`,
omitting the prior-density term and the utility prior's quadratic term alike;
that update was not a stationary point of the density
`penalized_log_likelihood` evaluates.

The second version restored the prior-density term as a bare `+4` -- correct
for the prior as it then stood, `Gamma((lambda_s+2)/2, ...)`, since then
`2*(alpha+1) - lambda_s = 4` -- but still omitted `nA - 1`.

The third and current version makes the denominator exactly `n_p + d + lambda_s`
and moves the prior to `Gamma((lambda_s-2)/2, lambda_s * s2bar / 2)`, which has
two effects. The prior mode of `phi_p` becomes exactly `sbar^2`; under the old
shape it was `lambda_s / (lambda_s + 4)` times that, so the manuscript sentence
"the prior mode `sbar^2` is placed on the inconsistency levels" was not true as
written. And the `+4` disappears from the update. The shape requires
`lambda_s > 2` to be a valid Gamma and `lambda_s > 4` for the prior mean to
exist; the manuscript states the latter.

The omission of `nA - 1` is invisible at large `n`: for the case study the two
denominators differ by about 0.15%, and refitting moves `pi` in the fourth
decimal. Where a component's effective sample size is small -- `K = 1` fits,
near-empty components, the external validation with ten observations per unit,
small-`A` simulation cells -- it is not.

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

### 5. Aggregate ranking is an expected utility, not a vote score

The manuscript used to aggregate with a vote score,
`Phi((u_pi - u_pj) / sigma_p)` averaged over pairs and components. That
expression exists only for the logit-normal variant, where the
probability that one alternative beats another has a closed form, so it
contradicted the claim that the observation model is exchangeable across
protocols. It is replaced by the expected utility

```
u_bar_i = sum_p pi_p * log(w_pi / w_pA)
```

which needs no distributional assumption. The implementation places the
benchmark alternative last in every priority vector and sets its utility to
zero, so the components share an origin and their utilities can be averaged;
averaging the raw weights instead would not be comparable across components.
`expected_utility` lives in `src/mop/mixture_of_priorities.py` and is called by
both the case-study and the simulation scripts.

A weight can sit at the edge of the simplex, so the log is taken against a
floor of `1e-300`. The near-degenerate component in the case study has weights
down to `1e-11`, which is finite but large enough in ratio terms to contribute
about a fifth of one alternative's aggregate; the paper says so rather than
suppressing it.

### 6. The noninformative limit keeps one term

`tests/` and the appendix both used to state that MAP reduces to maximum
likelihood as the priors flatten. That holds for the utility vectors and the
mixture weights but not for the variance. The normalising constant
`|lambda_u^-1 phi_p I|^{-1/2}` of the conditional utility prior carries
`phi_p^{-(A-1)/2}` at any `lambda_u`, so the denominator of the variance update
keeps its `A - 1`:

```
phi_p* = SS / (sum_k gamma_kp + (A - 1))
```

A numerical check confirms it: with `lambda_u = lambda_s = 1e-8`, `A = 4` and
`K = 80`, the maximum of `penalized_log_likelihood` sits at `SS / (n + A - 1)`
to a relative error of 1e-10, and differs from `SS / n` by 3.6%. The `A - 1`
would vanish only if the utility vector were integrated out first, which is the
marginal posterior rather than the joint one the EM iteration targets.

### 7. Beta-binomial

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

### 8. Transforms and ties

All transforms are natural-log based; nothing in `src/mop` uses base 10. A tie
is encoded as the score `0.5`, which maps to `logit = 0` — the same point as a
zero utility difference, which is what the model assumes it means. Scores are
clipped to `[1e-10, 1-1e-10]` before the logit so that the extremes stay finite.
