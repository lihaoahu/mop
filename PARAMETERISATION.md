# Parameterisation conventions

Statistical code fails quietly when the same symbol means different things in
different libraries: `scale` is a standard deviation in `scipy.stats.norm` but
a variance in some other toolkits, and "Gamma" is shape-rate in some places and
shape-scale in others. This file records every convention the project depends
on and where in the code each one is imposed.

## Summary

| Quantity | Convention in this project | scipy / numpy equivalent | Relation |
|---|---|---|---|
| Judgment transform | `logit(score)`, natural log | `scipy.special.logit` | |
| Structural inconsistency | `sigma2` is a **variance** on the log-odds scale | `norm.logpdf(x, scale=)` takes a **standard deviation** | `scale = sqrt(sigma2)` |
| Aggregate ranking | `sum_p pi_p * log(w_pi / w_pA)` | – | expected utility; utilities are the logs of the normalised priority vector, taken against the benchmark alternative |
| Variance prior | `phi ~ InvGamma(lambda_sigma/2, lambda_sigma*phi_0/2)` | `invgamma.logpdf(phi, a=lambda_sigma/2, scale=lambda_sigma*phi_0/2)` | one strength parameter; mode `lambda_sigma*phi_0/(lambda_sigma+2)`; density in `phi`, so the shape exponent is `(alpha+1)` |
| Utility prior | `u_p \| sigma2_p ~ N(u_0, lambda_u^-1 sigma2_p I)` | `norm.logpdf(u, loc=u_0, scale=sqrt(sigma2/lambda_u))` | |
| Mixture prior | `pi ~ Dirichlet(lambda_pi, ..., lambda_pi)` | `numpy.random.dirichlet` | concentration |
| Link / response | `logit` / `expit`, natural log | `scipy.special` | no base-10 anywhere |
| Ties | score `0.5` -> `logit = 0` | – | scores clipped to `[1e-10, 1-1e-10]` |
| Synthetic comparisons | noise added on the **log-odds** scale, `N(0, sigma2)` | – | verified empirically |

## Details

### 0. Code identifiers against manuscript symbols

The code names the dispersion `sigma2` and the prior locations `s2` and `u_bar`;
the manuscript writes the dispersion `phi_p`, its prior mode `phi_0`, and the
prior mean of the utilities `u_0`. The manuscript used to write the prior mode
`sbar^2`, which was left over from the parameterisation in which the dispersion
was itself a variance called `sigma^2`; `phi_0` says what the quantity is
measured in. The identifiers are unchanged, so a call site still reads
`s2=`/`u_bar=`, and `u_bar_i` in `mixture_of_priorities.py` is a different
object again: the expected utility of Section 2.6.

### 1. `sigma2` is a variance, never a standard deviation

`src/mop/inference.py` computes `stds = np.sqrt(sigma2)` before calling
`norm.logpdf(..., scale=stds)`. This is the convention every downstream quantity
inherits. The audit confirms that using `scale=sigma2` instead changes the
log-likelihood by a visible amount, so the distinction is worth stating
explicitly rather than assuming.

### 2. The dispersion prior is inverse-Gamma, with the Jacobian

`penalized_log_likelihood` evaluates

```
alpha * log(beta) - gammaln(alpha) - (alpha + 1) * log(phi) - beta / phi
```

with `alpha = lambda_sigma / 2` and `beta = lambda_sigma * phi_0 / 2`. That is
exactly `scipy.stats.invgamma.logpdf(phi, a=alpha, scale=beta)`, which is how
the manuscript states the prior in `eq:prior1` of `sections/02_model.tex`,
`phi_p ~ InvGamma(lambda_sigma/2, lambda_sigma*phi_0/2)`.

One parameter does two jobs here, and that is the point of the form. `beta` is
the scale and `2*alpha` the strength, both proportional to `lambda_sigma`, so as
`lambda_sigma -> 0` the two vanish together and the density tends to
`phi^{-1}` -- the scale-invariant prior, up to a constant in `phi` that the
posterior does not see. The non-informative case is therefore the limit of the
family rather than a value it excludes: an inverse-Gamma with a free shape has
to keep that shape above one for its mode and above two for its variance, so it
cannot reach the limit at all. The price is that the mode is
`lambda_sigma*phi_0/(lambda_sigma+2)`, which is `phi_0` only in the limit; at the case study's `lambda_sigma = 108.02`
it is `0.982*phi_0`. The mean needs `lambda_sigma > 2` and the variance
`lambda_sigma > 4`.

The same distribution can be written as a Gamma on `1/phi`, but only if the
density is then read on `phi` with the Jacobian the change of variables
contributes; dropping that Jacobian would give exponent `(alpha - 1)` instead
of `(alpha + 1)` and would no longer match. Writing the prior directly on
`phi`, as the manuscript does, avoids the question.

### 3. The M-step is the exact MAP of the stated prior

`inference.py` updates the dispersion as

```
phi_p <- (lambda_sigma * phi_0 + SS + lambda_u * ||u_p - u_0||^2)
         / (lambda_sigma + 2 + sum_k gamma_kp + (A - 1))
```

with `SS = sum_k gamma_kp * residual_k^2` and `A - 1` the dimension of the
utility vector. Each term is the contribution of one factor of the posterior
that carries `phi_p`:

| term | source |
|---|---|
| `sum_k gamma_kp` in the denominator | the Gaussian likelihood, one factor per observation, whose density carries `phi_p^{-1/2}` |
| `SS` in the numerator | the same factors' quadratic term |
| `A - 1` | the normalising constant of the utility prior, whose density carries `phi_p^{-(A-1)/2}` |
| `lambda_sigma + 2` | the inverse-Gamma read as a Gamma on `1/phi`: `2*beta = lambda_sigma*phi_0` is the numerator term and `2*alpha = lambda_sigma` the denominator term, while the `+2` is the Jacobian |

**The `+ 2` cannot be dropped**, and the history of this section is why it is
worth saying so. Four versions, in order:

The first used `(SS + lambda_sigma * phi_0) / (sum_k gamma_k + lambda_sigma)`,
omitting the prior-density term and the utility prior's quadratic term alike;
that update was not a stationary point of the density
`penalized_log_likelihood` evaluates.

The second restored the prior-density term as a bare `+4` -- correct for the
prior as it then stood, `Gamma((lambda_sigma+2)/2, ...)` -- but still omitted
`A - 1`.

The third made the denominator `n_p + d + lambda_sigma` and moved the prior to
`InvGamma((lambda_sigma-2)/2, lambda_sigma*phi_0/2)`, written directly on
`phi_p`. That put the mode exactly at `phi_0` and removed the `+4`, at the cost
of requiring `lambda_sigma > 2` for the mode and `> 4` for the mean to exist.

The fourth and current version changes the shape to `lambda_sigma/2`, which is
the third version's `lambda_sigma` reparameterised as `lambda_sigma + 2`. It
gives up the exact mode at `phi_0` and buys a family whose `lambda_sigma -> 0`
limit is the scale-invariant prior. The `+ 2` in the denominator is where the
old `+ 2` inside `lambda_sigma` now sits: it moved, it did not disappear, and a
reader who drops it is fitting a different prior.

The omission of `A - 1` is invisible at large `n`: for the case study the two
denominators differ by about 0.15%, and refitting moves `pi` in the fourth
decimal. Where a component's effective sample size is small -- `K = 1` fits,
near-empty components, small-`A` simulation cells -- it is not. `A` is the
parameter that exposes it, and it is the one to vary when the update is checked
against a numerical maximiser: a check that builds its own objective function
reproduces whatever the implementation assumed, so it agrees with a wrong
update as readily as with a right one.

### 4. The dispersion is weakly identified, so its reported precision is limited

The EM loop stops when successive values of the penalized log-posterior agree
to a relative tolerance, `|dPLL| / (|PLL| + 0.1) < 1e-6`. The `+ 0.1` is the
scale normalisation flexmix uses; without it the test tightens as the
log-posterior grows in magnitude, which for these data made a 1e-10 ratio
unreachable and ran the loop to its 10000-iteration cap.

The threshold is a choice with a cost. At 1e-10 the case study reproduces the
values the tables carried before this round, while at 1e-6 the third component's
PHEV/REEV split moves from (0.391, 0.415) to (0.407, 0.397) and the ordering
inside that component flips. The tables were regenerated at 1e-6 rather than the
code tightened to 1e-10, so the reported numbers and this threshold belong
together: moving the threshold moves what the tables should say.

The dispersion is the weakly identified quantity. On the cancellation replicate
of Table 1 (seed 1000) the two fitted dispersions are 0.0452 and 0.0427 against
a design value of 0.05, while the one-component model puts the same judgments at
1.368: the same data, an order of magnitude apart, depending on whether the
mixture is allowed a second component. That is why `phi` should not be read to
four significant figures while the mixture weights and the priority vectors are
stable to four.

### 4b. The prior mode of the dispersion has one definition

`inference.prior_mode_phi0(X, y)` returns `SS / (K - p)` with `p = A - 1` free
utilities, which is `eq:prior_mode` of the manuscript: the unbiased residual
variance of the single-preference fit. It is called by `mop`, for the fit, and
by `pbic`, for the criterion, so one run cannot use two different prior modes.

It used to. The scripts each computed their own as `SS / (K - p + 1)` — their
`nA` holds `p` but was used as `A` — while `mop` fell back to `SS / K` when the
caller passed nothing, which every caller did. So the fit used one prior mode
and the criterion another, 0.06 per cent apart on the case study. The scripts no
longer compute it at all; `prior_mode_phi0` is the only place the formula
appears.

### 5. Aggregate ranking is an expected utility, not a vote score

The manuscript used to aggregate with a vote score,
`Phi((u_pi - u_pj) / sigma_p)` averaged over pairs and components. That
expression exists only for the Gaussian variants, where the
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
`case_study/estimator.py`, which every case-study table goes through.

A weight can sit at the edge of the simplex, so the log is taken against a
floor of `1e-300`. The near-degenerate component in the case study has weights
down to `1e-11`, which is finite but large enough in ratio terms to contribute
about a fifth of one alternative's aggregate; the paper says so rather than
suppressing it.

### 6. The noninformative limit keeps two terms

It is natural to expect MAP to reduce to maximum likelihood as the priors
flatten. That holds for the utility vectors and the mixture weights but not for
the dispersion. Two factors of the posterior carry `phi_p` however weak the
priors are:

- the normalising constant `|lambda_u^-1 phi_p I|^{-1/2}` of the conditional
  utility prior, which contributes `-(A-1)/2 * log phi_p` at every `lambda_u`,
  hence the `A - 1`;
- the Jacobian of the inverse-Gamma read as a Gamma on `1/phi`, which is the
  `+ 2` of §3. It does not weaken with `lambda_sigma`, because it belongs to the
  change of variables rather than to the strength of the prior.

So the limit is

```
phi_p* = SS / (sum_k gamma_kp + (A - 1) + 2)
```

A numerical check at `lambda_u = lambda_sigma = 1e-8`, `A = 4` and `n = 83` puts
the maximum of `penalized_log_likelihood` at that value to a relative error
below `1e-9`, while `SS / n` sits 6.0% above it. Both terms would vanish only if
the utility vector were integrated out first *and* the prior were left on the
natural scale, which is the marginal posterior rather than the joint one the EM
iteration targets.

### 7. The beta-binomial baseline

The estimator the paper fits to binary data carries no dispersion -- that is
the point of Section 2.3 -- but the binomial protocol of Table S4 is compared
against a single preference whose overdispersion is explicit, and
`experiments/validation/binomial.py` fits it. The parameterisation is the
mean-dispersion one, `(n, mu, rho)`:

```
p    = expit(X @ u),   conc = 1/rho - 1,   a = p*conc,   b = (1-p)*conc
log f(m | n) = log C(n, m) + betaln(m + a, n - m + b) - betaln(a, b)
```

with `m = y*n` successes out of `n` trials. `rho` is the intra-class
correlation: at `rho -> 0` the Beta shapes diverge and the likelihood tends to
the binomial, which is what makes this a nesting rather than a rival family.
The script maps `rho = expit(z)*0.98 + 1e-4` and bounds `z` to `(-9, 4)`, so
`rho` stays inside `(1e-4, 0.98)`.

The cap at the top matters. As `rho -> 1` the Beta shapes `a, b -> 0`, and on
perfectly separated data -- every pair decided unanimously -- the likelihood is
monotone in `rho` and maximised at the boundary. `rho` is then not identified
for that unit, and the fitted ranking is the same either way.

The fit is `L-BFGS-B` on the penalised log-likelihood
`-sum log f + 0.5*lambda_u*||u||^2`, started from the one-component MoP fit on
the training fold, so the two models see the same data and the same utility
prior and the held-out comparison is like for like. Nothing here needs a
floored `digamma`: an earlier module fitted the same parameterisation by
gradient ascent and needed one, and that module (`inference_binomial.py`) was
removed in 73466ff together with the external validation that used it. The
current script is the only implementation in the tree.

### 8. Transforms and ties

All transforms are natural-log based; nothing in `src/mop` uses base 10. A tie
is encoded as the score `0.5`, which maps to `logit = 0` — the same point as a
zero utility difference, which is what the model assumes it means. Scores are
clipped to `[1e-10, 1-1e-10]` before the logit so that the extremes stay finite.
