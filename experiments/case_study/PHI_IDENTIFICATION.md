# Does the structural inconsistency change the selected number of states?

**Status: the parameters are now fixed and audited, and the answer is narrower than
in any earlier draft of this file.** Three earlier versions of this analysis were
wrong; the corrections are recorded because each one was found by a check that is
still in the code, and the checks are what make the current numbers quotable.

## The verified parameter set

Everything the comparison depends on lives in **one** file,
`experiments/mop_params.json`, read through `experiments/_mop_params.py`. Before that
file existed the same five values were copied into five scripts and described but
never read by two `config.json` files, so a change to one copy could silently
disagree with the rest.

| parameter | value | note |
|---|---|---|
| `lambda_u` | 1e-6 | ridge on the utilities |
| `lambda_pi` | 1 | uniform Dirichlet on the mixture weights |
| `lambda_sigma` | `0.02 * nJ` = `nJ / 50` | 108.02 on the corpus |
| **`phi_0`** | **`prior_mode_phi0(X, y)`** | the paper's rule, **chosen by the author** |
| grid | P = 1..5 | `mop_params.json` allows 9 |
| EM tol / cap | 1e-6 / 5000 | `mop.inference` |

`experiments/_mop_params_check.py` verifies three things and is the gate for quoting
any of this:

1. **the coefficient is 1/50** — an arithmetic identity, not a fitted quantity;
2. **the configs agree** — `phi_config.json` carries a copy, and a copy that disagrees
   is how the two drifted apart in the first place;
3. **`step7_analyze.py` reproduces `analysis_output.txt`** — that file is the source of
   every number the case study reports.

All three pass.

**A coefficient error was caught by check 3, not by review.** When `mop_params.json`
was first written, `lambda_sigma_over_n` was set to `0.05`, which is `nJ / 20` — a
prior differing by a factor of 2.5 from the paper's `nJ / 50 = 0.02 * nJ`. Nothing
complained, because a wrong prior is still a valid prior; what exposed it was that
`step7_analyze.py` stopped reproducing the recorded output, selecting K = 3 where the
recorded run has K = 4.

**The error did not contaminate the φ results below.** `phi_config.json` carried the
same wrong coefficient, but `step9_phi_identification.py` never read it — it computed
`nJ / 50` itself — so every φ number published before the fix is reproduced digit for
digit after it. That was checked rather than assumed: the two results files are
identical at every pBIC to 1e-6. The corrected file matters for what comes next, not
for what has already been reported.

**The identity that gates the comparison.** With `phi_0` shared by both fits, fixing
`phi = 1` restricts the free model to a face of its parameter space, so the two
penalized objectives must agree exactly at `phi = 1`. They do, to 0.000e+00, and
`_audit_params.py` fails if they do not. An earlier version dropped the Inverse-Gamma
constant from the fixed-phi objective, which broke the identity by 33 units and made
"the free fit must not do worse" a test of nothing.

## Result 1 (robust): the estimated dispersions sit mostly below one

Across every slice and every component count, under the paper's own prior rule,
`phi-hat` runs from 0.234 to 2.060 with median 0.650, and **76.0% of the estimates
are below 1**.

The claim holds under the two alternative priors as well:

| prior | `phi_0` | `lambda_sigma` | median `phi-hat` | share below 1 |
|---|---|---|---|---|
| data (chosen) | `prior_mode_phi0(X,y)` | `nJ/50` | 0.650 | 76.0% |
| one | 1 | `nJ/50` | 0.486 | 88.0% |
| weak | `prior_mode_phi0(X,y)` | 2.01 | 0.338 | 84.7% |

`_phi_robustness.py` produces that table. The median moves with the prior, as it
must — a smaller `phi_0` pulls the estimates down — but under all three the majority
of estimates is below one, and the per-slice pattern is the same.

**A noninformative prior is not a fourth candidate**, and that is a finding rather
than an omission. With `lambda_sigma -> 0` the likelihood is unbounded as
`phi -> 0`, which Section 2.4 of the paper states outright; fitting the policy slice
that way sends the secondary components' `phi` to 0.15 and 0.08. The prior is what
makes the problem well posed, so it cannot be used to test sensitivity to the prior's
location. The weakest proper prior (`lambda_sigma = 2.01`, just above the `> 2` that
the mode requires) is the honest limit of that exercise.

**What this means.** The nominal ratio model puts the variance of `logit(r)` at
`phi_p`. Independent human judgments with the assumed spread would put `phi-hat` near
`phi_0`. It comes out below one under every prior tried. The reading that fits:
these scores come from an extraction pipeline rather than from independent
elicitation, so the responses are **more consistent than an independent judgment
process would be**. That is the concrete answer to "what does `phi` buy" — on the
paper's own Gaussian data it is a free parameter that moves well away from the value
the binomial protocol would have imposed — and it is the only quantity in this whole
exercise that does not depend on a model-selection criterion.

## Result 2 (not robust): the component count

> **The table below was produced with the prior mode PINNED AT 1, under a note that
> claimed it was the paper's rule. It is superseded by `selection_facts.json`, produced
> by `_selection_facts.py` with the paper's `prior_mode_phi0` and the K-Ridge
> from-scratch initialisation.** The direction of the conclusion does not change — a fixed
> dispersion selects 4 states on the corpus and a free dispersion selects 3 — but the
> numbers below are for a different prior than the one the paper uses, so quote the
> script rather than this table. The fixed-φ column is the one affected: with the
> paper's prior it selects 4, not 3.

Selection by pBIC, `phi_0` per slice, grid P = 1..5:

| slice | n | φ fixed | φ free | moves |
|---|---|---|---|---|
| overall | 5401 | 4 | **3** | down |
| cost | 1570 | 2 | 2 | — |
| driving experience | 753 | 1 | 1 | — |
| range | 835 | 2 | 2 | — |
| refuelling convenience | 617 | 2 | 2 | — |
| reliability | 384 | 1 | **2** | up |
| policy | 281 | 1 | **2** | up |
| other | 798 | 3 | **2** | down |
| 2025 | 2973 | 4 | 4 | — |
| 2026 | 2156 | 3 | **2** | down |

So the count moves on 5 of 10 slices, in both directions. But:

* **The count depends on the prior as well.** Changing `phi_0` from the paper's rule
  to 1 changes the corpus from 4→3 to 4→5. With three defensible prior choices
  producing three different corpus answers, no component-count claim from this
  comparison can be quoted.
* **It depends on the grid.** On the cost slice the same procedure picks P = 3 in
  every bootstrap replicate once P = 2 is excluded, while the full grid puts P = 2
  first.
* **pBIC and an ordinary BIC disagree** on most slices (cost 2 against 4, driving
  experience 1 against 3, refuelling 2 against 4, other 2 against 4, and so on). The
  two differ by the prior terms, and on this data that is enough to change the answer.
* **The free-φ selection is unstable.** Bootstrapping the judgments, the fixed-φ
  count on the corpus is P = 4 in 9 of 10 replicates with a gap of about 56; the
  free-φ count is a three-way contest (P = 5 in 70%, P = 6 in 20%, P = 4 in 10%) with
  a gap of 33 ± 40.

**Conclusion for the paper.** The case study's four-component structure should not be
restated on the free-`phi` parameterisation. Whether it survives there is not
resolvable at this sample size, and that belongs in Limitations. What the paper can
say with support is Result 1.

## The three errors this analysis went through

Recorded because each was caught by a check that is still in the repository, and
because they are the reason the current numbers should be trusted and the earlier
ones should not.

1. **Prior mode confounded with the parameterisation.** The first comparison ran the
   fixed-φ fit at `s2 = var(y)` and the free-φ fit at `s2 = prior_mode_phi0`. Fixing
   φ at one is itself a claim about the prior mode, so that changed two things at
   once. Two conclusions drawn from it were wrong: "freeing φ removes a component"
   and "the policy minority component is unnecessary".
2. **The objectives were not nested.** The fixed-φ objective dropped the
   Inverse-Gamma constant rather than evaluating the same objective at φ = 1, so the
   two differed by 33 units at the point where they must agree. Found by the identity
   check in `_audit_params.py`.
3. **A grid that excluded the winner.** An early bootstrap on the cost slice used
   grid P = 3..6 and therefore never considered P = 2, which the full grid prefers by
   67.9. The bootstrap now reports its grid next to its result.

## Reproduce

```
uv run python experiments/case_study/_validate_phi.py       # the three acceptance checks
uv run python experiments/case_study/_audit_params.py       # parameters + identity
uv run python experiments/case_study/_check_nesting.py      # free >= fixed everywhere
uv run python experiments/case_study/_check_step9.py        # vs mop.inference
uv run python experiments/case_study/step9_phi_identification.py --all --grid 1 2 3 4 5
```

**Note on what was removed.** Five scripts that produced intermediate results for this
document have been deleted, because the analysis they belong to is finished and its
conclusions are now in the manuscript, so leaving them to be rerun served only to make the
experiment directory look larger than the paper's evidence:

  * `_phi_robustness.py` (the phi-hat table under three priors, above),
  * `_phi_summary.py`,
  * `_bootstrap_pbic.py` and `_bootstrap_via_mop.py`, the resampling studies of the
    selected component count. The manuscript stopped reporting them: the controlled
    simulations establish the recovery, so a resampling stability check on the case
    study was answering a question the simulations already answer, and it was removed
    from the paper along with the scripts.
  * `_selection_facts.py` and `_pbic_curve.py` (superseded by
    `_specification_sensitivity.py` and `selection.py`),
  * `_diag_policy_p1.py` and `_diag_step7_params.py`, which diagnosed problems that are
    now fixed.

The tables above are kept as the record of what was found at the time; the numbers the
manuscript quotes come from `_specification_sensitivity.py`, `protocol_identifiability.py`
and `phi_recovery.py`, all of which remain, and from `simulation/step6_binomial.py`, which
supplies the binomial-protocol results.
