# Does the structural inconsistency change the selected number of states?

Short answer: **the fixed-φ answer is stable, the free-φ answer is not.** Freeing φ
changes the reported count on 4 of 10 slices and makes the own-corpus selection
unstable. It does not follow that the fixed-φ count is wrong — it follows that the
count depends on the parameterisation *and* on how decisively the criterion resolves
the models, and that the second effect is the larger one here.

> **This file supersedes two earlier versions of itself.**
>
> *First*: a run that compared a fixed-φ fit using `s2 = var(y)` against a free-φ fit
> using `s2 = prior_mode_phi0(X, y)`. Fixing φ at one is itself a claim that one is
> the prior mode, so that comparison changed two things at once. The implementation
> was never wrong — the free-φ branch reproduced `inference.mop` exactly (pBIC 895.60
> against 895.60 at policy P = 1) — but the *comparison* was, and two conclusions
> drawn from it are retracted below.
>
> *Second*: an intermediate write-up that reported the corrected comparison as if it
> established that φ inflates the component count. The bootstrap below shows the
> free-φ selection is unstable, which is the opposite of establishing anything about
> the count.

## What was run

`step9_phi_identification.py` fits the case study's ratio judgments twice with the
paper's own Gaussian EM — once with `φ_p` estimated, once with `φ_p` fixed at one —
over the same grid, from the same K-Ridge initialisation strategy, with **`φ_0 = 1`
shared by both**, so the only difference is whether `φ_p` is a parameter. The
fixed-φ variant drops the Inverse-Gamma prior on `φ_p` and the `φ_p` dependence of
the normal prior's normalising constant, since a fixed value must not carry a penalty
it can no longer optimise against.

**Two checks were run before the comparison was trusted.**

*Implementation.* `_check_step9.py` compares the free-φ fit against the repository's
`inference.mop`: pBIC agrees to within 0.4 at P = 1, 2, 3.

*Nesting.* With `φ_0` shared, fixing `φ = 1` restricts the free model to a face of
its parameter space, so the free maximiser must reach a penalized objective at least
as high as the fixed one. `_check_nesting.py` verifies this from a common start: the
gain from freeing φ is non-negative in every case but one (policy P = 1, −0.37, and
that count is rejected under both parameterisations anyway). **The earlier, flawed
comparison failed this property; the corrected one passes it.**

## Selection by pBIC, φ₀ = 1, grid P = 1..5

| slice | n | φ fixed | φ free | moves | φ̂ at the free optimum |
|---|---|---|---|---|---|
| overall | 5401 | 4 | **5** | +1 | 0.53, 0.96, 0.37, 0.51, 0.60 |
| cost | 1570 | 2 | 2 | — | 0.79, 0.53 |
| driving experience | 753 | 1 | 1 | — | 1.04 |
| range | 835 | 2 | 2 | — | 0.52, 0.86 |
| refuelling convenience | 617 | 2 | 2 | — | 0.52, 0.74 |
| reliability | 384 | 1 | **2** | +1 | 0.23, 1.27 |
| policy | 281 | 1 | **2** | +1 | 0.48, 0.54 |
| other | 798 | 3 | **2** | −1 | 1.43, 0.78 |
| 2025 | 2973 | 4 | 4 | — | 0.47, 0.93, 0.43, 0.61 |
| 2026 | 2156 | 3 | **4** | +1 | 0.35, 0.42, 0.59, 0.58 |

The count is not invariant to the parameterisation on 4 of 10 slices, and one of
those moves downward. There is no one-way bias: of the four that move, three move up.

## Stability, which matters more than the point estimates

`_bootstrap_pbic.py` resamples the judgments with replacement and recounts, grid
P = 3..6, 10 replicates:

| slice | parameterisation | selection across replicates | best-vs-second gap |
|---|---|---|---|
| overall | φ fixed | P = 4 in **90%**, P = 3 in 10% | 56.4 ± 25.6 |
| overall | φ free | P = 5 in **70%**, P = 6 in 20%, P = 4 in 10% | 33.0 ± 39.6 |
| cost | φ fixed | P = 3 in **100%** | 69.0 ± 3.9 |
| cost | φ free | P = 3 in **75%**, P = 4 in 25% | 28.0 ± 21.2 |

Two readings, both of which need stating:

1. **The fixed-φ selection is the stable one.** On the corpus it picks P = 4 in 9 of
   10 replicates with a gap of about 56; freeing φ turns the same 10 replicates into
   a three-way contest with a gap of 33 ± 40, meaning individual replicates range
   from decisive to nearly tied.
2. **A selection can depend on the grid as much as on the data.** On the cost slice
   the same procedure picks P = 3 in every replicate when P = 2 is excluded from the
   grid, while the full grid puts P = 2 first with a 67.9 margin. Any component count
   reported from this model needs the grid and the criterion printed next to it.

So the honest summary of the whole exercise: **the case study's four-component
structure survives the correction on the fixed-φ parameterisation and does not
clearly survive on the free-φ one — but the free-φ model is the one the paper
defines, and on that model the corpus selection is not resolvable at this sample
size.** That is a limitation to report, not a result to lean on.

## The finding that is robust: φ̂ sits mostly below one

Across all slices and component counts the estimated dispersions run from 0.22 to
2.04, with **median 0.51**, and **88.7% of them are below one**.

The ratio protocol's nominal model puts the variance of `logit(r)` at `φ_p`
independently of the utility difference. If the scores were independent human
judgments with the assumed spread, `φ̂` would sit near `φ_0 = 1`. It sits at about
half that.

The reading that fits: these scores come from an extraction pipeline, not from
independent elicitation, so the responses are **more consistent than an independent
judgment process would be** — the underdispersion is a property of how the data were
produced. This is the strongest thing the exercise produced, and it is robust to
everything above: it holds on every slice, at every component count, on both
parameterisations.

It matters twice over. It is direct evidence that `φ_p` is doing real work on this
protocol, estimated well away from one; and it means the case study's dispersion
estimates should not be read as human inconsistency without a caveat about the
extraction step.

## Established, retracted, open

**Established.**

1. `φ_p` is identified on this protocol and estimated well below one (median 0.51).
   That is the concrete answer to "what does φ buy": on the paper's own Gaussian data
   it is a free parameter that moves a long way from the value the binomial protocol
   would have imposed.
2. The reported component count is a function of the parameterisation, changing on
   4 of 10 slices, and of the grid, as the cost slice shows.

**Retracted.**

1. ~~"Freeing φ removes a component overall (4 → 3)."~~ The corrected comparison
   gives 4 → 5.
2. ~~"The policy dimension's minority component is unnecessary under free φ."~~
   Wrong: on the corrected comparison that component is **supported**, P = 2 under
   both parameterisations and both criteria, with φ̂ = [0.48, 0.54]. The earlier claim
   came entirely from the `s2` confound.
3. ~~"φ inflates the component count."~~ The free-φ selection is unstable, so no
   direction can be claimed from it.

**Open.** Whether a free `φ_p` is the right specification at this sample size. It
buys a large gain in the penalized objective (the nesting check shows gains of 57 to
668) while making the selection less resolvable; both facts are true and the paper
should say so rather than choosing whichever suits the case study.

## A separate concern this surfaced

**pBIC and an ordinary BIC disagree on most slices.** With φ free, pBIC selects a
smaller count than BIC in 8 of 10 slices (cost 2 against 4, driving experience 1
against 3, refuelling 2 against 4, other 2 against 4, and so on). The criteria differ
by the prior terms, and on this data the difference is large enough to change the
answer. Whatever count the paper reports, it should name the criterion and report the
other alongside; a reader who computes BIC on the same fit gets a larger number.

## Reproduce

```
uv run python experiments/case_study/_check_step9.py
uv run python experiments/case_study/_check_nesting.py
uv run python experiments/case_study/step9_phi_identification.py --all --grid 1 2 3 4 5
uv run python experiments/case_study/_phi_summary.py
uv run python experiments/case_study/_bootstrap_pbic.py --reps 10 --grid 3 4 5 6
uv run python experiments/case_study/_bootstrap_pbic.py --reps 12 --grid 3 4 5 6 --dim cost
```
