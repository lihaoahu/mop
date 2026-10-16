# Recovering the structural inconsistency on synthetic data

The abstract says MoP "recovers latent structure, isolates noise, and separates the two
sources where a single dispersion parameter cannot". The first two are covered by
`experiments/simulation`. The third was not: that simulation fixes `phi = 0.05`
throughout (`simulation/config.json` holds one value) and never varies or tests the
dispersion, so the claim had no experiment behind it.

This is the experiment. Synthetic Gaussian data with a known `phi`, two true preference
states with opposite orderings, `A = 4` alternatives, `n = 2000` comparisons, six
replicates per setting. `phi_recovery.py`; results in `phi_recovery.json`.

## 1. The dispersion estimator is good

| `phi` true | 0.25 | 0.50 | 1.00 | 2.00 | 4.00 |
|---|---|---|---|---|---|
| `phi-hat` mean | 0.287 | 0.530 | 1.035 | 1.880 | 3.223 |
| sd | 0.008 | 0.037 | 0.062 | 0.065 | 0.452 |
| ratio to truth | 1.147 | 1.060 | 1.035 | 0.940 | 0.806 |

Across a sixteen-fold range the estimate tracks the truth, with a mild downward bias
that grows with `phi` (ratio 0.81 at `phi = 4`). **So `phi` is not merely identified in
principle — it is recovered, and this is the estimate the case study's 0.34–0.65 values
should be read against.** A downward bias of this size means those case-study values
may understate the true dispersion by roughly 5–15%, which is worth a sentence rather
than a correction.

## 2. The component count is not robust to the dispersion, even on Gaussian data

P = 2 is the true count. Selected count, by replicate:

| `phi` true | P = 2 chosen, φ fixed | P = 2 chosen, φ free |
|---|---|---|
| 0.25 | 6/6 | 6/6 |
| 0.50 | 6/6 | 6/6 |
| 1.00 | 6/6 | 6/6 |
| 2.00 | **0/6** | 4/6 |
| 4.00 | **0/6** | **0/6** |

**This is the finding, and it is the opposite of what the abstract implies.** As the
dispersion grows, the correct number of preference states stops being selected: at
`phi = 2` the fixed-`phi` criterion never finds both states, and at `phi = 4` neither
parameterisation does.

The mechanism is the one Section 2.5 already describes for the binomial protocol —
"one preference executed inconsistently and several preferences executed consistently
are two explanations of the same data" — except here it is happening in the **Gaussian**
protocol, which is the variant the paper says avoids the problem. A mixture is
overdispersed relative to a single component, so a large dispersion can be absorbed by
splitting or merging components, and pBIC responds to that trade-off regardless of
which variance function is in play.

**What this does and does not say.** It does not overturn the paper's separability
claim: `phi` is identified and recovered on this protocol, which is what
`protocol_identifiability.py` establishes and what the table above confirms. It does
mean the *count* of states is not a reliable read-out when the dispersion is large, and
that the paper cannot claim the mixture "isolates" inconsistency in the sense of
protecting the component count. The two claims are different and only the first is
supported.

## 3. What to do about the abstract

The sentence currently asserts an experiment that was never run and a result that this
one contradicts. Three options:

* **Narrow it** to what the simulation actually shows (recovers structure, isolates
  noise into a dedicated component, selects the correct count when the dispersion is
  moderate).
* **Replace it** with the identifiability result — `phi` is recovered on ratio
  judgments with a ratio to truth of 0.94–1.15 for `phi <= 2`, and is not identified at
  all for binary and repeated choices — which is a real, measured claim and is the
  paper's new thesis anyway.
* **Add the caveat** as a limitation, keeping the sentence.

The second is what the evidence supports and it is also the sharper claim. The first is
the smallest edit.

## Reproduce

```
uv run python experiments/phi_recovery.py --reps 6 --pairs 2000 --phi 0.25 0.5 1.0 2.0 4.0
```
