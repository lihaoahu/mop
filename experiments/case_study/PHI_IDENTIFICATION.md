# Does the structural inconsistency change the selected number of states?

Short answer: **yes, but not in one direction.** On the full case data the component
count falls from 4 to 3 when φ is freed; on the policy dimension it rises from 1 to
2; on cost and range it does not move. So the honest statement is not "φ inflates
the count" but "the count is not invariant to the parameterisation, and which way it
moves depends on the slice."

This matters because the case study's headline structural claim is a component
count, and that count turns out to be a function of a modelling choice.

## What was run

`step9_phi_identification.py` fits the case study's ratio judgments twice with the
paper's own Gaussian EM — once with `φ_p` estimated, once with `φ_p` fixed at one —
over the same grid of component counts. Both use the same `kridge`/K-Ridge
initialisation strategy, the same priors on `u` and `π`, and the same numerical
constants. The only difference is whether `φ_p` is a parameter.

The fixed-φ variant drops the Inverse-Gamma prior on `φ_p` and the `φ_p` dependence
of the normal prior's normalising constant, because a fixed value must not carry a
penalty it can no longer optimise against.

**The implementation was validated first.** `_check_step9.py` compares the
re-implemented free-φ fit against the repository's `inference.mop` on the same data:
pBIC agrees to within 0.4 at P = 1, 2, 3. So the comparison measures the
parameterisation, not a re-implementation difference.

## Results

Selection by the paper's pBIC:

| slice | n | φ fixed | φ free | moves? |
|---|---|---|---|---|
| overall | 5401 | **P = 4** | **P = 3** | down |
| cost | 1570 | P = 2 | P = 2 | no |
| range | 835 | P = 2 | P = 2 | no |
| policy | 281 | **P = 1** | **P = 2** | up |

Log-likelihood gain from freeing φ, at the fixed-φ optimum's component count:

| slice | gain | reading |
|---|---|---|
| overall | 48.1 | freeing φ explains spread the mixture was representing with components |
| cost | 22.7 | smaller |
| range | 31.6 | smaller |
| policy | **0.6** | negligible: φ frees almost nothing at the fixed-φ optimum |

That last column was intended to explain the pattern and instead kills the neat
story. The two slices whose selection moves have the largest and the smallest gain,
so "the selection moves when the fixed-φ fit is far from the free-φ fit" is false.
**I do not have an explanation for which way it moves, and the data say the pattern
is not a one-way bias.**

## The two substantive consequences

**1. The overall "four-component structure" is not invariant.** At P = 4 the two
parameterisations allocate mass very differently:

- φ fixed: π = [0.249, 0.0004, 0.386, 0.365] — the "four-component structure" the
  paper reports, with two large components of 0.39 and 0.37
- φ free: π = [0.444, 0.0004, 0.443, 0.113] — the fourth component collapses to 0.113

So the paper's four components are, on the free-φ parameterisation, closer to three
plus a remainder. The count is a reporting choice as much as a finding.

**2. Policy reverses.** The paper reports (03_empirical, the policy dimension) a
dominant component with π = 0.9006 ranking BEV first and a small component with
π = 0.0994 concentrating on REEV, and says that small component "carries the
aggregate" so that REEV leads on expected utility. That is a fixed-φ result:

- φ fixed: P = 2 selected, weights [0.925, 0.075] — this is the paper's finding
- φ free: **P = 1 selected**, weights at P = 2 would be [0.901, 0.099] with
  φ = [0.486, 0.568]

with free φ the estimated dispersions are well below one, so the data are *tighter*
than the φ = 1 model assumes. Once the model is allowed to know that, it no longer
needs a second component to explain the data, and the pBIC prefers a single state.

The estimated φ values are not degenerate: across all slices and component counts
they run from about 0.29 to 1.88. On this data φ is frequently below 1, which is
itself worth a sentence in the paper — the ratio responses are more consistent than
the nominal model implies, presumably because the scores come from an extraction
step rather than from independent human elicitation.

## What this does and does not establish

**Establishes:** the component count reported by this model is not invariant to
whether structural inconsistency is estimated. On the paper's own data, freeing φ
changes the answer on the full corpus and on one of the seven dimensions, and in the
policy dimension it removes the minority component that the paper's headline
substantive claim about that dimension rests on.

**Does not establish:** which parameterisation is correct, or a rule predicting
which way the count moves. The comparison is between two models, and the paper's
pBIC is defined for the free-φ one; the fixed-φ figures are produced under the same
criterion with the φ terms removed, which is the natural counterpart but is not the
paper's stated criterion.

## Why this is the paper's methodological point

The paper's framing is that intrinsic variability and structural inconsistency are
carried by separate parameters and can therefore be told apart. The result here is
the operational version of that claim: **telling them apart changes the number of
preference states the analyst reports**, which is the quantity the paper's decision
support is built on. That is a consequence, not a technicality — and the reverse
case on the policy dimension shows the consequence is not confined to one direction,
which is a further reason the choice has to be made explicit rather than left to the
default of φ ≡ 1.

## Reproduce

```
uv run python experiments/case_study/_check_step9.py
uv run python experiments/case_study/step9_phi_identification.py --grid 1 2 3 4 5 6
uv run python experiments/case_study/step9_phi_identification.py --dim policy --grid 1 2 3 4
uv run python experiments/case_study/_phi_summary.py
```
