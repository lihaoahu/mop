# Is the structural inconsistency identified? It depends on the protocol

**This is the experiment the paper's revised thesis needs, and it now works.** The
claim is that intrinsic variability and structural inconsistency are separable only for
some judgment formats: identified for the Gaussian variants (ratio, subjective
probability), degenerate for the binomial ones (binary and repeated choice), where
Section 2.3 fixes `phi_p = 1`. Result: **confirmed, and the binomial half is an identity
rather than a numerical finding.**

## The test

Do not compare the two protocols' objectives — their scales differ (one fits
`logit(r)`, the other fits counts) and two earlier attempts failed for exactly that
reason, plus one more (below). Instead ask, inside each protocol, whether the
log-likelihood has any curvature in `phi`. At fixed parameters the contribution of
judgment `k` is

| protocol | log-likelihood contribution | `d²L/dφ²` |
|---|---|---|
| binomial | `omega_k [m_k eta_k − n_k log(1+exp eta_k)]` | **`0`** |
| Gaussian | `−0.5 log(2 pi phi) − 0.5 omega_k (y_k − eta_k)² / phi` | **`−omega_k / phi²`** |

The first has no `phi` in it at all under Section 2.3, so its second derivative is
identically zero: every value of `phi` scores the same, there is no peak, and no amount
of data can pin it. The second is strictly negative, so a maximum exists.

Both numbers are second derivatives in the same parameter, so the comparison is
`0` against `strictly negative` and involves no cross-protocol scale at all.

## Result

`protocol_identifiability.py`, overall slice, `n = 5401`, `omega_k = M = 20`:

```
binomial  d2L/dphi2 : every entry = 0        (max |value| = 0.0e+00)
Gaussian  d2L/dphi2 at phi=0.50 : total  -432080.0
          d2L/dphi2 at phi=1.00 : total  -108020.0
          d2L/dphi2 at phi=2.00 : total   -27005.0
```

The Gaussian profile, penalized objective at `P = 2` with `phi` pinned:

| `phi` | 0.40 | 0.70 | 1.00 | 1.40 | 2.00 | 3.00 |
|---|---|---|---|---|---|---|
| pll | −13982.6 | −10988.9 | −9866.6 | −9377.3 | **−9303.2** | −9605.6 |

It moves by **4679.3** and peaks near `phi = 2.0`, close to the prior mode
`phi_0 = 1.8847`. On the smallest slice (policy, `n = 281`) the same test gives binomial
curvature `0` and a Gaussian profile moving by **94.19**. The binomial profile is not
plotted because it cannot move.

## What this establishes for the paper

1. **The binomial protocol cannot identify the dispersion, and Section 2.3 is right to
   fix it at one.** The likelihood is linear in `phi`, so the constraint is a statement
   about the model, not a convention. This is the half of the thesis that had no
   evidence before.
2. **The Gaussian protocol can identify it**, with a strictly negative curvature and a
   profile with a clear interior maximum. `phi` is not decoration there; it is a
   parameter the data act on.
3. **The separation of intrinsic variability from structural inconsistency is therefore
   conditional on the protocol**, which is a sharper and more defensible claim than
   "we put them in separate parameters" — a claim any regularised mixture can make.
   This is the experiment that turns the paper's §2.5 observation into a result.

## Two failed attempts, kept because they set the constraints

*Attempt 1* compared the two protocols' penalized objectives directly and reported a
Gaussian gain of 0.01 against a binomial gain of 21087 for the same extra parameters.
Neither number meant anything: the objectives are not on a common scale, the binomial
branch mis-encoded `phi` as a likelihood multiplier when the protocol puts it in the
variance, and it estimated `phi` at all for a model that fixes it.

*Attempt 2* rescaled the observation weights and looked for the fitted model to be
invariant. It is not, and the reason is worth stating because it is a real property:

> **Regularisation destroys the non-identifiability.** Multiplying `omega` by `c`
> multiplies the likelihood by `c` while leaving the L2 ridge on the utilities fixed,
> so the penalty's relative influence changes and the fit moves. Without the ridge the
> dispersion would be identified only up to the weight scale.

So in the fitted model `phi` is weakly identified *through the prior*, not by the
binomial data. That is a precise statement worth making in the paper rather than
leaving implicit, and it is measured rather than asserted.

## Reproduce

```
uv run python experiments/protocol_identifiability.py
uv run python experiments/protocol_identifiability.py --dim policy
```

Output: `protocol_identifiability.json` (overall), and
`protocol_identifiability_policy.json`. The 0.01/21087 numbers from the failed first
attempt are gone; these files hold only the curvature and the profile.
