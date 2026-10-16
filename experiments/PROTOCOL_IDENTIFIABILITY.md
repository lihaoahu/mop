# Protocol identifiability: an experiment that does not yet work

**Status: not usable. No conclusion may be drawn from it.**

The paper's revised thesis is that intrinsic variability and structural inconsistency
are separable *depending on the protocol*: identifiable for the Gaussian variants
(ratio, subjective probability), degenerate for the binomial ones (binary and repeated
choice) where Section 2.3 fixes `phi_p = 1`. That is a testable claim and this file was
meant to test it. It does not, yet, and the reason is worth recording.

## What was attempted

One dataset, both protocols. The same case-study judgments are read the Gaussian way
(the logit of the score) and the binomial way (the score as a choice with `M`
repetitions, observation weight `M`), and in each protocol a shared-dispersion model is
compared against a per-state-dispersion model at the same number of preference states.
Identifiability was to show up as what the extra `P - 1` dispersion parameters buy: a
lot where the dispersion is identified, almost nothing where it is not.

## Why it fails

**1. The two protocols' objectives are not on a common scale.** The Gaussian variant
fits `logit(r)` and the binomial variant fits counts. Their penalized objectives differ
by factors of order `M` and by different constants, so a gain of 32 in one is not
comparable with a gain of 21087 in the other. The measurement has no common unit, which
makes the comparison meaningless before any question of correctness arises.

**2. The binomial branch mis-encodes `phi`.** It enters `phi` as a multiplier on the
log-likelihood contribution. In the protocol the dispersion enters the *variance*,
`Var = phi * mu(1-mu) / omega`, so the multiplier is `1/phi`, not `phi` -- and scaling
the whole likelihood uniformly is exactly the operation that is not identified. The
branch therefore measured an artefact of its own parameterisation.

**3. `phi` is being estimated for the binomial protocol at all**, which is not what
Section 2.3 does. The paper fixes `phi_p = 1` there. A test of "is phi identified under
this protocol" is better posed as a property of the likelihood than as a free
parameter in a fit, and running it as a free parameter invited the encoding error above.

The observed numbers are consistent with all three defects: a Gaussian gain of 0.01 at
P = 2, and a binomial gain three orders of magnitude larger. Neither is a finding.

## What a correct version would need

1. **A common criterion across protocols.** Not the raw penalized objective. The
   dispersion's identifiability can be read off the profile likelihood in its own
   protocol -- how peaked that profile is, in units of its own curvature -- or from a
   simulation where the truth is known and the question is whether the estimator
   recovers it. Comparing two absolute objective values across protocols cannot work.

2. **`Var = phi * mu(1-mu) / omega` respected literally.** In the binomial protocol,
   changing `phi` and changing `omega` are the same operation, which IS the
   non-identifiability. The clean demonstration is to fit with `omega` fixed, then
   refit with `omega` scaled by a constant, and show the fit is unchanged apart from
   the absorbed constant. That is an identity that can be verified rather than a
   quantity that has to be compared.

3. **The Gaussian side already has its evidence** and does not need this script.
   `PHI_IDENTIFICATION.md` reports `phi-hat` estimated well below one on every slice
   and at every component count, under three priors, with the free-versus-fixed
   comparison passing a nesting identity to 0.000e+00. That is `phi` shown to be doing
   real work on the Gaussian protocol.

4. **The binomial half is an analytic statement**, not a fit: the variance is a
   function of the mean, so the dispersion is not separately identified. Section 2.3
   already says this. It needs a short proof, or the `omega`-absorption identity above,
   not a numerical contest between two objectives on different scales.

## Kept for the next attempt

The file is left in the repository with this note rather than deleted, because the
negative result constrains the design: any replacement must use a common unit across
protocols, and the binomial claim is an identity about `omega` rather than a free
`phi`. `protocol_identifiability.json` holds the numbers from the failed run and
should be regenerated, not cited.
