# Experiments

Every number in the paper comes from this directory. One command regenerates
all of them; each stage writes one artifact under `results/` and is named after
what it produces, so a reader who wants to check one table runs one stage.

```bash
python experiments/run_all.py --list          # the stages
python experiments/run_all.py                 # everything, about 30 minutes
python experiments/run_all.py --only holdout  # one table
python experiments/run_all.py --quick         # smoke test; numbers differ
```

The run needs the environment of `pyproject.toml` (`uv sync`). It ends by
checking every table cell against its artifact, re-running the three convention
audits in `tests/`, and running the test suite, so a green run means the
manuscript and the artifacts agree as well as the artifacts reproducing.

## What the paper reports, and where it comes from

Main text. `T#` and `F#` are the table and figure numbers of the manuscript.

| In the paper | Stage | Artifact |
|---|---|---|
| T1 recovery, five conditions | `validation/recovery.py` | `results/recovery.json` |
| F1 cost of pinning the dispersion | `validation/separability.py` | `results/separability.json` |
| the ratio `phi_hat/phi`, 0.93–1.14 | `validation/dispersion.py` | `results/dispersion.json` |
| the binomial protocol, 0.53 and -0.46 | `validation/binomial.py` | `results/binomial.json` |
| T2 four-component fit, BIC 18,339.8 | `case_study/fits.py` | `results/tables.json` |
| T3 the seven dimensions, K-hat and Delta-ell | `case_study/dim_curves.py` | `results/dim_curves.json` |
| T4 component-level estimates | `case_study/fits.py` | `results/tables.json` |
| F3 BIC against the number of components | `case_study/bic_curve.py` | `results/bic_curve.json` |
| bootstrap 51.0 / 28.0 / 13.0 / 8.0 | `case_study/bootstrap.py` | `results/bootstrap.json` |
| the count of 4 is a criterion's number | `case_study/dim_curves.py` | `results/dim_curves.json` |
| curvature -18 to -1,626 | `case_study/curvature.py` | `results/curvature.json` |
| 8 units of BIC, 18,347.8 pinned | `case_study/specification.py` | `results/specification.json` |
| -0.050 on the six named dimensions | `case_study/claims.py` | `results/claims.json` |

Online supplement.

| In the supplement | Stage | Artifact |
|---|---|---|
| S3 holdout, log score and CRPS | `case_study/holdout.py` | `results/holdout.json` |
| S4 retained gain, fold-level count | `case_study/holdout.py` | `results/holdout.json` |
| S6 the two yearly fits | `case_study/fits.py` | `results/tables.json` |
| 25 of 35 dimension-fold comparisons | `case_study/holdout.py` | `results/holdout.json` |
| the degenerate component (T2, S6) | `case_study/audit.py` | `results/audit.json` |

Two figures are drawn from these artifacts and compute nothing themselves: the
BIC curve reads `results/bic_curve.json` and the separability figure reads
`results/separability.json`. Both are typeset in the paper's source tree, which
is not part of this package.

`case_study/tables_check.py` reads the `.tex` files and the artifacts side by
side and compares 317 numeric cells at the precision the tables print, plus
relations between columns (the heaviest component, the holdout delta, and the
count of dimension-fold comparisons the mixture wins).

## Layout

```
config/       the hyperparameters, the corpus paths, the energy-type mapping
common/       paths, parameters, the reference EM, the P=1 model, metrics,
              synthetic data, replicate-level parallelism
validation/   Section 3.1: recovery, separability, dispersion, binomial
case_study/   Section 3.2: corpus, estimator, tables, bootstrap, holdout,
              specification, Laplace, curvature, audit, table check
results/      one JSON per stage; these are what the manuscript is checked against
```

Three rules the code follows, each of which was a bug before it was a rule:

* **One estimator.** Every case-study number goes through `case_study/estimator.py`
  with the priors of `config/mop_params.json` and a K-Ridge start from scratch at
  every candidate count. A chained start changes the selected count, because a
  small count's near-degenerate component is a bad seed for the next one.
* **Replicates are seeded individually**, `base + 1000003 * r`, so `--jobs 1` and
  `--jobs 14` produce the same artifact and a referee can rerun replicates 100–140.
* **Artifacts carry no wall-clock time.** Fits are timed and printed, never stored,
  which is what makes "byte for byte" true.

## When a number changes

Change it in the artifact's stage, rerun that stage, then run
`python experiments/case_study/tables_check.py`. If a table cell disagrees with
the artifact the check fails and names the cell; do not edit the artifact.

## Not in this package

The corpus starts at `case_study/energy_pairs.jsonl`: 5,401 cross-energy-type
judgments with their dimension labels, extracted from 2,328 items. The two stages
that produce it — the crawler and the classifier-plus-LLM extractor — are
separate projects with their own dependencies and are not imported by anything
here. `config/case_study.json` records their configuration and the classifier's
recorded evaluation, and `sections/AA_appendix.tex` in the manuscript reproduces
the prompts, so the extraction is documented even where it is not rerun.

## Cost

About half an hour on fourteen cores. `recovery` (500 replicates x 5
conditions) is 17 minutes of it and `separability` (1,000 x 8 cells) five;
`binomial` and `bootstrap` are a minute or two each, and every other stage is
under a minute. The table check, the three audits and the test suite add ten
seconds.
