# MoP — Mixture of Preferences

Bayesian mixture model for discovering latent preference components from
pairwise comparison data.

## Structure

```
src/mop/                      # Core library
    mixture_of_priorities.py  # Model definition, Priority, expected_utility
    inference.py              # closed-form EM for the Gaussian variants, pBIC
    general_fit.py            # direct numerical MAP, for any observation family
    winloss.py                # binomial family: EM with a numerically solved M-step
    pairwise_comparison.py    # Data representation, design matrix
    initialization.py         # K-Ridge seeding, and the GLM/IRLS variant
    decision.py               # Decision-theoretic ranking extraction
    visualization.py          # Ternary plot, priority bar charts

experiments/
    case_study/               # Section 3.2: Chinese EV energy type preferences
        energy_pairs.jsonl    # the filtered corpus: 2,328 items (1,925 text, 403 video)
        mop_results.json      # SUPERSEDED: the 6-11 artefact, lambda_u=0.1, lambda_pi=30,
                              # n=5408; no code reads it (_common and step7_analyze
                              # both import the path constant and neither uses it)
        step7_analyze.py      # MoP fitting + model selection, all subsets
        _gen_tables.py        # LaTeX for Tables 3, 5, 6 and 8
        _plot_pbic.py         # pBIC against P (figure)
        _plot_dim_distribution.py    # dimension share donut (figure)
        _paper_dump.py        # full result dump, for inspection

    simulation/               # Section 3.1.1: controlled simulations
        step1_simulation.py   # cancellation, general, noise scenarios
        step2_holdout.py      # SUPERSEDED: oracle selection, see the note below
        step3_report.py       # text summary of the above
        step4_revelatory.py   # log-likelihood gain, component count
        step5_holdout.py      # 5-fold CV, marginal log-likelihood (as reported)

tests/
    audit_parameterisation.py # verifies every distribution convention in use
    check_general_fit.py      # numerical route vs closed-form EM, on both families
```

`step2_holdout.py` writes `holdout_results.json`. Its MoP mean absolute error is
0.0764 against its own single-component baseline of 0.2456; the paper reports
0.2462 against 0.2456, from `step5_holdout.py`. Those are not two estimates of
one quantity: step2 scores a held-out comparison with whichever component
predicts it best, which reads the label it is supposed to predict, so its MoP
column is an oracle selection rather than a forecast and its apparent advantage
is the oracle at work (step5's own docstring says so). Recomputing the paper's
holdout numbers from step2's file gives different values, which is why the file
is marked rather than merely unused.

The dimension-balanced bootstrap is gone, together with its script and its two result
files. The manuscript stopped reporting it: the controlled simulations establish
recovery, so a resampling stability check of the case-study count answered a question
the simulations already answer. One file here is still kept for the record and read by
nothing -- `mop_results.json`, the 6-11 artefact described above -- and
`regenwetter_newinit.txt` at the repository root is the log of an external validation
removed in 73466ff.

## Parameterisation

Distribution parameters differ across libraries (`scale` is a standard
deviation in `scipy.stats.norm`, "Gamma" may be shape-rate or shape-scale
elsewhere). Every convention this project relies on is documented in
[`PARAMETERISATION.md`](PARAMETERISATION.md) and verified numerically by
`tests/audit_parameterisation.py`, which exits non-zero if any of them fails to
reproduce.

## Requirements

Python 3.10+, dependencies in `pyproject.toml`. Install with:

```bash
uv sync
```

The `collection` extra carries the crawling and extraction dependencies needed
only to rebuild the corpus from the platform; the analysis scripts do not
import them, so the default environment is small.

## Reproducing results

Every committed result file and printed table reproduces byte for byte from the
scripts below; wall-clock timings are printed but deliberately not stored, which
is what makes that true.

One exception is worth knowing before re-capturing it.
`experiments/case_study/analysis_output.txt` is the capture of
`step7_analyze.py`'s stdout, and how it is captured matters: the redirect has to
pass the bytes through. With `PYTHONIOENCODING=utf-8` and a shell redirect that
does not re-encode, the file comes back identical (4019 bytes, verified with
`cmd /c "... > analysis_output.txt"`); PowerShell's `>` decodes the child's
output and writes UTF-16, so the numbers are the same but the file is not.

### Case study (Section 3.2)

```bash
uv run python experiments/case_study/step7_analyze.py       # fits, all subsets
uv run python experiments/case_study/_plot_pbic.py          # pBIC figure
uv run python experiments/case_study/_plot_dim_distribution.py
uv run python experiments/case_study/_gen_tables.py --json case_study_tables.json
```

`step7_analyze.py` prints the tables: for every fit — overall, the seven
dimensions and both years — it reports each component's mixture weight, its
inconsistency level and its priority vector, then the aggregate. It is the
source of Tables 5, 6 and 8, so the values are read off its output rather than
computed a second time; the committed `analysis_output.txt` is that capture.
`_gen_tables.py` emits the same fits as LaTeX (Tables 3, 5, 6 and 8), reading
`run_mop` from `step7_analyze.py`, and caches them in `case_study_tables.json`.

The two plotting scripts read the same fits and write into `manuscript/figures/`.

### Controlled simulations (Section 3.1.1)

```bash
uv run python experiments/simulation/step1_simulation.py --trials 500
uv run python experiments/simulation/step2_holdout.py
uv run python experiments/simulation/step5_holdout.py       # as reported
uv run python experiments/simulation/step4_revelatory.py
uv run python experiments/simulation/step3_report.py
```

The 500-trial simulation runs for roughly 35 minutes at the stopping tolerance
in use. `step2_holdout.py` reports point-prediction metrics and is not the
source of the holdout table, which comes from `step5_holdout.py`.

### Self-checks

```bash
uv run python tests/audit_parameterisation.py
```
