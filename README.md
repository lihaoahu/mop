# MoP — Mixture of Preferences

Bayesian mixture model for discovering latent preference components from
pairwise comparison data.

## Structure

```
src/mop/                      # Core library
    mixture_of_priorities.py  # Model definition, Priority, expected_utility
    inference.py              # EM algorithm, pBIC model selection
    pairwise_comparison.py    # Data representation, design matrix
    initialization.py         # K-Ridge seeding, and the GLM/IRLS variant
    decision.py               # Decision-theoretic ranking extraction
    visualization.py          # Ternary plot, priority bar charts

experiments/
    case_study/               # Section 3.2: Chinese EV energy type preferences
        energy_pairs.jsonl    # 5,401 comparison pairs (LLM-extracted)
        mop_results.json      # MoP fit for the overall dataset
        balance_pooled_results.json  # dimension-balanced bootstrap
        step7_analyze.py      # MoP fitting + model selection, all subsets
        step8_balance.py      # dimension-balanced bootstrap
        _gen_tables.py        # LaTeX for Tables 2-4 and 6
        _plot_pbic.py         # pBIC against P (figure)
        _plot_dim_distribution.py    # dimension share donut (figure)
        _paper_dump.py        # full result dump, for inspection

    simulation/               # Section 3.1.1: controlled simulations
        step1_simulation.py   # cancellation, general, noise scenarios
        step2_holdout.py      # 5-fold CV, point-prediction metrics
        step3_report.py       # text summary of the above
        step4_revelatory.py   # log-likelihood gain, component count
        step5_holdout.py      # 5-fold CV, marginal log-likelihood (as reported)

tests/
    audit_parameterisation.py # verifies every distribution convention in use
```

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

### Case study (Section 3.2)

```bash
uv run python experiments/case_study/step7_analyze.py       # fits, all subsets
uv run python experiments/case_study/step8_balance.py --n-bootstrap 100
uv run python experiments/case_study/_plot_pbic.py          # pBIC figure
uv run python experiments/case_study/_plot_dim_distribution.py
```

`step7_analyze.py` prints the tables: for every fit — overall, the seven
dimensions and both years — it reports each component's mixture weight, its
inconsistency level and its priority vector, then the aggregate. It is the only
source of Tables 5, 6 and 8, so the values are read off its output rather than
computed a second time.

`step8_balance.py` takes about nine minutes at 100 replications, and its
`--n-bootstrap` default is 50: the paper reports 100, so the flag is not
optional. The two plotting scripts read the same fits and write into
`manuscript/figures/`.

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
