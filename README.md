# MoP — Mixture of Preferences

Bayesian mixture model for discovering latent preference components from pairwise comparison data.

## Structure

```
src/mop/                      # Core library
    mixture_of_priorities.py  # Model definition (Priority, MoP parameters)
    inference.py              # EM algorithm, pBIC model selection
    pairwise_comparison.py    # Data representation, design matrix
    visualization.py          # Ternary plot, priority bar charts
    initialization.py         # Ridge-initialized warm-start seeding
    decision.py               # Decision-theoretic ranking extraction
    inference_binomial.py     # Binomial / beta-binomial likelihoods (repeated choices)

experiments/
    v1/                       # Case study: Chinese EV energy type preferences
        energy_pairs.jsonl    # 5,401 comparison pairs (LLM-extracted)
        mop_results.json      # MoP fit results (P=1..9)
        balance_results.json  # Vote-balance analysis
        step7_analyze.py      # MoP fitting + model selection
        step8_balance.py      # Balance-and-vote analysis
        _gen_tables.py        # LaTeX table generation
        _paper_dump.py        # Full result dump
        _plot_pbic.py         # pBIC vs P plot

    v2/                       # Method validation (simulation + holdout)
        step1_simulation.py   # Simulation: cancellation, general, noise
        step2_holdout.py      # Holdout CV, best-match (superseded by step5)
        step3_report.py       # Text + LaTeX report generation
        step4_revelatory.py   # Revelatory metrics (LL gain, P selection)
        step5_holdout.py      # Holdout CV, marginal log-likelihood (as reported)
        simulation_results.json
        holdout_results.json
        holdout_marginal_results.json
        revelatory_results.json

    v3/                       # External validation on a published dataset
        data/regenwetter2012.csv  # Regenwetter & Davis-Stober (2012), 30 x 3 units
        step1_regenwetter.py  # binomial vs beta-binomial likelihoods

tests/
    audit_parameterisation.py # Verifies every distribution convention in use
```

## Parameterisation

Distribution parameters differ across libraries (`scale` is a standard deviation
in `scipy.stats.norm`, "Gamma" may be shape-rate or shape-scale elsewhere).
Every convention this project relies on is documented in
[`PARAMETERISATION.md`](PARAMETERISATION.md) and verified numerically by
`tests/audit_parameterisation.py`:

## Requirements

Python 3.11+, dependencies in `pyproject.toml`. Install with:

```bash
uv sync
```

## Reproducing results

### Case study (Section 3)

```bash
uv run python experiments/case_study/step7_analyze.py      # MoP fitting
uv run python experiments/case_study/step8_balance.py      # Balance analysis
uv run python experiments/case_study/_gen_tables.py        # Generate LaTeX tables
```

### Validation experiments (Section 3.x)

```bash
uv run python experiments/simulation/step1_simulation.py   # Simulation
uv run python experiments/simulation/step2_holdout.py      # Holdout CV (best-match, historical)
uv run python experiments/simulation/step5_holdout.py      # Holdout CV (marginal, as reported)
uv run python experiments/simulation/step4_revelatory.py   # Revelatory metrics
uv run python experiments/simulation/step3_report.py       # Report
```

### External validation (v3)

```bash
uv run python experiments/external/step1_regenwetter.py  # binomial vs beta-binomial
```

### Self-checks

```bash
uv run python tests/audit_parameterisation.py      # Parameterisation conventions
```
