# Mixture of Preferences

Code for *Separating Intrinsic Variability from Structural Inconsistency in Pairwise
Comparisons: Identification and a Scalable Mixture-of-Preferences Estimator*.

The model estimates, from pairwise comparisons, how many preference states a population
occupies, what each one ranks, and how inconsistently each is executed. Intrinsic
variability enters through the mixture, structural inconsistency through a
component-specific dispersion. The dispersion is identified only where the variance
function leaves it free, which is why the same code fits ratio judgments, subjective
probability judgments and repeated choices under one selection criterion.

## Layout

```
src/mop/                   the model
    pairwise_comparison.py     judgments, design matrix
    mixture_of_priorities.py   components, expected utility
    inference.py               closed-form EM, Gaussian variants
    winloss.py                 binomial family
    general_fit.py             direct MAP, any observation family
    criteria.py                BIC and the profiled variant
    initialization.py          K-Ridge start
    laplace.py                 analytic curvature, Newton polish

experiments/               one stage per artifact; see experiments/README.md
    validation/                simulation studies (Section 3.1)
    case_study/                the application (Section 3.2)
    common/                    shared fitting, generation, metrics, parallelism
    results/                   the JSON each stage writes
tests/                     criterion identities, scoring rules, cross-checks
PARAMETERISATION.md        distribution conventions, and how they are verified
```

## Install

```bash
uv sync                      # or: pip install -e .
```

Python 3.10 or newer; numpy, scipy and scikit-learn are the only runtime dependencies.

## Reproduce

```bash
python experiments/run_all.py --list         # the stages
python experiments/run_all.py                # all of them, about 30 minutes
python experiments/run_all.py --only holdout # one stage
python experiments/run_all.py --quick        # smoke test, numbers differ
```

Each stage writes one JSON under `experiments/results/`. The run ends by comparing
every printed table cell against its artifact and by running the test suite, so a green
run means the manuscript and the artifacts agree. Replicates draw their own seeds, so the
results do not depend on the number of workers.

## Tests

```bash
python -m pytest tests -q                # 16 tests
python tests/audit_parameterisation.py   # distribution conventions
python tests/check_general_fit.py        # numerical MAP against the closed-form EM
python tests/audit_laplace.py            # analytic curvature against finite differences
```

## Data

`experiments/case_study/energy_pairs.jsonl` holds the corpus: 5,401 cross-energy-type
judgments on four vehicle energy types, extracted from platform discourse on Dongchedi.
Each line records the two alternatives, the direction, the judged probability, the
evaluation dimension, the article it came from and a verbatim citation. The crawler and
the extraction prompts are documented in the paper's supplement; the extraction itself
needs the LLM API and is not part of this package.

## Licence

MIT, see `LICENSE`.
