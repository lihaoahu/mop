# Mixture of Preferences

Code for *Separating Intrinsic Variability from Structural Inconsistency in Pairwise
Comparisons: Identification and a Scalable Mixture-of-Preferences Estimator*.

<https://github.com/lihaoahu/mop>

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
```

## Install

```bash
git clone https://github.com/lihaoahu/mop
cd mop
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

Each stage writes one JSON under `experiments/results/`. Those files are what the paper
reports from: every table cell and every number in the text is read off one of them.
Replicates draw their own seeds, so the results do not depend on the number of workers.

## Data

`experiments/case_study/energy_pairs.jsonl` holds the corpus: 2,328 articles and videos
carrying 6,706 extracted comparisons, which the loader reduces to the 5,401
cross-energy-type judgments on four vehicle energy types that the paper fits. Each line
records one item's identity and, under `comparison_pairs`, the judgments drawn from it:
the two alternatives, the direction, the judged probability, the evaluation dimension and
a verbatim citation. The crawled page text and the comments are not redistributed. The
crawler and the extraction prompts are documented in the paper's supplement; the
extraction itself needs the LLM API and is not part of this package.

## Citation

Cite the paper, not the code.

```bibtex
@article{li2026separating,
  title   = {Separating Intrinsic Variability from Structural Inconsistency in Pairwise
             Comparisons: Identification and a Scalable Mixture-of-Preferences Estimator},
  author  = {Li, Hao and Zheng, Pan and Wu, Jie and Zhou, Ligang},
  journal = {INFORMS Journal on Computing},
  year    = {2026},
  note    = {Manuscript submitted for review}
}
```

## Licence

MIT, see `LICENSE`.
