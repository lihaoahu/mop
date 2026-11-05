"""Regenerate every number in the paper, then check it."""


from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = HERE / "results"


class Stage:
    """One script, the artifact it writes, and the paper object it is for.

    ``script`` is relative to the repository root, so the checks that live in
    ``tests/`` are listed the same way as the experiment stages.
    """

    def __init__(self, key, script, gives, quick=(), jobs=False):
        self.key = key
        self.script = PROJECT / script
        self.gives = gives
        self.quick = list(quick)
        self.jobs = jobs


STAGES = [
    Stage("recovery", "experiments/validation/recovery.py",
          "Table 1  recovery over 500 replicates per condition",
          quick=["--trials", "20"], jobs=True),
    Stage("separability", "experiments/validation/separability.py",
          "Figure 1  what pinning the dispersion costs",
          quick=["--reps", "10"], jobs=True),
    Stage("dispersion", "experiments/validation/dispersion.py",
          "the recovered-to-true dispersion ratio",
          quick=["--reps", "1", "--pairs", "500"]),
    Stage("binomial", "experiments/validation/binomial.py",
          "the binomial protocol table",
          quick=["--reps", "5"]),

    Stage("fits", "experiments/case_study/fits.py",
          "Tables 2 and 4, and the yearly table"),
    Stage("dim_curves", "experiments/case_study/dim_curves.py",
          "Table 3, and the BIC curve of every dimension"),
    Stage("bic_curve", "experiments/case_study/bic_curve.py",
          "Figure 3  BIC against the number of components"),
    Stage("bootstrap", "experiments/case_study/bootstrap.py",
          "the article-level bootstrap of the selected count",
          quick=["--reps", "10"], jobs=True),
    Stage("holdout", "experiments/case_study/holdout.py",
          "the holdout tables and the retained-gain column"),
    Stage("specification", "experiments/case_study/specification.py",
          "free versus pinned dispersion"),
    Stage("curvature", "experiments/case_study/curvature.py",
          "the curvature that identifies the dispersion"),
    Stage("claims", "experiments/case_study/claims.py",
          "the refit on the six named dimensions"),
    Stage("audit", "experiments/case_study/audit.py",
          "what the low-weight component is made of"),
]

CHECKS = [
    Stage("tables", "experiments/case_study/tables_check.py",
          "every printed table cell against its artifact"),
    Stage("conventions", "tests/audit_parameterisation.py",
          "every distribution convention, verified numerically"),
    Stage("estimator", "tests/check_general_fit.py",
          "the numerical MAP against the closed-form EM"),
    Stage("curvature_fd", "tests/check_curvature_fd.py",
          "the proposition's second derivative, finite differences"),
]


def _env():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_stage(stage, jobs=None, quick=False):
    cmd = [sys.executable, str(stage.script)]
    if quick:
        cmd += stage.quick
    if jobs and stage.jobs:
        cmd += ["--jobs", str(jobs)]
    print(f"\n=== {stage.key}: {stage.gives}", flush=True)
    print("    " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PROJECT), env=_env())
    print(f"    {stage.key} finished in {time.perf_counter() - t0:.0f}s",
          flush=True)
    return proc.returncode


def run_tests():
    print("\n=== tests: the conventions, the criterion, the scoring rules", flush=True)
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    proc = subprocess.run(cmd, cwd=str(PROJECT), env=_env())
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None,
                    help="run only these stages (repeatable)")
    ap.add_argument("--jobs", default=None,
                    help="worker processes; results do not depend on it")
    ap.add_argument("--quick", action="store_true",
                    help="few replicates; a smoke test, not the paper")
    ap.add_argument("--skip-checks", action="store_true",
                    help="skip the table check and the test suite")
    ap.add_argument("--list", action="store_true", help="list the stages")
    args = ap.parse_args()

    if args.list:
        for s in STAGES + CHECKS:
            print(f"  {s.key:<14} {s.gives}")
        return 0

    stages = STAGES
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s.key for s in STAGES + CHECKS}
        if unknown:
            print(f"unknown stage(s): {', '.join(sorted(unknown))}")
            return 2
        stages = [s for s in STAGES if s.key in wanted]

    RESULTS.mkdir(exist_ok=True)
    failed = []
    for stage in stages:
        if run_stage(stage, args.jobs, args.quick) != 0:
            failed.append(stage.key)

    if not args.skip_checks and not args.only:
        for check in CHECKS:
            if run_stage(check) != 0:
                failed.append(check.key)
        if run_tests() != 0:
            failed.append("tests")

    if failed:
        print("\nFAILED: " + ", ".join(failed))
        return 1
    print("\nall stages completed" + ("" if args.only else "; tables and tests pass"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
