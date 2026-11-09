"""Regenerate every number in the paper."""


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
    """One script, the artifact it writes, and the paper object it is for."""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None,
                    help="run only these stages (repeatable)")
    ap.add_argument("--jobs", default=None,
                    help="worker processes; results do not depend on it")
    ap.add_argument("--quick", action="store_true",
                    help="few replicates; a smoke test, not the paper")
    ap.add_argument("--list", action="store_true", help="list the stages")
    args = ap.parse_args()

    if args.list:
        for s in STAGES:
            print(f"  {s.key:<14} {s.gives}")
        return 0

    stages = STAGES
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s.key for s in STAGES}
        if unknown:
            print(f"unknown stage(s): {', '.join(sorted(unknown))}")
            return 2
        stages = [s for s in STAGES if s.key in wanted]

    RESULTS.mkdir(exist_ok=True)
    failed = []
    for stage in stages:
        if run_stage(stage, args.jobs, args.quick) != 0:
            failed.append(stage.key)

    if failed:
        print("\nFAILED: " + ", ".join(failed))
        return 1
    print("\nall stages completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
