"""One command that runs the acceptance checks for the Gaussian fit.

Nothing from `step9_phi_identification.py` should be quoted unless all of these pass. They
are ordered from cheapest to most expensive so a failure stops early.

  1. parameter audit        every value from the config; the two objectives agree
                            exactly at phi = 1
  2. nesting                the free fit must reach a penalized objective at least
                            as high as the fixed one, from a common start
  3. implementation         the free-phi branch must reproduce mop.inference

A fourth and fifth check used to run here -- the dispersion's robustness under alternative
priors, and the stability of the selected component count under resampling. Both were part
of the phi comparison; that analysis is finished and its conclusions are in the manuscript
and in PHI_IDENTIFICATION.md, so the scripts that produced them were removed rather than
left to be rerun for no reason. The resampling study was removed from the manuscript
as well: the controlled simulations establish recovery, so a resampling stability check
of the case-study count answered a question the simulations already answer.

Run: uv run python experiments/case_study/_validate_phi.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("parameter audit", ["_audit_params.py"]),
    ("nesting", ["_check_nesting.py"]),
    ("implementation vs mop.inference", ["_check_step9.py"]),
]

def run(script, extra=()):
    cmd = [PY, "-u", str(HERE / script), *extra]
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def tail(text, n=8):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def main():
    results = []
    for name, argv in STEPS:
        code, out = run(argv[0], argv[1:])
        # `_audit_params.py` returns 1 when the identity fails, which is the signal.
        ok = ("PASS" in out and "FAIL" not in out) if name == "parameter audit" else (
            code == 0)
        if name == "nesting":
            ok = "NO" not in out and "Traceback" not in out
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(tail(out, 20))
        results.append((name, ok))

    print()
    print("=" * 70)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 70)
    bad = [n for n, ok in results if not ok]
    if bad:
        print(f"\nNot acceptable until these pass: {', '.join(bad)}")
    else:
        print("\nAll acceptance checks pass. The parameter set is in "
              "experiments/mop_params.json, and the paper-level gate is "
              "experiments/_mop_params_check.py.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
