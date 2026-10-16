"""One command that runs every acceptance check for the phi comparison.

Nothing from `step9_phi_identification.py` should be quoted unless all of these
pass. They are ordered from cheapest to most expensive so a failure stops early.

  1. parameter audit        every value from the config; the two objectives agree
                            exactly at phi = 1
  2. nesting                the free fit must reach a penalized objective at least
                            as high as the fixed one, from a common start
  3. implementation         the free-phi branch must reproduce mop.inference
  4. phi-hat robustness     the dispersion estimates must keep the same shape under
                            every candidate prior mode, since they are the finding
                            that does not depend on a criterion
  5. bootstrap              the selected component count must be stable under
                            resampling, or reported as unstable

Run: uv run python experiments/case_study/_validate_phi.py [--skip-bootstrap]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

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

    if not args.skip_bootstrap:
        code, out = run("_bootstrap_pbic.py", ["--reps", "10", "--grid", "3", "4", "5", "6"])
        ok = code == 0 and "frequencies" in out
        print(f"[{'PASS' if ok else 'FAIL'}] bootstrap stability")
        if ok:
            print(tail(out, 8))
        else:
            print(tail(out, 20))
        results.append(("bootstrap stability", ok))

    print()
    print("=" * 70)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 70)
    bad = [n for n, ok in results if not ok]
    if bad:
        print(f"\nNot quotable until these pass: {', '.join(bad)}")
    else:
        print("\nAll acceptance checks pass. Results may be quoted, with the "
              "parameter set recorded in phi_config.json.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
