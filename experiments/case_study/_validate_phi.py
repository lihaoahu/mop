"""One command that runs the acceptance checks for the Gaussian fit."""
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
