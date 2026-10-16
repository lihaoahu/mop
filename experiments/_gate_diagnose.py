"""Run the parameter gate and, if it fails, preserve everything needed to explain why.

The gate's third check runs `step7_analyze.py` as a subprocess and compares its output to
`analysis_output.txt`. It has failed intermittently -- three times across many runs, never
reproducible in isolation, and step7 is deterministic when run on its own (two consecutive
runs byte-identical). Rather than guess at the cause, this wrapper records enough to settle
it next time:

  * the gate's stdout and stderr, separately and unbuffered;
  * step7's stdout, stderr and exit code from an independent run;
  * sha256 of `analysis_output.txt` and of the normalised step7 output;
  * the first differing line, if any.

Artifacts go to `_scratch/gate_failure_<timestamp>/` and the path is printed.

Run:
    uv run python experiments/_gate_diagnose.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
REVISED = EXPERIMENTS.parent
CASE = EXPERIMENTS / "case_study"
RECORDED = CASE / "analysis_output.txt"


def norm(text):
    """The comparison the gate uses: stripped, non-empty lines."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    scratch = REVISED.parent / "_scratch"
    env = {**os.environ, "PYTHONPATH": str(REVISED / "src")}

    # 1. the gate itself
    gate = subprocess.run([sys.executable, "-u", str(EXPERIMENTS / "_mop_params_check.py")],
                          capture_output=True, text=True, env=env)

    # 2. step7 alone
    s7 = subprocess.run([sys.executable, "-u", str(CASE / "step7_analyze.py")],
                        cwd=str(CASE), capture_output=True, text=True, env=env)

    recorded_text = RECORDED.read_text(encoding="utf-8-sig")
    want, got = norm(recorded_text), norm(s7.stdout)
    first_diff = None
    for i in range(max(len(want), len(got))):
        w = want[i] if i < len(want) else "<absent>"
        g = got[i] if i < len(got) else "<absent>"
        if w != g:
            first_diff = (i, w, g)
            break

    report = [
        f"gate exit code      : {gate.returncode}",
        f"step7 exit code     : {s7.returncode}",
        f"recorded lines      : {len(want)}  sha {sha(chr(10).join(want))}",
        f"step7 lines         : {len(got)}  sha {sha(chr(10).join(got))}",
        f"identical           : {want == got}",
        f"first difference    : {first_diff}",
    ]

    # Artifacts only when something actually failed. A passing run writing a directory
    # every time just fills the scratch tree with twelve copies of the same result, which
    # is what the first version of this script did.
    if gate.returncode != 0 or want != got:
        out = scratch / f"gate_failure_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate.stdout").write_text(gate.stdout, encoding="utf-8")
        (out / "gate.stderr").write_text(gate.stderr, encoding="utf-8")
        (out / "step7.stdout").write_text(s7.stdout, encoding="utf-8")
        (out / "step7.stderr").write_text(s7.stderr, encoding="utf-8")
        (out / "recorded.copy.txt").write_text(recorded_text, encoding="utf-8")
        report += ["", "gate stderr tail:", *gate.stderr.splitlines()[-8:],
                   "", "step7 stderr tail:", *s7.stderr.splitlines()[-8:],
                   f"", f"artifacts: {out}"]

    print("\n".join(report))
    return 0 if (gate.returncode == 0 and want == got) else 1


if __name__ == "__main__":
    raise SystemExit(main())
