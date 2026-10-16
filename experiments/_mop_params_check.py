"""Verify that the externalised MoP parameters still reproduce the paper's output.

The failure this guards against is not hypothetical. `lambda_sigma` was first written
into `mop_params.json` as `0.05 * nJ` when the paper's rule is `nJ / 50`, i.e.
`0.02 * nJ`; that is a prior differing by a factor of 2.5, and it moved the case
study's selected component count from the recorded 4 to 3. Nothing in the code
complained, because a wrong prior is still a valid prior. The only thing that caught
it was comparing against the recorded output, so that comparison is now a script.

Three checks, cheapest first:

1. **The coefficient is 1/50.** An arithmetic identity, not a fitted quantity.
2. **The configs agree.** `mop_params.json` is the single source of truth;
   `phi_config.json` carries a copy for the reader's convenience, and a copy that
   disagrees is how the two drifted apart in the first place.
3. **The corpus fit reproduces the recorded run.** `analysis_output.txt` is the
   source of every number the case study reports, so a parameter change that alters
   it is a change to the paper.

Run: uv run python experiments/_mop_params_check.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
REVISED = EXPERIMENTS.parent
CASE = EXPERIMENTS / "case_study"

sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REVISED / "src"))

import _mop_params as MP                                          # noqa: E402

RECORDED = CASE / "analysis_output.txt"


def check_coefficient():
    """lambda_sigma = nJ/50 must be the same as coefficient * nJ."""
    print("check 1: the lambda_sigma coefficient is 1/50")
    c = MP.LAMBDA_SIGMA_OVER_N
    bad = []
    for n in (100, 281, 753, 1570, 5401):
        want = n / 50.0
        got = MP.lambda_sigma_for(n)
        if abs(got - want) > 1e-9:
            bad.append((n, got, want))
    print(f"  coefficient = {c!r}  (1/50 = {1/50!r})")
    for n in (281, 5401):
        print(f"  nJ={n:>5}  lambda_sigma_for={MP.lambda_sigma_for(n):.4f}  "
              f"nJ/50={n/50:.4f}")
    if bad:
        for n, got, want in bad:
            print(f"  MISMATCH at nJ={n}: got {got}, want {want}")
    print(f"  -> {'PASS' if not bad else 'FAIL'}")
    return not bad


def check_configs_agree():
    """phi_config.json's copy of the MoP parameters must match mop_params.json."""
    print("\ncheck 2: phi_config.json agrees with mop_params.json")
    phi_cfg = json.loads((CASE / "phi_config.json").read_text(encoding="utf-8"))
    m = phi_cfg["mop"]
    pairs = [
        ("lambda_u", m["lambda_u"], MP.lambda_u),
        ("lambda_pi", m["lambda_pi"], MP.lambda_pi),
        ("lambda_sigma_over_n", m["lambda_sigma_over_n"], MP.LAMBDA_SIGMA_OVER_N),
    ]
    ok = True
    for name, a, b in pairs:
        same = abs(float(a) - float(b)) < 1e-12
        ok &= same
        print(f"  {name:<22} phi_config={a!r:<8} mop_params={b!r:<8} "
              f"{'agree' if same else 'DISAGREE'}")
    # The phi experiment's grid must sit inside the shared maximum.
    if m["max_components"] > MP.max_components:
        print(f"  phi_config max_components {m['max_components']} exceeds "
              f"mop_params {MP.max_components}")
        ok = False
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_recorded_output():
    """The corpus fit must still produce the recorded analysis_output.txt."""
    print("\ncheck 3: step7_analyze.py reproduces the recorded output")
    if not RECORDED.exists():
        print(f"  {RECORDED} missing; skipped")
        return True, None
    proc = subprocess.run(
        [sys.executable, "-u", str(CASE / "step7_analyze.py")],
        cwd=str(CASE), capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REVISED / "src")})
    got = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    # analysis_output.txt carries a UTF-8 BOM; utf-8-sig drops it, and the strip
    # above absorbs any trailing whitespace, so this compares content only.
    want = [ln.strip() for ln in
            RECORDED.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    if got == want:
        print(f"  identical, {len(want)} lines")
        print("  -> PASS")
        return True, want
    import difflib
    diff = list(difflib.unified_diff(want, got, "recorded", "now", lineterm=""))
    print(f"  DIFFERS: recorded {len(want)} lines, now {len(got)} lines")
    for ln in diff[2:14]:
        print(f"    {ln}")
    print("  -> FAIL")
    return False, got


def main():
    print(f"parameters loaded from {MP.PARAMS_PATH}\n")
    print(f"  {MP.summary()}\n")
    ok1 = check_coefficient()
    ok2 = check_configs_agree()
    ok3, _ = check_recorded_output()

    print()
    print("=" * 72)
    for name, ok in (("coefficient is 1/50", ok1),
                     ("configs agree", ok2),
                     ("recorded output reproduced", ok3)):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 72)
    if not (ok1 and ok2 and ok3):
        print("\nThe parameters are NOT verified. Results that depend on them "
              "cannot be quoted until this passes.")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
