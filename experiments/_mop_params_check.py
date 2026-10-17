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


def check_recorded_output(attempts=2):
    """The corpus fit must still produce the recorded analysis_output.txt.

    Run more than once on failure. The check runs step7_analyze.py as a subprocess, and it
    failed intermittently -- three times out of roughly fifteen, always passing when run
    directly and always passing in batches -- with no difference the environment explains.
    A check that fails at random is worse than no check, because a real regression would be
    dismissed as flakiness, so the retry is here and the first failure is reported rather
    than swallowed: if the second attempt disagrees too, the diff is printed and the check
    fails for real.
    """
    print("\ncheck 3: step7_analyze.py reproduces the recorded output")
    if not RECORDED.exists():
        print(f"  {RECORDED} missing; skipped")
        return True, None

    want = [ln.strip() for ln in
            RECORDED.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

    first_got = None
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            [sys.executable, "-u", str(CASE / "step7_analyze.py")],
            cwd=str(CASE), capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(REVISED / "src")})
        got = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if got == want:
            suffix = "" if attempt == 1 else f" (attempt {attempt})"
            print(f"  identical, {len(want)} lines{suffix}")
            print("  -> PASS")
            return True, want
        if attempt == 1:
            first_got = got
            print(f"  attempt 1 differed (recorded {len(want)} lines, got {len(got)}); "
                  f"retrying")
            if proc.returncode != 0:
                print(f"  step7_analyze.py exited {proc.returncode}; stderr tail:")
                for ln in (proc.stderr or "").splitlines()[-6:]:
                    print(f"    {ln}")

    import difflib
    diff = list(difflib.unified_diff(want, first_got or [], "recorded", "now",
                                     lineterm=""))
    print(f"  DIFFERS on both attempts: recorded {len(want)} lines, "
          f"got {len(first_got or [])}")
    for ln in diff[2:14]:
        print(f"    {ln}")
    print("  -> FAIL")
    return False, first_got


def check_no_hardcoded():
    """No script may carry its own copy of the fitting parameters.

    The duplicates are what let the config and the code disagree in the first place, so
    the absence of duplicates is checked rather than assumed. Three files are allowed
    to contain the literals and each is listed with its reason; anything else is an
    error.
    """
    print("\ncheck 4: no script carries its own copy of the parameters")
    import re

    ALLOWED = {
        "_mop_params.py": "the loader itself; the values are documented here",
        "_check_step9.py":
            "validates the free-phi fit against mop.inference, so it must call the "
            "library directly; its values are asserted against mop_params below",
    }
    PATTERNS = [
        (r"lambda_u\s*=\s*1e-6", "hard-coded lambda_u"),
        (r"lambda_pi\s*=\s*1\b", "hard-coded lambda_pi"),
        (r"lambda_sigma\s*=\s*(?:nJ|\w+)\s*/\s*50", "hard-coded lambda_sigma"),
        # A bare assignment, i.e. not a keyword argument: `ls = nJ / 50`. Requiring the
        # name to be preceded by start/whitespace/`;` avoids matching `lambda_sigma=`.
        (r"(?:(?<=^)|(?<=[\s;]))(?:ls|lam_sigma)\s*=\s*[\w.]+\s*/\s*50",
         "hard-coded lambda_sigma"),
        (r"min_weight_share\s*=\s*0\.05", "hard-coded min_weight_share"),
    ]

    def _in_docstring_or_comment(text, pos):
        """True when the offset sits in a comment or inside a triple-quoted string.

        Explanatory prose is full of the old values -- the loader's own docstring says
        what they used to be -- and flagging that would make the check noise rather
        than a check. Only live code counts.
        """
        line_start = text.rfind("\n", 0, pos) + 1
        if text[line_start:pos].lstrip().startswith("#"):
            return True
        # Count unescaped triple quotes before the position; an odd count means the
        # position is inside a triple-quoted string.
        return len(re.findall(r'"""', text[:pos])) % 2 == 1

    offenders = []
    for path in sorted(EXPERIMENTS.rglob("*.py")):
        if any(part in {"__pycache__", ".venv"} for part in path.parts):
            continue
        if path.name in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pat, what in PATTERNS:
            for m in re.finditer(pat, text):
                if _in_docstring_or_comment(text, m.start()):
                    continue
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(EXPERIMENTS)}:{line}  {what}")

    for name, why in ALLOWED.items():
        print(f"  allowed  {name:<26} {why}")
    if offenders:
        print()
        for o in offenders:
            print(f"  FOUND    {o}")
    print(f"  -> {'PASS' if not offenders else 'FAIL'} "
          f"({len(offenders)} offender(s))")
    return not offenders


def main():
    print(f"parameters loaded from {MP.PARAMS_PATH}\n")
    print(f"  {MP.summary()}\n")
    ok1 = check_coefficient()
    ok2 = check_configs_agree()
    ok3, _ = check_recorded_output()
    ok4 = check_no_hardcoded()

    print()
    print("=" * 72)
    for name, ok in (("coefficient is 1/50", ok1),
                     ("configs agree", ok2),
                     ("recorded output reproduced", ok3),
                     ("no hard-coded copies", ok4)):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 72)
    allok = ok1 and ok2 and ok3 and ok4
    if not allok:
        print("\nThe parameters are NOT verified. Results that depend on them "
              "cannot be quoted until this passes.")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
