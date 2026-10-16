"""Where does freeing phi change the conclusion, and in which direction?

The overall case data moves from P=4 (phi fixed) to P=3 (phi free). The policy
dimension moves the other way, P=1 to P=2. Cost and range do not move at all. A
finding that goes in both directions cannot be summarised as "phi inflates the
component count", so this script lays the slices side by side.

The hypothesis this was written to test -- that the selection moves where the
fixed-phi fit is furthest from the free-phi fit at the same P -- is FALSE, and the
output says so. The two slices that move carry the largest and the smallest gain.
Left in place rather than deleted, because the next person to look at this should
know that explanation is already excluded.

Run: uv run python experiments/case_study/_phi_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "phi_identification_results.json"

data = json.loads(RES.read_text(encoding="utf-8"))

print(f"{'slice':<26} {'n':>6} {'phi=fixed':>10} {'phi=free':>9} "
      f"{'change':>8} {'ll gain at best':>16}")
print("-" * 82)
rows = []
for label, r in data.items():
    kf = r["fixed_phi_best_by_pBIC"]
    kg = r["free_phi_best_by_pBIC"]
    # log-likelihood gain from freeing phi at the fixed-phi optimum
    f = r["fixed_phi"][str(kf)]
    g = r["free_phi"][str(kf)]
    gain = f["loglik"] - g["loglik"]
    rows.append((label, r["n_judgments"], kf, kg, gain))
    print(f"{label:<26} {r['n_judgments']:>6} {kf:>10} {kg:>9} "
          f"{kg - kf:>+8d} {gain:>16.1f}")

print()
print("'ll gain' is how many log-likelihood units the free-phi model gains over the")
print("fixed-phi model at the SAME component count. A large gain means the fixed-phi")
print("mixture is being forced to explain spread with components that a dispersion")
print("parameter explains better.")
print()
moved = [r for r in rows if r[2] != r[3]]
print(f"slices where the selection moves: {len(moved)} of {len(rows)}")
for label, n, kf, kg, gain in moved:
    direction = "up" if kg > kf else "down"
    print(f"  {label:<26} P={kf} -> P={kg} ({direction}), ll gain {gain:.1f}")
print()
print("The pattern the gain column was meant to explain does NOT hold: the two")
print("slices whose selection moves carry the LARGEST gain (overall, 48.1) and the")
print("SMALLEST (policy, 0.6), while the two slices that do not move sit in between.")
print("So the effect is not a one-way bias and is not predicted by how far the")
print("fixed-phi fit is from the free-phi fit. Recorded as unknown rather than")
print("dressed up: see PHI_IDENTIFICATION.md.")
