"""Compare every printed table cell with its artifact.

A LaTeX table can carry a number typed by hand, so each cell is parsed out of the
.tex file and compared at the precision the table prints. This is what catches a
refit that updated the JSON and not the manuscript.
"""


from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "manuscript" / "sections" / "03_empirical.tex"
TEX_ANNEX = ROOT / "manuscript" / "sections" / "EE_tables.tex"
JSON_PATH = ROOT / "experiments" / "results" / "tables.json"
CURVES = ROOT / "experiments" / "results" / "dim_curves.json"

TYPES = ["BEV", "PHEV", "REEV", "ICEV"]

DIMS = {
    "Cost": "cost",
    "Driving Experience": "driving_experience",
    "Driving": "driving_experience",
    "Range": "range",
    "Refueling Convenience": "refueling_convenience",
    "Refueling": "refueling_convenience",
    "Reliability": "reliability",
    "Policy": "policy",
    "Other": "other",
}


def strip_tex(cell: str) -> str:
    cell = cell.replace("\\textbf{", "")
    cell = cell.replace("{,}", "")   # thousands separator, before braces are dropped
    cell = cell.replace("}", "")
    cell = cell.replace("$", "")
    return cell.strip()


def floats(cell: str) -> float | None:
    s = strip_tex(cell)
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return None
    return float(s)


def ints(cell: str) -> int | None:
    """Integer from a cell, tolerating the printed thousands separator."""
    v = floats(cell)
    return None if v is None else int(v)


def rows_of(tex: str, label: str) -> list[str]:
    """Return the body rows of the tabular that follows \\label{label}."""
    start = tex.index("\\label{" + label + "}")
    start = tex.index("\\begin{tabular}", start)
    start = tex.index("\n", start)
    end = tex.index("\\end{tabular}", start)
    body = tex[start:end]
    out = []
    for line in body.split("\\\\"):
        line = line.strip()
        if not line or line in {"\\toprule", "\\midrule", "\\bottomrule"}:
            continue
        out.append(line)
    return out


def split_row(line: str) -> list[str]:
    line = line.split("\\midrule")[-1].split("\\toprule")[-1]
    return [c for c in line.split("&")]


def decimals(cell: str) -> int:
    s = strip_tex(cell)
    return len(s.split(".")[1]) if "." in s else 0


COMPARED: list[str] = []


def compare(name: str, cell: str, truth: float, tol_scale: float = 1.0) -> str | None:
    got = floats(cell)
    if got is None:
        return None
    COMPARED.append(name)
    nd = decimals(cell)
    # Half a unit in the last printed place, with a little slack for ties.
    tol = 0.5 * 10 ** (-nd) * tol_scale + 1e-9
    if abs(got - truth) > tol:
        return f"{name}: printed {got} (tol {tol:.5f}) vs json {truth:.6f}"
    return None


def check_overall(tex: str, data: dict) -> list[str]:
    errs: list[str] = []
    rows = [r for r in rows_of(tex, "tab:mop_overall") if keeps_row(r)]
    comps = data["components"]
    if len(rows) != len(comps) + 1:
        return [f"tab:mop_overall has {len(rows)} numeric rows, expected {len(comps) + 1}"]
    for i, comp in enumerate(comps):
        cells = split_row(rows[i])
        if not cells[0].strip().startswith(f"C{i + 1}"):
            errs.append(f"row {i + 1} label is {cells[0]!r}, expected C{i + 1}")
        truth = list(comp["utilities"]) + [0.0] + [comp["pi"], comp["phi"]]
        for j, t in enumerate(truth):
            e = compare(f"C{i + 1} col{j + 1}", cells[j + 1], t)
            if e:
                errs.append(e)
    eu = data["expected_utility"]
    cells = split_row(rows[-1])
    for j, t in enumerate(eu):
        e = compare(f"E[U] col{j + 1}", cells[j + 1], t)
        if e:
            errs.append(e)
    return errs


def check_dimensions(tex: str, data: dict) -> list[str]:
    errs: list[str] = []
    rows = [r for r in rows_of(tex, "tab:component_detail")]
    block = None
    idx = 0
    for row in rows:
        if "\\multicolumn" in row:
            title = re.search(r"\\textbf\{([^}]*)\}", row).group(1)
            block = DIMS[title]
            idx = 0
            continue
        if block is None:
            continue
        cells = split_row(row)
        comps = data[block]["components"]
        if cells[0].strip().startswith("$\\bar{u}$"):
            eu = data[block]["expected_utility"]
            for j, t in enumerate(eu):
                e = compare(f"{block} E[U] col{j + 1}", cells[j + 1], t)
                if e:
                    errs.append(e)
            continue
        comp = comps[idx]
        truth = list(comp["utilities"]) + [0.0] + [comp["pi"], comp["phi"]]
        for j, t in enumerate(truth):
            e = compare(f"{block} C{idx + 1} col{j + 1}", cells[j + 1], t)
            if e:
                errs.append(e)
        idx += 1
    return errs


def keeps_row(row: str) -> bool:
    cells = split_row(row)
    return bool(cells) and bool(re.match(r"\s*(C\d|\$\\bar\{u\}\$)", cells[0]))


def name_of(cell: str) -> str:
    m = re.search(r"(BEV|PHEV|REEV|ICEV)", cell)
    return m.group(1) if m else ""


def check_summary(tex: str, data: dict) -> list[str]:
    """Table 3, the per-dimension summary: n, P-hat, Delta-ell, heaviest, E[U]."""
    errs: list[str] = []
    curves = json.loads(CURVES.read_text(encoding="utf-8"))
    rows = [r for r in rows_of(tex, "tab:dimension_summary") if "&" in r][1:]
    if len(rows) != 8:
        return [f"tab:dimension_summary has {len(rows)} rows, expected 8"]
    for row in rows:
        cells = [c.strip() for c in split_row(row)]
        dim = cells[0]
        key = "overall" if dim == "Overall" else DIMS[dim]
        fit = data[key]
        # n is printed with a thousands separator, so the printed precision is the integer.
        for name, cell, truth in (("n", cells[1], fit["n"]),
                                  ("Phat", cells[2], fit["k"]),
                                  ("dll", cells[3], curves[key]["delta_ll"])):
            e = compare(f"{key} {name}", cell, truth)
            if e:
                errs.append(e)
        comps = fit["components"]
        heavy = max(range(len(comps)), key=lambda i: comps[i]["pi"])
        # ICEV is the benchmark at zero, so it competes with the three utilities.
        utils = list(comps[heavy]["utilities"]) + [0.0]
        top = TYPES[max(range(4), key=lambda j: utils[j])]
        if name_of(cells[4]) and name_of(cells[4]) != top:
            errs.append(f"{key} heaviest: printed {name_of(cells[4])} vs json {top}")
        m = re.search(r"\(([\d.]+)\)", cells[4])
        if m:
            e = compare(f"{key} heaviest weight", m.group(1), comps[heavy]["pi"])
            if e:
                errs.append(e)
        eu = fit["expected_utility"]
        lead = TYPES[max(range(4), key=lambda j: eu[j])]
        last = TYPES[min(range(4), key=lambda j: eu[j])]
        if name_of(cells[5]) != lead:
            errs.append(f"{key} E[U] leader: printed {name_of(cells[5])} vs json {lead}")
        if name_of(cells[6]) != last:
            errs.append(f"{key} E[U] last: printed {name_of(cells[6])} vs json {last}")
    return errs


def check_holdout(tex_path: Path, data: dict) -> list[str]:
    """Table S3, the fold-level holdout table, plus the 30-of-35 sentence.

    Every cell comes from ``holdout.json``, which scores the
    predictive distribution each model implies for a held-out judgment.  The point
    metrics of ``old point-error files are no longer printed.
    """
    errs: list[str] = []
    tex = tex_path.read_text(encoding="utf-8")
    rows = [[c.strip() for c in split_row(r)] for r in rows_of(tex, "tab:validation_holdout")]
    rows = [r for r in rows if r and r[0] != "Dimension"]
    if len(rows) != 8:
        return [f"tab:validation_holdout has {len(rows)} rows, expected 8"]
    order = [("Cost", "cost"), ("Driving", "driving_experience"), ("Range", "range"),
             ("Refueling", "refueling_convenience"), ("Reliability", "reliability"),
             ("Policy", "policy"), ("Other", "other"), ("Overall", "overall")]
    for cells, (label, key) in zip(rows, order):
        sec = data[key]
        summ = sec["summary"]
        if ints(cells[1]) != sec["n_pairs"]:
            errs.append(f"{key} n: printed {cells[1]} vs json {sec['n_pairs']}")
        lo, hi = min(summ["k_selected"]), max(summ["k_selected"])
        printed = cells[2]
        want = str(lo) if lo == hi else f"{lo}--{hi}"
        if printed != want:
            errs.append(f"{key} Phat: printed {printed} vs json {want}")
        for field, col in (("log_score", 3), ("crps", 6)):
            e = compare(f"{key} MoP {field}", cells[col], summ["MoP"][field])
            if e:
                errs.append(e)
            e = compare(f"{key} P1 {field}", cells[4 if field == "log_score" else 7],
                        summ["P=1"][field])
            if e:
                errs.append(e)
        gain = summ["MoP"]["log_score"] - summ["P=1"]["log_score"]
        e = compare(f"{key} gain", cells[5], gain)
        if e:
            errs.append(e)
        # The printed gain must also equal the difference of the two printed columns.
        d_printed = floats(cells[3]) - floats(cells[4])
        if abs(d_printed - gain) > 1.5e-3:
            errs.append(f"{key} gain is inconsistent with the two ll columns "
                        f"({d_printed:+.4f} vs {gain:+.4f})")
    wins = 0
    total = 0
    for key in data:
        if key == "overall":
            continue
        for fold in data[key]["per_fold"]:
            total += 1
            wins += fold["MoP"]["log_score"] > fold["P=1"]["log_score"]
    flat = re.sub(r"\s+", " ", tex)
    if f"{wins} of the {total}" not in flat:
        errs.append(f"text should say {wins} of the {total} dimension-fold comparisons")
    # The paired interval quoted in the text must match the bootstrap in the json.
    ci = data["overall"]["paired"]["crps"]
    for value in (ci["lo"], ci["hi"]):
        if f"{value:.4f}" not in flat:
            errs.append(f"CRPS interval bound {value:.4f} not found in the text")
    return errs



def check_retention(tex_path: Path, data: dict) -> list[str]:
    """Table S4: the share of the in-sample gain that survives out of sample.

    The denominator is the Delta-ell of Table 3, read from ``dim_curves.json``, so the
    two tables cannot drift apart without one of the two checks failing.
    """
    errs: list[str] = []
    tex = tex_path.read_text(encoding="utf-8")
    curves = json.loads(CURVES.read_text(encoding="utf-8"))
    rows = [[c.strip() for c in split_row(r)] for r in rows_of(tex, "tab:retention")]
    rows = [r for r in rows if len(r) == 6 and r[0] != "Dimension"]
    if len(rows) != 8:
        return [f"tab:retention has {len(rows)} rows, expected 8"]
    order = [("Cost", "cost"), ("Driving", "driving_experience"), ("Range", "range"),
             ("Refueling", "refueling_convenience"), ("Reliability", "reliability"),
             ("Policy", "policy"), ("Other", "other"), ("Overall", "overall")]
    for cells, (label, key) in zip(rows, order):
        if cells[0] != label:
            errs.append(f"tab:retention row {cells[0]!r} where {label!r} expected")
            continue
        ins = data[key]["summary"]["in_sample"]
        n_val = floats(cells[1])
        if n_val is not None and n_val != data[key]["n_pairs"]:
            errs.append(f"{key} n: printed {cells[1]} vs json {data[key]['n_pairs']}")
        # A section whose count the criterion leaves at one has no in-sample gain
        # and therefore no ratio; the table prints dashes and the check skips them.
        ratio = None if ins["retained"] is None else ins["retained"] * 100.0
        for name, cell, want in (
            (f"{key} dl/n", cells[2], ins["delta_ll_per_judgment"]),
            (f"{key} held-out/n", cells[3], ins["holdout_gain_per_judgment"]),
            (f"{key} retained", cells[4], ratio),
        ):
            if want is None:
                if floats(cell) is not None:
                    errs.append(f"{key} retained: printed {cell}, but the ratio is undefined")
                continue
            e = compare(name, cell, want)
            if e:
                errs.append(e)
        # The printed share must agree with the two printed gains.
        a, b, r = floats(cells[2]), floats(cells[3]), floats(cells[4])
        if None not in (a, b, r) and a > 0 and abs(b / a * 100.0 - r) > 0.15:
            errs.append(f"{key} retained is inconsistent with the two gain columns "
                        f"({b / a * 100.0:.1f} vs {r:.1f})")
        # Last column: folds whose search returns the full-data count.
        k_full = int(curves[key]["k"])
        k_folds = data[key]["summary"]["k_selected"]
        hit = sum(1 for k in k_folds if int(k) == k_full)
        want = f"{hit}/{len(k_folds)}"
        if cells[5] != want:
            errs.append(f"{key} reproduced: printed {cells[5]!r} vs json {want!r}")
        COMPARED.append(f"{key} reproduced")
    flat = re.sub(r"\s+", " ", tex)
    for phrase in ("between 88 and 98\\%", "no ratio to report", "19\\%"):
        if phrase not in flat:
            errs.append(f"prose should state {phrase!r}")
    # The main text quotes the same column; it must not drift from the table.
    main = re.sub(r"\s+", " ", TEX.read_text(encoding="utf-8"))
    for phrase in ("in four folds of five for Other and for the pooled fit",):
        if phrase not in main:
            errs.append(f"main text should state {phrase!r}")
    return errs


def check_recovery(tex_path: Path, data: dict) -> list[str]:
    """Table 1, the five simulation conditions."""
    errs: list[str] = []
    tex = tex_path.read_text(encoding="utf-8")
    rows = [[c.strip() for c in split_row(r)] for r in rows_of(tex, "tab:recovery")]
    rows = [r for r in rows if r and r[0] != "Condition"]
    blocks = {
        "Cancellation": (data["cancellation"]["sigma2=0.05_n=500"], 2),
        "Inattentive noise": (data["noise"]["sigma2=0.05_noise=0.2_n=500"], 2),
        "Homogeneous": (data["general"]["K=1_sigma2=0.05_n=500"], 1),
        "Two components": (data["general"]["K=2_sigma2=0.05_n=500"], 2),
        "Four components": (data["general"]["K=4_sigma2=0.05_n=500"], 4),
    }
    if len(rows) != len(blocks):
        return [f"tab:recovery has {len(rows)} rows, expected {len(blocks)}"]
    for cells in rows:
        label = re.sub(r"\s+", " ", cells[0])
        if label not in blocks:
            errs.append(f"tab:recovery row {label!r} is not a known condition")
            continue
        block, truth_k = blocks[label]
        mop, p1 = block["MoP"], block["P=1"]
        if int(cells[1]) != truth_k:
            errs.append(f"{label} P: printed {cells[1]} vs json {truth_k}")
        e = compare(f"{label} Phat", cells[2], mop["k_mean"])
        if e:
            errs.append(e)
        # The noise condition has no true count, so its "hit rate" is the share of
        # replicates that selected more than one component, not k_accuracy. Both are
        # printed as proportions on [0, 1]; the JSON stores the second as a fraction too.
        if "k_accuracy" in mop:
            e = compare(f"{label} hit rate", cells[3], mop["k_accuracy"])
        elif "mop_k_gt1_pct" in block:
            e = compare(f"{label} hit rate", cells[3], block["mop_k_gt1_pct"])
        else:
            e = f"{label} hit rate: no source in the JSON"
        if e:
            errs.append(e)
        e = compare(f"{label} cosine MoP", cells[4], mop["cosine_sim"]["mean"])
        if e:
            errs.append(e)
        e = compare(f"{label} cosine P=1", cells[5], p1["cosine_sim"]["mean"])
        if e:
            errs.append(e)
        if "kl_pi" in mop and floats(cells[6]) is not None:
            e = compare(f"{label} KL", cells[6], mop["kl_pi"]["mean"])
            if e:
                errs.append(e)
    return errs


def check_binomial(tex_path: Path, data: dict) -> list[str]:
    """Table S2, the binomial protocol: the tally, the cosine and the held-out gap.

    The count is printed as a tally (``3 in 85``) rather than as a number, so that
    cell is rebuilt from the JSON and compared as a string.
    """
    errs: list[str] = []
    tex = tex_path.read_text(encoding="utf-8")
    rows = [r for r in rows_of(tex, "tab:binomial-supplement") if "&" in r]
    rows = [[c.strip() for c in split_row(r)] for r in rows]
    rows = [r for r in rows if len(r) == 5 and r[0] != "Generating process"]
    if len(rows) != 2:
        return [f"tab:binomial-supplement has {len(rows)} rows, expected 2"]

    def tally(counts: dict) -> str:
        pairs = sorted(((int(k), int(v)) for k, v in counts.items()), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{k} in {n}" for k, n in pairs)

    for cells, key in zip(rows, ("mixture", "betabinom")):
        src = data[key]
        want = tally(src["p_hat_counts"])
        COMPARED.append(f"binomial {key} tally")
        if cells[2] != want:
            errs.append(f"binomial {key} tally: printed {cells[2]!r} vs json {want!r}")
        e = compare(f"binomial {key} held-out",
                    cells[4], src["delta_ll_mixture_vs_betabinom"]["mean_per_test_judgment"])
        if e:
            errs.append(e)
        if key == "mixture":
            e = compare("binomial cosine", cells[3], src["cosine_mean"])
            if e:
                errs.append(e)
    return errs


def check_years(tex: str, data: dict) -> list[str]:
    """Table S6, the two annual fits."""
    errs: list[str] = []
    rows = rows_of(tex, "tab:year_overall")
    year = None
    idx = 0
    seen = 0
    for row in rows:
        if "\\multicolumn" in row:
            year = re.search(r"\\textbf\{(20\d\d)\}", row).group(1)
            fit = data[year]
            n_cell = re.search(r"n=([\d{},]+)", row).group(1).replace("{,}", "")
            if int(n_cell) != fit["n"]:
                errs.append(f"{year} n: printed {n_cell} vs json {fit['n']}")
            k_cell = re.search(r"\\widehat\{\\mathcal\{P\}\}=(\d)", row).group(1)
            if int(k_cell) != fit["k"]:
                errs.append(f"{year} Phat: printed {k_cell} vs json {fit['k']}")
            p_cell = re.search(r"BIC\$=([\d{},.]+)", row).group(1).replace("{,}", "")
            e = compare(f"{year} BIC", p_cell, fit["bic"])
            if e:
                errs.append(e)
            idx = 0
            seen += 1
            continue
        if year is None or "&" not in row:
            continue
        cells = split_row(row)
        if cells[0].strip().startswith("$\\bar{u}$"):
            eu = data[year]["expected_utility"]
            for j, t in enumerate(eu):
                e = compare(f"{year} E[U] col{j + 1}", cells[j + 1], t)
                if e:
                    errs.append(e)
            continue
        comp = data[year]["components"][idx]
        truth = list(comp["utilities"]) + [0.0] + [comp["pi"], comp["phi"]]
        for j, t in enumerate(truth):
            e = compare(f"{year} C{idx + 1} col{j + 1}", cells[j + 1], t)
            if e:
                errs.append(e)
        idx += 1
    if seen != 2:
        errs.append(f"tab:year_overall has {seen} blocks, expected 2")
    return errs


def main() -> int:
    if not TEX.exists():
        print(f"skipped: {TEX.relative_to(ROOT)} is not present, so the tables cannot be read")
        return 0
    tex = TEX.read_text(encoding="utf-8")
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    fits = {"overall": data["overall"], **data["dimensions"]}
    HOLD = json.loads((ROOT / "experiments" / "results" / "holdout.json").read_text(encoding="utf-8"))["sections"]
    SIM = json.loads((ROOT / "experiments" / "results" / "recovery.json").read_text(encoding="utf-8"))
    BINOMIAL = json.loads((ROOT / "experiments" / "results" / "binomial.json").read_text(encoding="utf-8"))

    def run(fn, *args) -> tuple[list[str], int]:
        before = len(COMPARED)
        return fn(*args), len(COMPARED) - before

    errs: list[str] = []
    counts: dict[str, int] = {}
    for name, fn, args in (
        ("Table 1", lambda: check_recovery(ROOT / "manuscript" / "sections" / "03_simulation.tex", SIM), ()),
        ("Tables 2 and 4", lambda: check_overall(tex, data["overall"]) + check_dimensions(tex, data["dimensions"]), ()),
        ("Table 3", lambda: check_summary(tex, fits), ()),
        ("Table S6", lambda: check_years(TEX_ANNEX.read_text(encoding="utf-8"), data["years"]), ()),
        ("Table S3", lambda: check_holdout(ROOT / "manuscript" / "sections" / "BB_holdout.tex", HOLD), ()),
        ("Table S4", lambda: check_retention(ROOT / "manuscript" / "sections" / "BB_holdout.tex", HOLD), ()),
        ("Table S2", lambda: check_binomial(ROOT / "manuscript" / "sections" / "FF_validation.tex", BINOMIAL), ()),
    ):
        e, n = run(fn, *args)
        errs += e
        counts[name] = n

    n_blocks = sum(1 for d in set(DIMS.values()) if any(c.startswith(d + " ") for c in COMPARED))
    n_blocks += 1 if any(c.startswith("E[U] col") for c in COMPARED) else 0
    if errs:
        print(f"FAIL: {len(errs)} mismatched cells of {len(COMPARED)} compared")
        for e in errs:
            print("  " + e)
        return 1
    # Coverage floors, so that a parser that silently skips rows cannot report OK.
    floors = {"Table 1": 23, "Tables 2 and 4": 134, "Table 3": 32, "Table S6": 52,
              "Table S3": 40, "Table S4": 31, "Table S2": 4}
    short = {k: (counts[k], floors[k]) for k in floors if counts[k] < floors[k]}
    if short or n_blocks != 8:
        print(f"FAIL: compared {counts} over {n_blocks} blocks, floors {floors}")
        return 1
    print("OK: " + ", ".join(f"{counts[k]} cells of {k}" for k in floors) +
          " match the JSON artifacts at the printed precision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
