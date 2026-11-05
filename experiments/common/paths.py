"""Paths and constants shared by every experiment."""


from __future__ import annotations

import json
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENTS.parent
CONFIG_DIR = EXPERIMENTS / "config"
RESULTS = EXPERIMENTS / "results"
FIGURES = PROJECT / "manuscript" / "figures"

#: the committed corpus: 5,401 cross-energy-type judgments, one article per line
ENERGY_PAIRS = EXPERIMENTS / "case_study" / "energy_pairs.jsonl"


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return json.load(f)


CASE = load_config("case_study.json")
SIM = load_config("simulation.json")

#: maps the extractor's raw labels onto the four analysed energy types
TYPE_MAP = CASE["energy_type_mapping"]

#: the four alternatives, in the order every table and figure uses
T = CASE["canonical_types"]

#: the seven evaluation dimensions, in table order (``other`` last)
DIMS = CASE["stage2_llm"]["dimensions"]

#: long label of each dimension, as the extractor's prompt defines it
DIM_LABELS = {
    "cost": "Cost",
    "driving_experience": "Driving",
    "range": "Range",
    "refueling_convenience": "Refueling",
    "reliability": "Reliability",
    "policy": "Policy",
    "other": "Other",
}

#: short English names used in the tables
ABBREV = {"纯电": "BEV", "插混": "PHEV", "增程": "REEV", "燃油": "ICEV"}

#: the alternative every utility vector is measured against
REFERENCE = "燃油"


def result(name: str) -> Path:
    """Path of an artifact under ``experiments/results``."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS / name
