"""共享工具：从 config.json 加载参数和路径"""

import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
PROJECT = CASE_DIR.parent.parent

# 加载配置
with open(CASE_DIR / "config.json", encoding="utf-8") as f:
    CFG = json.load(f)

# 快捷访问
MOP = CFG["mop"]
TYPE_MAP = CFG["energy_type_mapping"]
T = CFG["canonical_types"]
DIMS = CFG["stage2_llm"]["dimensions"]
DIM_LABELS = CFG["stage2_llm"]["dimension_labels"]

# 数据管道路径
DATA_DIR = PROJECT / "data"
IDS_FILE = DATA_DIR / "article_ids.jsonl"
RAW_HTML_DIR = DATA_DIR / "raw" / "pages"
DETAILS_FILE = DATA_DIR / "article_details.jsonl"
TRANSCRIPT_DIR = DATA_DIR / "transcript"

# 实验文件路径
def _resolve(key):
    return PROJECT / CFG["files"][key]

ALL_ARTICLES = CASE_DIR / "all_articles.jsonl"
COMPARISON_ARTICLES = CASE_DIR / "comparison_articles.jsonl"
ENERGY_PAIRS = _resolve("energy_pairs")
MOP_RESULTS = _resolve("mop_results")

# 确保目录存在
for d in [DATA_DIR, RAW_HTML_DIR, TRANSCRIPT_DIR, CASE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def expected_utility(priority_vectors, mixture_probabilities):
    """Aggregate ranking: the expected utility of each alternative.

        u_bar_i = sum_p pi_p * log(w_pi / w_pA)

    Utilities are expressed relative to the benchmark alternative, which the
    model places last in the priority vector. Every component is therefore
    anchored at the same origin and the utilities can be averaged across
    components, which the raw weights could not be. The value returned for the
    benchmark is exactly zero.
    """
    w0 = np.asarray(priority_vectors[0], dtype=float)
    u_bar = np.zeros(w0.shape[0])
    for w, pi in zip(priority_vectors, mixture_probabilities):
        w = np.asarray(w, dtype=float)
        # a component can sit at the edge of the simplex, where a weight is
        # numerically zero; the floor keeps the log finite rather than returning
        # an infinity that would swamp every other component
        u_bar += pi * (np.log(np.maximum(w, 1e-300)) - np.log(max(w[-1], 1e-300)))
    return u_bar
