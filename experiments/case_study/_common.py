"""共享工具：从 config.json 加载参数和路径"""

import json
from pathlib import Path

V1_DIR = Path(__file__).resolve().parent
PROJECT = V1_DIR.parent.parent

# 加载配置
with open(V1_DIR / "config.json", encoding="utf-8") as f:
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

ALL_ARTICLES = V1_DIR / "all_articles.jsonl"
COMPARISON_ARTICLES = V1_DIR / "comparison_articles.jsonl"
ENERGY_PAIRS = _resolve("energy_pairs")
MOP_RESULTS = _resolve("mop_results")

# 确保目录存在
for d in [DATA_DIR, RAW_HTML_DIR, TRANSCRIPT_DIR, V1_DIR]:
    d.mkdir(parents=True, exist_ok=True)
