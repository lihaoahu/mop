"""
成对比较数据结构
"""

from dataclasses import dataclass
from typing import Any, List
import numpy as np


@dataclass
class PairwiseComparison:
    """成对比较数据结构"""
    former_id: int
    latter_id: int
    agent_id: int
    judgment: Any

    def __repr__(self):
        return f"({self.former_id}→{self.latter_id}, {self.agent_id}, {self.judgment})"


def pairwise_comparisons_extractor(R: np.ndarray, agent: int = 1) -> List[PairwiseComparison]:
    """从判断矩阵提取成对比较信息"""
    n_row, n_col = R.shape
    if n_row != n_col:
        raise ValueError("非方阵！")

    nA = n_row
    prvector = []

    for i in range(nA):
        for j in range(i + 1, nA):
            if not np.isnan(R[i, j]):
                prvector.append(PairwiseComparison(
                    former_id=i + 1,
                    latter_id=j + 1,
                    agent_id=agent,
                    judgment=R[i, j]
                ))

    return prvector


def pairwise_comparisons_extractor_multi(Rs: List[np.ndarray]) -> List[PairwiseComparison]:
    """从多个判断矩阵提取成对比较信息"""
    if not Rs:
        return []

    R1 = Rs[0]
    prvector = pairwise_comparisons_extractor(R1, 1)

    for i, Ri in enumerate(Rs[1:], start=2):
        prvector.extend(pairwise_comparisons_extractor(Ri, i))

    return prvector


def comparisons2regression(
    pcvector: List[PairwiseComparison],
    link_functions: List,
    nA: int = None
):
    """
    将比较数据转换为回归问题格式（优化版）
    
    Args:
        pcvector: 成对比较对象列表
        link_functions: 链接函数列表
        nA: 备选方案数量，如果为None则自动推断
    
    Returns:
        X: 设计矩阵 (nJ x nA-1)
        y: 响应变量
        d: 决策主体ID数组
    
    Raises:
        ValueError: 当决策主体数量与链接函数数量不一致时
    """
    if nA is None:
        all_ids = set()
        for pc in pcvector:
            all_ids.add(pc.former_id)
            all_ids.add(pc.latter_id)
        # 使用最大ID而非数量，确保设计矩阵有足够的列
        nA = max(all_ids) if all_ids else 0

    nJ = len(pcvector)
    d = np.array([pc.agent_id for pc in pcvector])
    
    # 创建 agent_id 到连续索引的映射，处理不连续的 agent_id
    unique_agents = sorted(set(d))
    agent_to_idx = {agent: idx for idx, agent in enumerate(unique_agents)}
    nE = len(unique_agents)

    if len(link_functions) != nE:
        raise ValueError(
            f"决策主体数量({nE})与链接函数数量({len(link_functions)})不一致"
        )

    X = np.zeros((nJ, nA), dtype=int)
    y = np.zeros(nJ, dtype=float)

    for k in range(nJ):
        pc = pcvector[k]
        X[k, pc.former_id - 1] = 1
        X[k, pc.latter_id - 1] = -1

        judgment = pc.judgment
        # 使用映射后的索引访问链接函数
        link_func = link_functions[agent_to_idx[pc.agent_id]]

        try:
            y_val = link_func(judgment)
            if np.isnan(y_val) or np.isinf(y_val):
                y[k] = 0.0
            else:
                y[k] = y_val
        except Exception:
            y[k] = 0.0

    return X[:, :nA - 1], y, d
