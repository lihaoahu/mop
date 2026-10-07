"""混合优先级模型数据结构"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Priority:
    """单个优先级组件"""
    priority_vector: np.ndarray      # 优先级向量 (归一化后)
    inconsistency_level: float       # 不一致性水平 σ²

    def __repr__(self):
        return f"Priority(w={self.priority_vector.round(4)}, σ²={self.inconsistency_level:.4f})"


class MixtureOfPriorities:
    """混合优先级模型"""

    def __init__(self, priorities: list[Priority], mixture_probabilities: np.ndarray):
        self.priorities = priorities
        self.mixture_probabilities = mixture_probabilities
        self._log_likelihood = None

    def __getitem__(self, index: int) -> Priority:
        return self.priorities[index]

    def __len__(self) -> int:
        return len(self.priorities)

    def __repr__(self):
        lines = [f"MixtureOfPriorities(K={len(self.priorities)})"]
        for i, (pi, pri) in enumerate(zip(self.mixture_probabilities, self.priorities)):
            lines.append(f"  [{i}] π={pi:.3f}, {pri}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "mixture": {
                "pis": self.mixture_probabilities.tolist(),
                "priorities": [p.priority_vector.tolist() for p in self.priorities],
                "inconsistencies": [p.inconsistency_level for p in self.priorities],
            },
            "log_likelihood": self._log_likelihood,
        }


def distance(priority_a: Priority, priority_b: Priority) -> float:
    """两个优先级向量之间的欧氏距离"""
    return float(np.linalg.norm(priority_a.priority_vector - priority_b.priority_vector))
