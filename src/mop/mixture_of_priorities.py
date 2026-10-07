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


def expected_utility(priorities, mixture_probabilities, floor=1e-300):
    """Aggregate ranking: the expected utility of each alternative.

        u_bar_i = sum_p pi_p * log(w_pi / w_pA)

    Utilities are expressed against the benchmark alternative, which the model
    places last in every priority vector. Each component is therefore anchored
    at the same origin and the utilities can be averaged across components,
    which the raw weights could not be. The value returned for the benchmark is
    exactly zero.

    ``priorities`` may be ``Priority`` objects or plain weight vectors.
    """
    out = None
    for pri, pi in zip(priorities, mixture_probabilities):
        w = np.asarray(getattr(pri, "priority_vector", pri), dtype=float)
        # a component can sit at the edge of the simplex, where a weight is
        # numerically zero; the floor keeps the log finite rather than letting
        # an infinity swamp every other component
        u = np.log(np.maximum(w, floor)) - np.log(max(w[-1], floor))
        out = pi * u if out is None else out + pi * u
    return out
