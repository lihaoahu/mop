"""Pairwise judgments and the design matrix they imply."""


from dataclasses import dataclass
from typing import Any, List
import numpy as np


@dataclass
class PairwiseComparison:
    """One judgment: alternatives ``former_id`` and ``latter_id``, the judge, and the response."""
    former_id: int
    latter_id: int
    agent_id: int
    judgment: Any

    def __repr__(self):
        return f"({self.former_id}→{self.latter_id}, {self.agent_id}, {self.judgment})"


def pairwise_comparisons_extractor(R: np.ndarray, agent: int = 1) -> List[PairwiseComparison]:
    """Read the upper triangle of a judgment matrix, skipping unrecorded pairs."""
    n_row, n_col = R.shape
    if n_row != n_col:
        raise ValueError("judgment matrix must be square")

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
    """Same, one judge per matrix."""
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
    """Design matrix, response and judge index for the mixed linear predictor.

    ``link_functions`` holds one response transform per judge: the paper's ``g``, which maps
    a judgment onto the response scale. Raises if a judgment has no finite transform rather
    than substituting a default.
    """
    if nA is None:
        all_ids = set()
        for pc in pcvector:
            all_ids.add(pc.former_id)
            all_ids.add(pc.latter_id)
        nA = max(all_ids) if all_ids else 0

    nJ = len(pcvector)
    d = np.array([pc.agent_id for pc in pcvector])

    unique_agents = sorted(set(d))
    agent_to_idx = {agent: idx for idx, agent in enumerate(unique_agents)}
    nE = len(unique_agents)

    if len(link_functions) != nE:
        raise ValueError(
            f"{nE} judges but {len(link_functions)} link functions; one transform per judge is needed"
        )

    X = np.zeros((nJ, nA), dtype=int)
    y = np.zeros(nJ, dtype=float)

    for k in range(nJ):
        pc = pcvector[k]
        X[k, pc.former_id - 1] = 1
        X[k, pc.latter_id - 1] = -1

        judgment = pc.judgment
        link_func = link_functions[agent_to_idx[pc.agent_id]]

        try:
            y_val = float(link_func(judgment))
        except Exception as exc:
            raise ValueError(
                f"judgment {k} of comparison ({pc.former_id}, {pc.latter_id}) could not be "
                f"transformed: {judgment!r} raised {type(exc).__name__}: {exc}"
            ) from exc
        if not np.isfinite(y_val):
            raise ValueError(
                f"judgment {k} of comparison ({pc.former_id}, {pc.latter_id}) transforms to "
                f"{y_val}; a non-finite response is not a valid observation. Clip the judgment "
                f"before the link function or remove the comparison."
            )
        y[k] = y_val

    return X[:, :nA - 1], y, d
