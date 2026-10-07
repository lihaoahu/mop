"""
决策层面的概率计算
"""

import numpy as np
from scipy.stats import norm

try:
    from .pairwise_comparison import PairwiseComparison, comparisons2regression
    from .mixture_of_priorities import MixtureOfPriorities, Priority
except ImportError:
    from pairwise_comparison import PairwiseComparison, comparisons2regression
    from mixture_of_priorities import MixtureOfPriorities, Priority


def likelihood(pc: PairwiseComparison, priority: Priority, link_function):
    """
    计算单个比较在给定优先级下的似然
    
    Args:
        pc: 成对比较
        priority: 优先级
        link_function: 链接函数
        
    Returns:
        似然值
    """
    w = priority.priority_vector
    nA = len(w)
    u = np.log(w[:nA - 1] / w[nA - 1])
    sigma2 = priority.inconsistency_level
    
    X, y, _ = comparisons2regression([pc], [link_function], nA)
    
    mean = X[0, :] @ u
    std = np.sqrt(sigma2)
    
    return norm.pdf(y[0], loc=mean, scale=std)


def responsivity(pc: PairwiseComparison, mop: MixtureOfPriorities, link_function):
    """
    计算responsivity（后验概率）
    
    Args:
        pc: 成对比较
        mop: 混合优先级模型
        link_function: 链接函数
        
    Returns:
        后验概率向量
    """
    nP = len(mop.mixture_probabilities)
    likelihoods = np.array([likelihood(pc, mop[p], link_function) for p in range(nP)])
    
    numerator = mop.mixture_probabilities * likelihoods
    denominator = np.sum(numerator)
    
    return numerator / denominator
