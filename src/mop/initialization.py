"""
初始化算法 - KRidge
"""

import numpy as np

# 尝试导入Cython版本
USE_CYTHON = False
try:
    from cython_initialization import kridge_cython
    USE_CYTHON = True
except ImportError:
    pass

try:
    from .pairwise_comparison import comparisons2regression
except ImportError:
    from pairwise_comparison import comparisons2regression


def kridge(X: np.ndarray, y: np.ndarray, nP: int, lambda_reg: float = 1e-10):
    """
    KRidge算法 - 用于初始化混合优先级模型
    
    Args:
        X: 设计矩阵 (nJ x nA-1)
        y: 响应变量 (nJ,)
        nP: 优先级组件数量
        lambda_reg: 岭回归正则化参数
        
    Returns:
        u: 优先级向量 (nP x nA-1)
        sigma2: 不一致性水平 (nP,)
        mp: 混合概率 (nP,)
    """
    if USE_CYTHON:
        X_float = np.asarray(X, dtype=np.float64)
        y_float = np.asarray(y, dtype=np.float64)
        return kridge_cython(X_float, y_float, nP, lambda_reg)
    
    nJ, nA_minus_1 = X.shape
    nA = nA_minus_1 + 1
    
    u = np.zeros((nP, nA_minus_1))
    
    # 使用岭回归初始化第一个优先级向量
    XtX = X.T @ X
    Xty = X.T @ y
    u[0, :] = np.linalg.solve(XtX + lambda_reg * np.eye(nA_minus_1), Xty)
    
    s2 = np.sum((X @ u[0, :] - y) ** 2) / nJ
    
    # 初始化后续的优先级向量
    for p in range(nP - 1):
        RSE = np.zeros((nJ, p + 1))
        for k in range(p + 1):
            RSE[:, k] = (X @ u[k, :] - y) ** 2
        
        # 找到残差平方和最大的样本
        max_rse = np.max(RSE, axis=1)
        # 选择残差较大的样本作为边界样本
        n_edge = min(nA, nJ)  # 确保不超过样本总数
        threshold = np.sort(max_rse)[-n_edge] if n_edge > 0 else np.max(max_rse)
        idx = max_rse >= threshold  # 选择残差较大的样本
        
        X_edge = X[idx, :]
        y_edge = y[idx]
        
        if X_edge.shape[0] > 0:
            XtX_edge = X_edge.T @ X_edge
            Xty_edge = X_edge.T @ y_edge
            u[p + 1, :] = np.linalg.solve(
                XtX_edge + lambda_reg * np.eye(nA_minus_1),
                Xty_edge
            )
    
    # 迭代更新优先级向量
    for _ in range(100):
        RSE = np.zeros((nJ, nP))
        for p in range(nP):
            RSE[:, p] = (X @ u[p, :] - y) ** 2
        
        # 聚类：将样本分配给残差最小的组件
        cls = np.argmin(RSE, axis=1)
        
        for p in range(nP):
            mask = cls == p
            if np.sum(mask) > 0:
                Xp = X[mask, :]
                yp = y[mask]
                XtX_p = Xp.T @ Xp
                Xty_p = Xp.T @ yp
                u[p, :] = np.linalg.solve(
                    XtX_p + lambda_reg * np.eye(nA_minus_1),
                    Xty_p
                )
    
    sigma2 = np.ones(nP) * s2
    mp = np.ones(nP) / nP  # 初始化混合概率
    
    return u, sigma2, mp
