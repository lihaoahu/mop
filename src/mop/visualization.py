"""
可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from .mixture_of_priorities import MixtureOfPriorities
except ImportError:
    from mixture_of_priorities import MixtureOfPriorities


def mop_plot(u, w, sigma2):
    """
    绘制优先级堆叠条形图
    
    Args:
        u: 对数优先级比 (nC x nA-1)
        w: 混合概率 (nC,)
        sigma2: 不一致性水平 (nC,)
        
    Returns:
        fig, ax: 图形对象
    """
    nC, nA_minus_1 = u.shape
    nA = nA_minus_1 + 1
    
    # 添加最后一列（全为0，因为log(w[n]/w[n]) = 0）
    u_full = np.hstack([u, np.zeros((nC, 1))])
    
    fig, ax = plt.subplots(figsize=(8, 2), dpi=300)
    
    colors = cm.get_cmap("viridis", nA)
    centers = np.cumsum(np.concatenate([[0], w + 0.01]))[:-1]
    centers = centers + w / 2
    
    bottom = np.cumsum(np.concatenate([[0], u_full[0, :]]))
    for i in range(nA):
        ax.bar(
            centers[0],
            u_full[0, i],
            w[0],
            bottom[i],
            color=colors(i),
            label=f"$w_{i+1}$"
        )
    
    if nC >= 2:
        for c in range(1, nC):
            bottom = np.cumsum(np.concatenate([[0], u_full[c, :]]))
            for i in range(nA):
                ax.bar(
                    centers[c],
                    u_full[c, i],
                    w[c],
                    bottom[i],
                    color=colors(i)
                )
    
    right = 1 + (nC - 1) * 0.01
    ax.set_xlim([-right / 100, right])
    ax.set_ylim([0, 1])
    ax.set_xticks(centers)
    
    # 创建x轴标签
    xticklabels = []
    for c in range(nC):
        label = (
            f"$\\pi_{c+1}={round(w[c] * 100, 2)}\\%$\n"
            f"$\\sigma^2={round(sigma2[c], 2)}$"
        )
        xticklabels.append(label)
    ax.set_xticklabels(xticklabels)
    
    ax.set_yticks([0, 0.5, 1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.legend(loc=(0, 1), ncols=nA, frameon=False)
    
    plt.subplots_adjust(
        top=0.899,
        bottom=0.097,
        left=0,
        right=1,
        hspace=0.2,
        wspace=0.2
    )
    plt.tight_layout()
    
    return fig, ax


def mop_hist(w, mp, sigma2, ax=None, legend=True):
    """
    绘制优先级直方图
    
    Args:
        w: 优先级向量矩阵 (nC x nA)
        mp: 混合概率 (nC,)
        sigma2: 不一致性水平 (nC,)
        ax: matplotlib轴对象（可选）
        legend: 是否显示图例
        
    Returns:
        fig, ax: 图形对象
    """
    nC, nA = w.shape
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2))
    else:
        fig = plt.gcf()
    
    colors = cm.get_cmap("viridis", nA)
    
    sp = np.cumsum(mp)
    
    left = 0
    right = sp[0]
    range_width = right - left
    binwidth = range_width * 0.9 / nA
    
    for i in range(nA):
        ax.bar(
            left + 0.05 * range_width + (i + 1) * binwidth - 0.5 * binwidth,
            w[0, i],
            binwidth,
            0,
            color=colors(i),
            label=f"$w_{i+1}$"
        )
        ax.text(
            left + 0.05 * range_width + (i + 1) * binwidth - 0.5 * binwidth,
            w[0, i] + 0.1,
            f"{round(w[0, i], 3)}",
            ha="center",
            va="top",
            size=8
        )
    
    # 添加文本注释
    if nC > 1:
        annotation = f"$\\pi_1={round(mp[0] * 100, 2)}\\%$\n$\\sigma^2_1={round(sigma2[0], 2)}$"
    else:
        annotation = f"$\\sigma^2_1={round(sigma2[0], 2)}$"
    
    ax.text(right - 0.05 * range_width, 1, annotation, ha="right", va="top")
    
    if nC == 1:
        ax.text(0.05, 1, "Single priority", ha="left", va="top", color="red", style="oblique")
    else:
        ax.text(
            0.05, 1, f"Mixture of {nC} priorities",
            ha="left", va="top", color="red", style="oblique"
        )
    
    if nC >= 2:
        for c in range(1, nC):
            left = sp[c - 1]
            ax.plot([left, left], [0, 1], "r--", alpha=0.2)
            right = sp[c]
            range_width = right - left
            binwidth = range_width * 0.9 / nA
            
            for i in range(nA):
                ax.bar(
                    left + 0.05 * range_width + (i + 1) * binwidth - 0.5 * binwidth,
                    w[c, i],
                    binwidth,
                    0,
                    color=colors(i)
                )
                ax.text(
                    left + 0.05 * range_width + (i + 1) * binwidth - 0.5 * binwidth,
                    w[c, i] + 0.1,
                    f"{round(w[c, i], 3)}",
                    ha="center",
                    va="top",
                    size=8
                )
            
            annotation = (
                f"$\\pi_{c+1}={round(mp[c] * 100, 2)}\\%$\n"
                f"$\\sigma^2_{c+1}={round(sigma2[c], 2)}$"
            )
            ax.text(right - 0.05 * range_width, 1, annotation, ha="right", va="top")
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if nC == 1:
        ax.set_xticks([])
    else:
        ax.set_xticks(sp - 0.5 * mp)
        ax.set_xticklabels([f"Priority {c+1}" for c in range(nC)])
    
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    if legend:
        ax.legend(loc=(0, 1), ncols=nA, frameon=False)
    
    plt.subplots_adjust(
        top=0.899,
        bottom=0.097,
        left=0,
        right=1,
        hspace=0.2,
        wspace=0.2
    )
    plt.tight_layout()
    
    return fig, ax
