from utils import *
from scipy.optimize import minimize
from sklearn import metrics
import pandas as pd
import os
import networkx as nx
import numpy as np
import math
import time
from joblib import Parallel, delayed

# =========================
# CUDA配置（保留GPU限制）
# =========================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    import torch
    torch.backends.cudnn.enabled = False
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except:
    DEVICE = 'cpu'

GPU_THRESHOLD = 1500


# =========================
# Triangle Count（保留自适应）
# =========================
def triangle_count_fast(adj):
    n = adj.shape[0]

    if DEVICE == 'cuda' and n <= GPU_THRESHOLD:
        try:
            A = torch.from_numpy(adj).float().to(DEVICE)
            if torch.isnan(A).any():
                raise ValueError("NaN detected in adjacency matrix")

            A3 = A @ A @ A
            return int(torch.trace(A3).item() // 6)

        except Exception as e:
            print("⚠️ GPU fallback:", e)

    if n <= 4000:
        A3 = adj @ adj @ adj
        return int(np.trace(A3) // 6)

    G = nx.from_numpy_array(adj)
    return sum(nx.triangles(G).values()) // 3


def triangle_count(G):
    return triangle_count_fast(nx.to_numpy_array(G))


# =========================
# DP机制
# =========================
def laplace(value, sensitivity, epsilon):
    return value + np.random.laplace(0, sensitivity / epsilon)


def laplace_mechanism(data, epsilon):
    noise = np.random.laplace(0, 1.0 / epsilon, size=data.shape)
    return np.maximum(data + noise, 0)


# =========================
# Local Sensitivity（O(n)）
# =========================
def cal_local_sensitivity(G):
    triangles_dict = nx.triangles(G)
    max_tri = max(triangles_dict.values())

    if max_tri == 0:
        return 1

    return max_tri


def sensitivity(beta, local_sensitivity):
    return local_sensitivity * np.exp(-beta)


# =========================
# Degree处理
# =========================
def sorted_degree_vector(G):
    return np.sort(np.array([d for _, d in G.degree()]))


# =========================
# compute_s_bar
# =========================
def compute_s_bar(s_tilde):
    n = len(s_tilde)
    s_bar = np.zeros(n)
    J = []

    def avg(l, r):
        if l > r:
            return float('inf')
        return np.mean(s_tilde[l:r + 1])

    J.append(n - 1)

    for k in range(n - 1, -1, -1):
        j_star = k
        while J:
            j = J[-1]
            if avg(j_star + 1, j) <= avg(k, j_star):
                j_star = J.pop()
            else:
                break
        J.append(j_star)

    b = 0
    while J:
        j_star = J.pop()
        if b <= j_star:
            s_bar[b:j_star + 1] = avg(b, j_star)
        b = j_star + 1

    return np.round(s_bar).astype(int)


# =========================
# ⭐ 新增：SKG采样函数（替代Kronecker）
# =========================
def skg_sample_graph(n, E_target, a, b, c, k):
    """
    使用R-MAT方式逐边采样，避免Kronecker矩阵爆炸
    """
    edges = set()

    # 概率归一化
    total = a + 2*b + c
    a_, b_, c_ = a/total, b/total, c/total

    probs = [a_, b_, b_, c_]  # 四象限

    while len(edges) < E_target:
        i = 0
        j = 0
        step = n // 2

        for _ in range(k):
            r = np.random.rand()

            if r < probs[0]:  # 左上
                pass
            elif r < probs[0] + probs[1]:  # 右上
                j += step
            elif r < probs[0] + probs[1] + probs[2]:  # 左下
                i += step
            else:  # 右下
                i += step
                j += step

            step = max(step // 2, 1)

        if i != j:
            u, v = min(i, j), max(i, j)
            edges.add((u, v))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    return G


# =========================
# DP-SKG（修改版）
# =========================
def DP_SKG(G, epsilon, local_sensitivity=None):

    if local_sensitivity is None:
        local_sensitivity = cal_local_sensitivity(G)

    epsilon1 = epsilon / 2
    epsilon2 = epsilon / 2

    sorted_degrees = sorted_degree_vector(G)
    noisy_array = laplace_mechanism(sorted_degrees, epsilon1)
    noisy_degree = compute_s_bar(noisy_array)

    if np.isnan(noisy_degree).any():
        raise ValueError("NaN detected in noisy_degree")

    E_tilde = int(max(1, 0.5 * np.sum(noisy_degree)))
    H_tilde = 0.5 * np.sum(np.maximum(noisy_degree * (noisy_degree - 1), 1))
    T_tilde = (1 / 6) * np.sum(np.maximum(sorted_degrees * (noisy_degree - 1) * (noisy_degree - 2), 1))

    if E_tilde <= 0 or H_tilde <= 0 or T_tilde <= 0:
        raise ValueError("Invalid statistics encountered")

    delta = 0.01
    beta = epsilon2 / (2 * math.log(2 / delta))
    beta = math.floor(beta * 10) / 10

    ss_beta_f = sensitivity(beta, local_sensitivity)

    tri = triangle_count(G)
    delta_tilde = laplace(tri, 2 * ss_beta_f, epsilon2)

    while delta_tilde <= 0:
        delta_tilde = laplace(tri, 2 * ss_beta_f, epsilon2)

    n = G.number_of_nodes()
    k = int(np.floor(np.log2(n)))

    def objective(params):
        a, b, c = params

        if not (0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1):
            return 1e10

        try:
            E = 0.5 * ((a + 2*b + c)**k - (a + c)**k)
            H = 0.5 * (((a + b)**2 + (b + c)**2)**k)
            T = (1/6) * (((a + b)**3 + (b + c)**3)**k)
            delta_val = (1/6) * ((a**3 + c**3)**k)

            return (
                ((E - E_tilde)**2)/(E_tilde**2) +
                ((H - H_tilde)**2)/(H_tilde**2) +
                ((T - T_tilde)**2)/(T_tilde**2) +
                ((delta_val - delta_tilde)**2)/(delta_tilde**2)
            )
        except:
            return 1e10

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: 1 - x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: 1 - x[1]},
        {'type': 'ineq', 'fun': lambda x: x[2]},
        {'type': 'ineq', 'fun': lambda x: 1 - x[2]},
        {'type': 'ineq', 'fun': lambda x: x[0] - x[2]},
    ]

    def optimize_once():
        init = np.random.rand(3)
        return minimize(objective, init, method='COBYLA', constraints=constraints)

    results = Parallel(n_jobs=-1)(
        delayed(optimize_once)() for _ in range(50)  # ⭐ 降低次数避免卡死
    )

    best = min(results, key=lambda r: r.fun)
    a, b, c = best.x

    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
        raise ValueError("Optimization failed")

    # =========================
    # ⭐ 核心替换：采样代替Kronecker
    # =========================
    G_syn = skg_sample_graph(n, E_tilde, a, b, c, k)

    return G_syn


# =========================
# ❗ 不允许修改
# =========================
def dp_skg_from_matrix(adj_matrix: np.ndarray, epsilon: float):
    n = len(adj_matrix)
    G = nx.Graph()

    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j)

    dp_graph = DP_SKG(G, epsilon)

    result_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if dp_graph.has_edge(i, j):
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1

    return result_matrix


# =========================
# 主函数
# =========================
def main_function(dataset_name='Wiki-Vote', eps=[1,2,3], exp_num=3):

    mat0, _ = get_mat('./data/' + dataset_name + '.txt')
    G = nx.from_numpy_array(mat0)

    local_sen = cal_local_sensitivity(G)

    for epsilon in eps:
        for i in range(exp_num):
            print(f"eps={epsilon}, exp={i}")
            _ = DP_SKG(G, epsilon, local_sen)


if __name__ == '__main__':
    main_function()