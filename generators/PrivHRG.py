import numpy as np
import random
import math


############################################
# Laplace Noise（保持不变）
############################################

def laplace_sample(scale):
    u = random.random() - 0.5
    sign = 1 if u >= 0 else -1
    return -scale * sign * math.log(1 - 2 * abs(u))


############################################
# Graph Structure（轻量优化）
############################################

class Graph:

    def __init__(self, n):
        self.n = n
        self.adj = [set() for _ in range(n)]

    def add_edge(self, u, v):
        self.adj[u].add(v)
        self.adj[v].add(u)

    def num_nodes(self):
        return self.n


############################################
# Matrix -> Graph（保持）
############################################

def matrix_to_graph(matrix):
    n = matrix.shape[0]
    g = Graph(n)

    rows, cols = np.where(np.triu(matrix, 1) == 1)

    for u, v in zip(rows, cols):
        g.add_edge(u, v)

    return g


############################################
# Graph -> Matrix（向量优化）
############################################

def graph_to_matrix(g):
    n = g.num_nodes()
    matrix = np.zeros((n, n), dtype=np.float32)

    for u in range(n):
        for v in g.adj[u]:
            matrix[u, v] = 1

    matrix = np.maximum(matrix, matrix.T)
    return matrix


############################################
# Dendrogram Node
############################################

class DendroNode:

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

        self.vertices = []

        self.p = 0
        self.noisy_p = 0

    def is_leaf(self):
        return self.left is None and self.right is None


############################################
# Random Dendrogram（保持）
############################################

def build_random_tree(vertices):

    nodes = []

    for v in vertices:
        node = DendroNode()
        node.vertices = [v]
        nodes.append(node)

    while len(nodes) > 1:
        a = random.choice(nodes)
        nodes.remove(a)

        b = random.choice(nodes)
        nodes.remove(b)

        parent = DendroNode(a, b)
        parent.vertices = a.vertices + b.vertices

        nodes.append(parent)

    return nodes[0]


############################################
# 🚀 优化：矩阵方式计算概率
############################################

def compute_probabilities(node, adj_matrix):

    if node.is_leaf():
        return

    compute_probabilities(node.left, adj_matrix)
    compute_probabilities(node.right, adj_matrix)

    left = node.left.vertices
    right = node.right.vertices

    if len(left) == 0 or len(right) == 0:
        node.p = 0
        return

    # 🔥 核心优化：子矩阵求和（替代双循环）
    submatrix = adj_matrix[np.ix_(left, right)]
    edges = np.sum(submatrix)

    node.p = edges / (len(left) * len(right))


############################################
# 加噪（保持）
############################################

def add_noise(node, epsilon):

    if node.is_leaf():
        return

    scale = 1 / epsilon

    node.noisy_p = node.p + laplace_sample(scale)
    node.noisy_p = max(0, min(1, node.noisy_p))

    add_noise(node.left, epsilon)
    add_noise(node.right, epsilon)


############################################
# 🚀 优化：向量化采样
############################################

def sample_graph(node, n):

    g = Graph(n)

    def recurse(nd):

        if nd.is_leaf():
            return

        left = nd.left.vertices
        right = nd.right.vertices

        p = nd.noisy_p

        if len(left) > 0 and len(right) > 0:

            # 🔥 向量化随机采样
            rand_matrix = np.random.rand(len(left), len(right))
            edges = rand_matrix < p

            rows, cols = np.where(edges)

            for i, j in zip(rows, cols):
                g.add_edge(left[i], right[j])

        recurse(nd.left)
        recurse(nd.right)

    recurse(node)

    return g


############################################
# Main PrivHRG Algorithm（接口完全不变）
############################################

def privhrg_generate(adj_matrix, epsilon):

    g = matrix_to_graph(adj_matrix)

    vertices = list(range(g.num_nodes()))

    tree = build_random_tree(vertices)

    # 🔥 直接用矩阵，不再用 graph.has_edge
    compute_probabilities(tree, adj_matrix)

    add_noise(tree, epsilon)

    synthetic_graph = sample_graph(tree, g.num_nodes())

    synthetic_matrix = graph_to_matrix(synthetic_graph)
    num_edges = np.sum(synthetic_matrix) / 2
    print(f"✅ [PrivHRG] 生成完成 | epsilon={epsilon} | 节点数={g.num_nodes()} | 边数={int(num_edges)}", flush=True)

    return synthetic_matrix