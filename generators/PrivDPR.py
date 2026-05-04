import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import networkx as nx
import scipy.sparse as sp
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # PyTorch中使用单GPU
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--walk_num', default=2, type=int)
parser.add_argument('--walk_len', default=16, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--n_epochs', default=3, type=int)
parser.add_argument('--n_earlystop', default=2, type=int)
parser.add_argument('--tau', default=0.5, type=float)
parser.add_argument('--delay_factor', default=0.85, type=float)
parser.add_argument('--num_of_gra', default=None, type=int)
parser.add_argument('--delta', default=0.00001, type=float)
parser.add_argument('--is_GradientClip', default=True, type=bool)
parser.add_argument('--layer_num', default=1, type=int)
parser.add_argument('--epsilon', default=0.1, type=float)
parser.add_argument('--hidden_layer_dim', default=64, type=int)
parser.add_argument('--g_clip', default=5, type=float)
parser.add_argument('--w_clip', default=1 / 8, type=float)

args = parser.parse_args()


def loadGraphFromEdgeListTxt(file_name, directed=True):
    with open(file_name, 'r') as f:
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for line in f:
            edge = line.strip().split()
            if len(edge) == 3:
                w = float(edge[2])
            else:
                w = 1.0
            G.add_edge(int(edge[0]), int(edge[1]), weight=w)
    return G


class DiGraSynModel(nn.Module):
    def __init__(self, graph, Layer_num, node_embed_init=None):
        super(DiGraSynModel, self).__init__()

        self.n_node = graph.number_of_nodes()
        self.n_edge = graph.number_of_edges()
        args.num_of_gra = self.n_node
        self.node_emd_init = node_embed_init
        self.Layer_num = Layer_num

        # 创建节点嵌入矩阵
        if self.node_emd_init is not None:
            self.node_embedding_matrix = nn.Parameter(
                torch.tensor(self.node_emd_init, dtype=torch.float32)
            )
        else:
            self.node_embedding_matrix = nn.Parameter(
                torch.randn(self.n_node, args.embedding_dim) * 0.1
            )
            nn.init.xavier_normal_(self.node_embedding_matrix)

        # 创建多层MLP的权重和偏置
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for l in range(Layer_num):
            if l == 0:
                weight = nn.Parameter(torch.randn(args.embedding_dim, args.hidden_layer_dim) * 0.1)
                bias = nn.Parameter(torch.randn(args.hidden_layer_dim) * 0.1)
                nn.init.xavier_normal_(weight)
                nn.init.zeros_(bias)  # 偏置初始化为0
            elif l == Layer_num - 1:
                weight = nn.Parameter(torch.randn(args.hidden_layer_dim, 1) * 0.1)
                bias = nn.Parameter(torch.randn(1) * 0.1)
                nn.init.xavier_normal_(weight)
                nn.init.zeros_(bias)  # 偏置初始化为0
            else:
                weight = nn.Parameter(torch.randn(args.hidden_layer_dim, args.hidden_layer_dim) * 0.1)
                bias = nn.Parameter(torch.randn(args.hidden_layer_dim) * 0.1)
                nn.init.xavier_normal_(weight)
                nn.init.zeros_(bias)  # 偏置初始化为0

            self.weights.append(weight)
            self.biases.append(bias)

        # 创建优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)

        # 将设备设置为GPU（如果可用）
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def generate_node(self, node_embedding, Layer_num):
        """生成节点分数"""
        input_tensor = node_embedding.view(-1, args.embedding_dim)
        for l in range(Layer_num):
            # 谱归一化
            weight_norm = self.spectral_norm(self.weights[l]) * args.w_clip
            layer_result = torch.sigmoid(torch.matmul(input_tensor, weight_norm) + self.biases[l])
            input_tensor = layer_result
        return layer_result

    def spectral_norm(self, w, iteration=1):
        """谱归一化实现"""
        w_shape = w.shape
        w_reshaped = w.view(-1, w_shape[-1])

        # 初始化u向量
        u = torch.randn(1, w_shape[-1], device=w.device)
        u = u / (torch.norm(u) + 1e-12)

        for i in range(iteration):
            v = torch.matmul(u, w_reshaped.t())
            v = v / (torch.norm(v) + 1e-12)

            u = torch.matmul(v, w_reshaped)
            u = u / (torch.norm(u) + 1e-12)

        sigma = torch.matmul(torch.matmul(v, w_reshaped), u.t())
        w_norm = w_reshaped / (sigma + 1e-12)

        return w_norm.view(w_shape)

    def Gau_Noise(self, tensor, sens_value):
        """生成高斯噪声用于差分隐私"""
        total_iters = args.n_epochs * math.floor(self.n_node / args.batch_size)
        sigma = math.sqrt(2 * math.log(1.25 / (args.delta / total_iters))) / args.epsilon
        noise = torch.randn_like(tensor) * sigma * sens_value
        return noise

    def random_walk(self, s, graph):
        """随机游走采样"""
        walk = []
        p = s
        while len(walk) < args.walk_len:
            if p not in graph.nodes() or len(list(graph.neighbors(p))) == 0:
                break
            neighbors = list(graph.neighbors(p))
            p = random.choice(neighbors)
            walk.append(p)
        return walk

    def random_walk_sampling(self, index, node_list, graph):
        """批量随机游走采样"""
        node_head_ids = []
        node_tail_ids = []
        node_head_outDeg = []
        node_tail_inDeg = []

        batch_nodes = node_list[index * args.batch_size: (index + 1) * args.batch_size]
        for node_id in batch_nodes:
            for k in range(args.walk_num):
                walk = self.random_walk(node_id, graph)
                for t in walk:
                    node_head_ids.append(node_id)
                    node_tail_ids.append(t)

        for head_id in node_head_ids:
            node_head_outDeg.append(graph.out_degree(head_id))
        for tail_id in node_tail_ids:
            node_tail_inDeg.append(graph.in_degree(tail_id))

        return node_head_ids, node_tail_ids, node_head_outDeg, node_tail_inDeg

    def train_step(self, head_ids, tail_ids, head_outDeg, tail_inDeg):
        """单步训练"""
        # 将数据转换为张量并移到设备
        head_ids_tensor = torch.tensor(head_ids, dtype=torch.long, device=self.device)
        tail_ids_tensor = torch.tensor(tail_ids, dtype=torch.long, device=self.device)
        head_outDeg_tensor = torch.tensor(head_outDeg, dtype=torch.float32, device=self.device)
        tail_inDeg_tensor = torch.tensor(tail_inDeg, dtype=torch.float32, device=self.device)

        # 获取节点嵌入
        node_head_embedding = self.node_embedding_matrix[head_ids_tensor]
        node_tail_embedding = self.node_embedding_matrix[tail_ids_tensor]

        # 计算节点分数
        head_node_score = self.generate_node(node_head_embedding, self.Layer_num).squeeze(-1)
        tail_node_score = self.generate_node(node_tail_embedding, self.Layer_num).squeeze(-1)

        # 计算损失
        loss_fir_term = tail_inDeg_tensor * (args.delay_factor ** 2) * \
                        (head_node_score / head_outDeg_tensor - tail_node_score / (
                                    tail_inDeg_tensor * args.delay_factor)) ** 2
        loss_sec_term = (head_node_score / head_outDeg_tensor - tail_node_score / (
                    tail_inDeg_tensor * args.delay_factor)) \
                        * 2 * args.delay_factor * (1 - args.delay_factor) / self.n_node
        loss_third_term = (1 - args.delay_factor) ** 2 / (tail_inDeg_tensor * (self.n_node ** 2))

        AsyPreser_loss = loss_fir_term + loss_sec_term + loss_third_term
        loss = AsyPreser_loss.mean()

        return loss, head_node_score, tail_node_score

    def train(self, graph):
        """训练主循环"""
        self.node_list = list(graph.nodes())
        node_count_mat = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))

        for each_epoch in range(args.n_epochs):
            num_batches = math.floor(len(self.node_list) / args.batch_size)
            random.shuffle(self.node_list)  # 打乱节点顺序

            for index in range(num_batches):
                head_ids, tail_ids, head_outDeg, tail_inDeg = self.random_walk_sampling(
                    index, self.node_list, graph
                )

                if len(head_ids) == 0 or len(tail_ids) == 0:
                    continue

                # 前向传播和计算损失
                self.optimizer.zero_grad()
                loss, head_node_score, tail_node_score = self.train_step(
                    head_ids, tail_ids, head_outDeg, tail_inDeg
                )

                # 反向传播
                loss.backward()

                # 梯度裁剪和噪声注入
                if args.is_GradientClip:
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            if "node_embedding_matrix" in name:
                                # 对节点嵌入梯度添加噪声
                                noise = self.Gau_Noise(param.grad, args.g_clip)
                                param.grad = param.grad + noise
                            # L2梯度裁剪
                            grad_norm = torch.norm(param.grad)
                            if grad_norm > args.g_clip:
                                param.grad = param.grad * (args.g_clip / (grad_norm + 1e-12))

                # 更新参数
                self.optimizer.step()

                # 计算离散输出用于更新计数矩阵
                with torch.no_grad():
                    output_upped_w = torch.matmul(self.node_embedding_matrix,
                                                  self.node_embedding_matrix.t())
                    output_discrete_batch_w = self.gumbel_softmax(output_upped_w, args.tau, hard=True)
                    discrete_batch_w_index = torch.argmax(output_discrete_batch_w, dim=1).cpu().numpy()

                # 更新计数矩阵
                for i, j in zip(head_ids, discrete_batch_w_index):
                    if i < node_count_mat.shape[0] and j < node_count_mat.shape[1]:
                        node_count_mat[i, j] += 1

                if index % 100 == 0:
                    print(
                        f'Epoch {each_epoch + 1}/{args.n_epochs}, Batch {index}/{num_batches}, Loss: {loss.item():.4f}')

        return node_count_mat

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Gumbel-Softmax采样"""
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(1, torch.argmax(y, dim=1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        return y

    def gumbel_softmax_sample(self, logits, temperature):
        """Gumbel-Softmax采样"""
        y = logits + self.sample_gumbel(logits.shape)
        return F.softmax(y / temperature, dim=-1)

    def sample_gumbel(self, shape, eps=1e-20):
        """生成Gumbel噪声"""
        U = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)


def generate_SynGraphs(SynDigraName, num_of_edge, node_count_mat, node_embeddings):
    """生成合成图"""
    sparse_node_count_mat = sp.csr_matrix(node_count_mat)
    syn_graph_adj = graph_from_scores(sparse_node_count_mat, num_of_edge)
    syn_graph = transform_adj_to_Graph(syn_graph_adj)
    saveGraphToEdgeListTxtn2v(syn_graph, SynDigraName)


def transform_adj_to_Graph(adj):
    """将邻接矩阵转换为NetworkX图"""
    n = adj.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] > 0:
                graph.add_edge(i, j, weight=adj[i, j])
    return graph


def saveGraphToEdgeListTxtn2v(graph, file_name):
    """保存图为边列表格式"""
    with open(file_name, 'w') as f:
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('%d %d\n' % (i, j))


def graph_from_scores(scores, n_edges=None):
    """从分数矩阵生成图"""
    target_g = np.zeros(scores.shape)
    scores = scores + scores.T
    scores_int = scores.toarray().copy()
    scores_int[np.diag_indices_from(scores_int)] = 0

    # 计算总度数
    degrees_int = scores_int.sum(1) + scores_int.sum(0)

    N = target_g.shape[0]
    for n in range(N):
        if degrees_int[n] == 0:
            continue

        # 计算概率并归一化
        probs = scores_int[n] / degrees_int[n]
        if probs.sum() > 0:
            probs = probs / probs.sum()
            target = np.random.choice(N, p=probs)
            target_g[n, target] = 1
            target_g[target, n] = 1

    sigmoid_embeddings = 1 / (1 + np.exp(-scores_int))
    upper_triangle_indices = np.triu_indices_from(sigmoid_embeddings, k=1)
    upper_triangle_values = sigmoid_embeddings[upper_triangle_indices]
    estimated_edge_num = np.sum(upper_triangle_values > 0.5) / 250

    diff = np.round((2 * estimated_edge_num - target_g.sum()) / 2)
    if diff > 0:
        triu = np.triu(scores_int)
        triu[target_g.nonzero()] = 0
        if triu.sum() > 0:
            triu = triu / triu.sum()
            n_possible = np.count_nonzero(triu)
            triu_ixs = np.triu_indices_from(scores_int)
            extra_edges = np.random.choice(
                triu_ixs[0].shape[0], replace=False,
                p=triu[triu_ixs], size=min(int(diff), int(n_possible))
            )
            target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
            target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    return target_g


def generate_synthetic_graph_from_adjmatrix(adjmatrix, epsilon, directed=True):
    """
    从邻接矩阵生成差分隐私保护的合成图

    Parameters:
    -----------
    adjmatrix : numpy.ndarray or scipy.sparse matrix
        输入图的邻接矩阵（可以是稠密或稀疏矩阵）
    epsilon : float
        差分隐私参数 epsilon
    directed : bool, default=True
        是否为有向图

    Returns:
    --------
    numpy.ndarray
        合成图的邻接矩阵（稠密矩阵）
    """
    # 将输入转换为 NetworkX 图
    if sp.issparse(adjmatrix):
        adjmatrix = adjmatrix.toarray()

    if directed:
        G = nx.DiGraph(adjmatrix)
    else:
        G = nx.Graph(adjmatrix)

    # 计算必要的参数
    num_of_node = G.number_of_nodes()
    num_of_edge = G.number_of_edges()

    # 设置 epsilon 参数
    original_epsilon = args.epsilon
    args.epsilon = epsilon

    # 计算层数
    M = (2 * (num_of_node - 1) * args.delay_factor ** 2 + 2 * args.delay_factor \
         + 2 * args.delay_factor * (1 - args.delay_factor) / num_of_node) * (1 + 1 / args.delay_factor)
    num_of_sampledNodePairs = args.batch_size * args.walk_num * args.walk_len
    iterNum_in_each_epoch = math.floor(num_of_node / args.batch_size)
    x = args.g_clip / (num_of_sampledNodePairs * M * args.n_epochs * iterNum_in_each_epoch)
    base = args.w_clip
    Layer_num = math.ceil(math.log(x, base) - 1)

    # 创建模型并训练
    model = DiGraSynModel(G, Layer_num)
    node_count_mat = model.train(G)
    node_embeddings = model.node_embedding_matrix.detach().cpu().numpy()

    # 生成合成图邻接矩阵
    sparse_node_count_mat = sp.csr_matrix(node_count_mat)
    syn_graph_adj = graph_from_scores(sparse_node_count_mat, num_of_edge)

    # 恢复原始的 epsilon 值
    args.epsilon = original_epsilon

    return syn_graph_adj


if __name__ == '__main__':
    dataset_name = 'Wiki-Vote'
    Alg_name = 'PrivDPR.txt'
    w_clip_values = [1 / 8]
    epsilon_values = [0.1]

    Pre_name = 'Processed_'
    train_filename = './data/' + dataset_name + '.txt'

    # 加载图
    OriGraph = loadGraphFromEdgeListTxt(train_filename, directed=True)
    num_of_edge = OriGraph.number_of_edges()
    num_of_node = OriGraph.number_of_nodes()

    # 计算层数
    M = (2 * (num_of_node - 1) * args.delay_factor ** 2 + 2 * args.delay_factor \
         + 2 * args.delay_factor * (1 - args.delay_factor) / num_of_node) * (1 + 1 / args.delay_factor)
    num_of_sampledNodePairs = args.batch_size * args.walk_num * args.walk_len
    iterNum_in_each_epoch = math.floor(num_of_node / args.batch_size)
    x = args.g_clip / (num_of_sampledNodePairs * M * args.n_epochs * iterNum_in_each_epoch)
    base = args.w_clip
    Layer_num = math.ceil(math.log(x, base) - 1)

    # 创建模型并训练
    model = DiGraSynModel(OriGraph, Layer_num)
    node_count_mat = model.train(OriGraph)
    node_embeddings = model.node_embedding_matrix.detach().cpu().numpy()
    generate_SynGraphs(Alg_name, num_of_edge, node_count_mat, node_embeddings)

    print('performing is end')