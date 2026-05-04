import numpy as np
import networkx as nx
from utils import *
import pandas as pd
from sklearn import metrics
import os
import time
from PrivDPR import generate_synthetic_graph_from_adjmatrix
import comm.community_main as community


def main_function(dataset_name='Facebook', eps=[1, 2, 3], exp_num=5, save_csv=True):
    t_begin = time.time()
    data_path = './data/' + dataset_name + '.txt'
    mat0, mid = get_mat(data_path)

    cols = ['eps', 'exper', 'num_node_RE', 'num_edge_RE', 'tria_count_RE', 'avg_deg_RE',
            'deg_var_RE', 'deg_dsb_KL', 'diam_RE', 'SP_RE', 'dis_dsb_KL', 'GCC_RE',
            'ACC_RE', 'CD_NMI', 'MOD_RE', 'Ass_RE', 'evc_MAE']

    all_data = pd.DataFrame(None, columns=cols)

    mat0_graph = nx.from_numpy_array(mat0)

    mat0_par = community.best_partition(mat0_graph, epsilon_EM=9999)

    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    mat0_degree = np.sum(mat0, 0)
    mat0_avg_degree = np.mean(mat0_degree)
    mat0_degree_variance = np.var(mat0_degree, ddof=1)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree))
    mat0_dis_dist = distance_distribution(mat0_graph)
    mat0_evc = nx.eigenvector_centrality(mat0_graph, max_iter=10000)
    mat0_evc_val = np.array(list(mat0_evc.values()))
    evc_kn = np.int64(0.01 * mat0_node)
    mat0_diam = cal_diam(mat0)
    mat0_cc = nx.transitivity(mat0_graph)
    mat0_mod = community.modularity(mat0_par, mat0_graph)
    mat0_total_triangle = sum(nx.triangles(mat0_graph).values()) // 3
    mat0_acc = nx.average_clustering(mat0_graph)
    mat0_ass = nx.degree_assortativity_coefficient(mat0_graph)
    mat0_SP = average_shortest_path_length_custom(mat0_graph)

    for epsilon in eps:
        for exper in range(exp_num):
            print(f'epsilon={epsilon}, exper={exper + 1}')

            # ========== 核心修改开始 ==========
            # 1. 将无向邻接矩阵转换为有向邻接矩阵（无向边→双向有向边）
            # 先转为无向图，再转为有向图（自动拆分为双向边）
            undir_graph = nx.from_numpy_array(mat0)
            dir_graph = nx.DiGraph(undir_graph)  # 无向图转有向图（双向边）
            mat0_dir = nx.to_numpy_array(dir_graph)  # 有向图的邻接矩阵

            # 2. 调用生成函数时，指定 directed=True（适配有向图）
            mat2_dir = generate_synthetic_graph_from_adjmatrix(mat0_dir, epsilon, directed=True)

            # 3. 将生成的有向图邻接矩阵转回无向图（对称化）
            mat2 = (mat2_dir + mat2_dir.T) / 2  # 双向边合并为无向边
            mat2[mat2 > 0] = 1  # 二值化（确保邻接矩阵是0/1）
            # ========== 核心修改结束 ==========

            mat2_graph = nx.from_numpy_array(mat2)

            mat2_par = community.best_partition(mat2_graph, epsilon_EM=epsilon)

            mat2_edge = mat2_graph.number_of_edges()
            mat2_node = mat2_graph.number_of_nodes()
            mat2_total_triangle = sum(nx.triangles(mat2_graph).values()) // 3
            mat2_degree = np.sum(mat2, 0)
            mat2_avg_degree = np.mean(mat2_degree)
            mat2_degree_variance = np.var(mat2_degree, ddof=1)
            mat2_SP = average_shortest_path_length_custom(mat2_graph)
            mat2_dis_dist = distance_distribution(mat2_graph)
            mat2_mod = community.modularity(mat2_par, mat2_graph)
            mat2_cc = nx.transitivity(mat2_graph)

            mat2_deg_dist = np.bincount(np.int64(mat2_degree))
            mat2_acc = nx.average_clustering(mat2_graph)
            mat2_ass = nx.degree_assortativity_coefficient(mat2_graph)
            mat2_evc = nx.eigenvector_centrality(mat2_graph, max_iter=10000)
            mat2_evc_val = np.array(list(mat2_evc.values()))
            mat2_diam = cal_diam(mat2)

            # metrics
            data_col = [
                epsilon, exper,
                cal_rel(mat0_node, mat2_node),
                cal_rel(mat0_edge, mat2_edge),
                cal_rel(mat0_total_triangle, mat2_total_triangle),
                cal_rel(mat0_avg_degree, mat2_avg_degree),
                cal_rel(mat0_degree_variance, mat2_degree_variance),
                cal_kl(mat0_deg_dist, mat2_deg_dist),
                cal_rel(mat0_diam, mat2_diam),
                cal_rel(mat0_SP, mat2_SP),
                cal_kl(mat0_dis_dist, mat2_dis_dist),
                cal_rel(mat0_cc, mat2_cc),
                cal_rel(mat0_acc, mat2_acc),
                metrics.normalized_mutual_info_score(
                    list(mat0_par.values()), list(mat2_par.values())
                ),
                cal_rel(mat0_mod, mat2_mod),
                cal_rel(mat0_ass, mat2_ass),
                cal_MAE(mat0_evc_val, mat2_evc_val, k=evc_kn)
            ]

            all_data = all_data._append(pd.DataFrame([data_col], columns=cols))

    if save_csv:
        os.makedirs('./result', exist_ok=True)
        all_data.to_csv(f'./result/{dataset_name}_PrivDPR.csv', index=False)

    print("Done. Time:", time.time() - t_begin)


if __name__ == '__main__':
    dataset_name = 'Wiki-Vote'
    eps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9999]
    exp_num = 5
    main_function(dataset_name=dataset_name, eps=eps, exp_num=exp_num)