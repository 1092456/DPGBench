import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import numpy as np
import networkx as nx
import pandas as pd
import random
import importlib.util
import sys
import time
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


class ExactMIAAttacker:
    def __init__(self, original_graph_path: str, dpgs_module_path: str,
                 dpgs_function_name: str = 'create_bter_graph_from_adjacency_matrix'):
        self.original_graph = self.load_graph_from_txt(original_graph_path)
        self.dpgs_function = self.load_dpgs_function(dpgs_module_path, dpgs_function_name)
        self.feature_dimensions = {'node': None, 'edge': None}
        self.shadow_model_cache = {}

        self.core_args = self._create_core_args()

    def _create_core_args(self):

        class Args:
            def __init__(self):
                self.target_epochs = 50
                self.target_batch_size = 100
                self.target_learning_rate = 0.01
                self.n_shadow = 10
                self.target_n_hidden = 50
                self.target_l2_ratio = 1e-7
                self.target_model = 'nn'
                self.save_model = False
                self.attack_epochs = 30
                self.attack_batch_size = 100
                self.attack_learning_rate = 0.01
                self.attack_n_hidden = 50
                self.attack_l2_ratio = 1e-7
                self.attack_model = 'nn'
                self.target_data_size = 1000
                self.target_test_train_ratio = 0.2
                self.train_dataset = 'graph_data'
                self.target_privacy = 'no_privacy'
                self.target_dp = 'dp'
                self.target_epsilon = 0.5
                self.target_delta = 1e-5
                self.target_clipping_threshold = 1

        return Args()

    def load_graph_from_txt(self, file_path: str) -> nx.Graph:
        G = nx.Graph()
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        nodes = line.strip().split()
                        if len(nodes) >= 2:
                            G.add_edge(int(nodes[0]), int(nodes[1]))
            print(f"成功加载图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
            return G
        except Exception as e:
            print(f"加载图文件错误: {e}")
            return nx.Graph()

    def load_dpgs_function(self, module_path: str, function_name: str):
        try:
            spec = importlib.util.spec_from_file_location("dpgs_module", module_path)
            dpgs_module = importlib.util.module_from_spec(spec)
            original_argv = sys.argv
            sys.argv = [sys.argv[0]]
            spec.loader.exec_module(dpgs_module)
            sys.argv = original_argv
            print(f"成功加载DPGS函数: {function_name}")
            return getattr(dpgs_module, function_name)
        except Exception as e:
            print(f"加载DPGS函数错误: {e}")
            return lambda adj_matrix, epsilon: adj_matrix

    def graph_to_adjacency_matrix(self, graph: nx.Graph) -> Tuple[np.ndarray, List]:
        if graph.number_of_nodes() == 0:
            return np.array([]), []
        nodes = sorted(graph.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        adj_matrix = np.zeros((len(nodes), len(nodes)))
        for edge in graph.edges():
            i, j = node_index[edge[0]], node_index[edge[1]]
            adj_matrix[i][j] = adj_matrix[j][i] = 1
        return adj_matrix, nodes

    def adjacency_matrix_to_graph(self, adj_matrix: np.ndarray, nodes: List) -> nx.Graph:
        G = nx.Graph()
        if len(adj_matrix) == 0:
            return G
        n = len(nodes)
        for i in range(n):
            G.add_node(nodes[i])
            for j in range(i + 1, n):
                if adj_matrix[i][j] > 0:
                    G.add_edge(nodes[i], nodes[j])
        return G

    def generate_synthetic_graph(self, original_graph: nx.Graph, epsilon: float) -> nx.Graph:
        try:
            if original_graph.number_of_nodes() == 0:
                return nx.Graph()
            adj_matrix, nodes = self.graph_to_adjacency_matrix(original_graph)
            synthetic_adj_matrix = self.dpgs_function(adj_matrix, epsilon)
            return self.adjacency_matrix_to_graph(synthetic_adj_matrix, nodes)
        except Exception as e:
            print(f"生成合成图错误: {e}")
            return original_graph.copy()

    def sample_base_graph(self, target_element: Any, element_type: str,
                          reference_graph: nx.Graph, target_size: Tuple[int, int]) -> nx.Graph:
        n_nodes_target, n_edges_target = target_size
        all_elements = list(reference_graph.nodes()) if element_type == 'node' else list(reference_graph.edges())

        if target_element in all_elements:
            all_elements.remove(target_element)

        if not all_elements:
            return nx.Graph()

        base_size = int(0.8 * (n_nodes_target if element_type == 'node' else n_edges_target))
        sample_k = random.randint(max(1, base_size - 5), base_size + 5)

        sampled_elements = random.sample(all_elements, min(len(all_elements), sample_k))

        G_base = nx.Graph()
        if element_type == 'node':
            G_base.add_nodes_from(sampled_elements)
            for u, v in reference_graph.subgraph(sampled_elements).edges():
                G_base.add_edge(u, v)
        else:
            G_base.add_edges_from(sampled_elements)

        return G_base

    def construct_training_graph(self, target_element: Any, element_type: str,
                                 include_target: bool, reference_graph: nx.Graph) -> nx.Graph:
        target_size = (self.original_graph.number_of_nodes(), self.original_graph.number_of_edges())
        G_base = self.sample_base_graph(target_element, element_type, reference_graph, target_size)
        return self._add_target_element(G_base, target_element, element_type,reference_graph) if include_target else self._add_replacement_element(G_base,target_element,element_type,reference_graph)

    def _add_target_element(self, G_base: nx.Graph, target_element: Any,
                            element_type: str, reference_graph: nx.Graph) -> nx.Graph:
        G_R = G_base.copy()
        if element_type == 'node':
            G_R.add_node(target_element)
            if target_element in reference_graph.nodes():
                neighbors = [n for n in reference_graph.neighbors(target_element) if n in G_base.nodes()]
                for neighbor in random.sample(neighbors, min(len(neighbors), random.randint(1, 3))):
                    G_R.add_edge(target_element, neighbor)
        else:
            u, v = target_element
            G_R.add_nodes_from([u, v])
            G_R.add_edge(u, v)
        return G_R

    def _add_replacement_element(self, G_base: nx.Graph, target_element: Any,
                                 element_type: str, reference_graph: nx.Graph) -> nx.Graph:
        G_R = G_base.copy()
        if element_type == 'node':
            all_nodes = [n for n in reference_graph.nodes() if n != target_element]
            if all_nodes:
                replacement = random.choice(all_nodes)
                G_R.add_node(replacement)
                neighbors = [n for n in reference_graph.neighbors(replacement) if n in G_base.nodes()]
                for neighbor in random.sample(neighbors, min(len(neighbors), random.randint(1, 3))):
                    G_R.add_edge(replacement, neighbor)
        else:
            all_edges = [e for e in reference_graph.edges() if e != target_element]
            if all_edges:
                u, v = random.choice(all_edges)
                G_R.add_nodes_from([u, v])
                G_R.add_edge(u, v)
        return G_R

    def graph_to_features(self, graph: nx.Graph, target_element: Any, element_type: str) -> np.ndarray:
        features = []

        if element_type == 'node':
            num_nodes, num_edges = graph.number_of_nodes(), graph.number_of_edges()
            degrees = [d for _, d in graph.degree()] if num_nodes > 0 else []

            features.extend([num_nodes, num_edges])
            if degrees:
                features.extend([np.mean(degrees), np.std(degrees), np.max(degrees), np.min(degrees)])
            else:
                features.extend([0, 0, 0, 0])

            try:
                clustering_coeffs = list(nx.clustering(graph).values())
                features.append(np.mean(clustering_coeffs) if clustering_coeffs else 0)
            except:
                features.append(0)

            try:
                features.append(nx.transitivity(graph) if num_nodes >= 3 and num_edges > 0 else 0)
            except:
                features.append(0)

            if target_element in graph.nodes():
                try:
                    hop2_neighbors = set()
                    hop1 = list(graph.neighbors(target_element))
                    hop2 = []
                    for nbr in hop1:
                        hop2.extend(list(graph.neighbors(nbr)))
                    hop2_neighbors.update(hop1)
                    hop2_neighbors.update(hop2)
                    hop2_neighbors.discard(target_element)

                    subgraph_nodes = list(hop2_neighbors)
                    if len(subgraph_nodes) > 0:
                        subg = graph.subgraph(subgraph_nodes)
                        sub_edges = subg.number_of_edges()
                        max_possible = len(subgraph_nodes) * (len(subgraph_nodes) - 1) / 2
                        density_2hop = sub_edges / max_possible if max_possible > 0 else 0

                        degs = [graph.degree(n) for n in subgraph_nodes]
                        mean_deg_2hop = np.mean(degs) if degs else 0
                        std_deg_2hop = np.std(degs) if len(degs) > 1 else 0
                    else:
                        density_2hop = 0
                        mean_deg_2hop = 0
                        std_deg_2hop = 0

                    features.extend([
                        density_2hop,
                        mean_deg_2hop,
                        std_deg_2hop,
                        len(hop1),
                        len(subgraph_nodes)
                    ])
                except Exception as e:
                    print(f"2-hop 特征计算失败: {e}")
                    features.extend([0, 0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0, 0])

        else:
            u, v = target_element
            if u not in graph.nodes() or v not in graph.nodes():
                return np.array([0] * 10, dtype=np.float32)

            edge_exists = 1 if graph.has_edge(u, v) else 0
            features.append(edge_exists)

            try:
                common_neighbors = len(set(graph.neighbors(u)) & set(graph.neighbors(v)))
                features.append(common_neighbors)
            except:
                features.append(0)

            try:
                degree_product = graph.degree(u) * graph.degree(v)
                features.append(degree_product)
            except:
                features.append(0)

            try:
                edge_betweenness = nx.edge_betweenness_centrality(graph).get((u, v), 0)
                features.append(edge_betweenness)
            except:
                features.append(0)

            try:
                triangles = nx.triangles(graph, u) + nx.triangles(graph, v) if graph.has_edge(u, v) else 0
                features.append(triangles)
            except:
                features.append(0)

            try:
                jaccard = len(set(graph.neighbors(u)) & set(graph.neighbors(v))) / \
                          len(set(graph.neighbors(u)) | set(graph.neighbors(v))) if len(
                    set(graph.neighbors(u)) | set(graph.neighbors(v))) > 0 else 0
                features.append(jaccard)
            except:
                features.append(0)

            try:
                degree_diff = abs(graph.degree(u) - graph.degree(v))
                features.append(degree_diff)
            except:
                features.append(0)

            try:
                avg_clust = (nx.clustering(graph, u) + nx.clustering(graph, v)) / 2
                features.append(avg_clust)
            except:
                features.append(0)

        expected_length = 10
        if len(features) < expected_length:
            features.extend([0.0] * (expected_length - len(features)))
        elif len(features) > expected_length:
            features = features[:expected_length]

        return np.array(features, dtype=np.float32)

    def create_shadow_dataset(self, target_element: Any, element_type: str,
                              epsilon: float, reference_graph: nx.Graph, n_shadow: int = 10) -> Tuple[
        np.ndarray, np.ndarray]:
        features, labels = [], []

        for i in range(n_shadow):
            G_R_in = self.construct_training_graph(target_element, element_type, True, reference_graph)
            G_S_in = self.generate_synthetic_graph(G_R_in, epsilon)
            feat_in = self.graph_to_features(G_S_in, target_element, element_type)
            features.append(feat_in)
            labels.append(1)

            G_R_out = self.construct_training_graph(target_element, element_type, False, reference_graph)
            G_S_out = self.generate_synthetic_graph(G_R_out, epsilon)
            feat_out = self.graph_to_features(G_S_out, target_element, element_type)
            features.append(feat_out)
            labels.append(0)

        if len(features) > 0:
            X = np.vstack(features)
            unique_rows = len(np.unique(X, axis=0))
            print(f"[DEBUG] ε={epsilon}, elem={target_element}: {unique_rows}/{len(X)} 唯一特征向量")
            if unique_rows < 5:
                print("⚠️  警告：特征多样性不足！")
        else:
            X = np.zeros((0, 10), dtype=np.float32)
            labels = []

        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

    def train_shadow_models_core(self, target_element: Any, element_type: str,
                                 epsilon: float, reference_graph: nx.Graph, n_shadow: int = 10) -> Dict:

        shadow_features, shadow_labels = self.create_shadow_dataset(
            target_element, element_type, epsilon, reference_graph, n_shadow)

        if len(shadow_features) == 0:
            return {'trained': False}

        n_total = len(shadow_features)
        n_train = int(0.8 * n_total)

        train_x = shadow_features[:n_train]
        train_y = shadow_labels[:n_train]

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)

        try:
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(train_x_scaled, train_y)

            print(f"✅ 成功训练 Random Forest 影子模型 | ε={epsilon}, {len(train_x)} 样本")

            return {
                'classifier': clf,
                'scaler': scaler,
                'trained': True,
                'feature_dimension': train_x.shape[1]
            }

        except Exception as e:
            print(f"❌ 训练 Random Forest 失败: {e}")
            import traceback
            traceback.print_exc()
            return {'trained': False}

    def get_shadow_model(self, target_element: Any, element_type: str,
                         epsilon: float, reference_graph: nx.Graph) -> Dict:
        cache_key = (element_type, target_element, epsilon)

        if cache_key not in self.shadow_model_cache:
            print(f"🔥 临时禁用缓存，重新训练模型")
            return self.train_shadow_models_core(target_element, element_type, epsilon, reference_graph, n_shadow=10)
        else:
            print(f"🟠 使用缓存模型 (潜在过拟合风险): {element_type}-{target_element}, ε={epsilon}")

        return self.shadow_model_cache[cache_key]

    def predict_with_shadow_model(self, shadow_model: Dict, graph: nx.Graph, target_element: Any,
                                  element_type: str) -> Tuple[int, np.ndarray]:
        if not shadow_model['trained']:
            return 0, np.array([0.5, 0.5])

        try:
            clf = shadow_model['classifier']
            scaler = shadow_model.get('scaler', None)

            features = self.graph_to_features(graph, target_element, element_type).reshape(1, -1)
            if scaler is not None:
                features = scaler.transform(features)

            prediction = int(clf.predict(features)[0])
            pred_proba = clf.predict_proba(features)[0]

            return prediction, pred_proba

        except Exception as e:
            print(f"Random Forest 预测错误: {e}")
            return 0, np.array([0.5, 0.5])

    def execute_mia_attack(self, target_element: Any, element_type: str,
                           epsilon: float, reference_graph: nx.Graph) -> Dict:
        include_target = random.choice([True, False])
        G_R_test = self.construct_training_graph(target_element, element_type, include_target, reference_graph)
        G_S_test = self.generate_synthetic_graph(G_R_test, epsilon)

        G_X = G_S_test
        release_original = False

        shadow_model = self.get_shadow_model(target_element, element_type, epsilon, reference_graph)

        if shadow_model['trained']:
            try:
                guess, pred_proba = self.predict_with_shadow_model(shadow_model, G_X, target_element, element_type)

                actual_membership = 1 if include_target else 0
                attack_success = 1 if guess == actual_membership else 0

                correct_confidence = pred_proba[actual_membership]

                print(f"    实际成员: {actual_membership}, 预测: {guess}, 成功: {attack_success}, "
                      f"置信度(P_正确): {correct_confidence:.3f}")

                return {
                    'target_element': target_element,
                    'element_type': element_type,
                    'epsilon': epsilon,
                    'release_type': "synthetic",
                    'actual_membership': int(include_target),
                    'prediction': guess,
                    'attack_success': attack_success,
                    'correct_confidence': correct_confidence,
                    'prediction_proba': pred_proba,
                    'graph_size': (G_X.number_of_nodes(), G_X.number_of_edges())
                }

            except Exception as e:
                print(f"MIA分类器预测错误: {e}")
                return {
                    'target_element': target_element,
                    'element_type': element_type,
                    'epsilon': epsilon,
                    'release_type': "synthetic",
                    'actual_membership': 0,
                    'prediction': 0,
                    'attack_success': 0,
                    'correct_confidence': 0.5,
                    'prediction_proba': [0.5, 0.5],
                    'graph_size': (0, 0)
                }
        else:
            return {
                'target_element': target_element,
                'element_type': element_type,
                'epsilon': epsilon,
                'release_type': "synthetic",
                'actual_membership': 0,
                'prediction': 0,
                'attack_success': 0,
                'correct_confidence': 0.5,
                'prediction_proba': [0.5, 0.5],
                'graph_size': (0, 0)
            }

    def calculate_final_statistics_table(self, all_results: List[Dict]) -> pd.DataFrame:
        table_data = []

        for element_type in ['node', 'edge']:
            for epsilon in sorted(set(r['epsilon'] for r in all_results)):
                type_epsilon_results = [r for r in all_results
                                        if r['element_type'] == element_type and r['epsilon'] == epsilon]

                if not type_epsilon_results:
                    continue

                unique_targets = len(set(r['target_element'] for r in type_epsilon_results))

                in_member_results = [r for r in type_epsilon_results if r['actual_membership'] == 1]
                out_member_results = [r for r in type_epsilon_results if r['actual_membership'] == 0]

                if in_member_results:
                    tpr = np.mean([r['prediction'] for r in in_member_results])
                else:
                    tpr = 0.0

                if out_member_results:
                    fpr = np.mean([r['prediction'] for r in out_member_results])
                else:
                    fpr = 0.0

                advantage = tpr - fpr

                success_rate = np.mean([r['attack_success'] for r in type_epsilon_results])

                element_advantages = {}
                for result in type_epsilon_results:
                    element_key = result['target_element']
                    if element_key not in element_advantages:
                        element_advantages[element_key] = []
                    element_advantages[element_key].append(result['attack_success'])

                if element_advantages:
                    element_success_rates = [np.mean(successes) for successes in element_advantages.values()]
                    advantage_std = np.std(element_success_rates)
                    advantage_range = np.max(element_success_rates) - np.min(element_success_rates)
                else:
                    advantage_std = 0.0
                    advantage_range = 0.0

                table_data.append({
                    'attack_type': element_type,
                    'epsilon': epsilon,
                    'unique_targets': unique_targets,
                    'total_attacks': len(type_epsilon_results),
                    'in_member_attacks': len(in_member_results),
                    'out_member_attacks': len(out_member_results),
                    'tpr': tpr,
                    'fpr': fpr,
                    'advantage': advantage,
                    'success_rate': success_rate,
                    'advantage_std': advantage_std,
                    'advantage_range': advantage_range,
                })

        return pd.DataFrame(table_data)

    def save_final_table(self, final_table: pd.DataFrame, output_path: str = "./mia_results",
                         output_prefix: str = None):
        os.makedirs(output_path, exist_ok=True)

        if output_prefix:
            filename = f"{output_prefix}_statistics.csv"
        else:
            filename = f"mia_final_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = os.path.join(output_path, filename)
        final_table.to_csv(filepath, index=False)

        print(f"\n最终统计表格已保存到: {filepath}")
        return filepath

    def run_mia_evaluation(self, epsilon_values: List[float], reference_graph_path: str,
                           node_target_count: int = 5, edge_target_count: int = 5,
                           attacks_per_target: int = 10, output_path: str = "./mia_results",
                           output_prefix: str = None) -> Tuple[pd.DataFrame, List[Dict]]:
        import time

        os.makedirs(output_path, exist_ok=True)

        reference_graph = self.load_graph_from_txt(reference_graph_path)
        all_nodes, all_edges = list(self.original_graph.nodes()), list(self.original_graph.edges())
        node_targets = random.sample(all_nodes, min(node_target_count, len(all_nodes)))
        edge_targets = random.sample(all_edges, min(edge_target_count, len(all_edges)))

        all_results = []
        total_attacks = len(epsilon_values) * (
                len(node_targets) * attacks_per_target + len(edge_targets) * attacks_per_target)
        current_attack = 0

        start_total_time = time.time()

        print(f"\n开始MIA评估")
        print(f"Epsilon值: {epsilon_values}")
        print(f"节点目标数: {len(node_targets)}, 每个目标攻击次数: {attacks_per_target}")
        print(f"边目标数: {len(edge_targets)}, 每个目标攻击次数: {attacks_per_target}")
        print(f"总攻击次数: {total_attacks}")
        print("=" * 80)

        timing_logs = []

        for epsilon in epsilon_values:
            print(f"\n▶ Epsilon = {epsilon}")
            print("-" * 60)

            start_epsilon_time = time.time()

            node_epsilon_results = []
            node_element_success_rates = {}

            for target_idx, target_node in enumerate(node_targets):
                print(f"  节点目标 {target_idx + 1}/{len(node_targets)}: {target_node}")
                node_target_results = []
                node_element_success_rates[target_node] = []

                for attack_round in range(attacks_per_target):
                    current_attack += 1
                    result = self.execute_mia_attack(target_node, 'node', epsilon, reference_graph)
                    all_results.append(result)
                    node_target_results.append(result)
                    node_epsilon_results.append(result)
                    node_element_success_rates[target_node].append(result['attack_success'])

                    status = "✓" if result['prediction'] == 1 else "✗"
                    success_str = "成功" if result['attack_success'] == 1 else "失败"
                    actual_membership = "在" if result['actual_membership'] == 1 else "不在"
                    release_type = "原始图" if result['release_type'] == 'original' else "合成图"
                    conf = result.get('correct_confidence', 0.5)

                    print(f"    [{current_attack:3d}/{total_attacks}] 第{attack_round + 1:2d}次 | "
                          f"发布: {release_type:4s} | 实际: {actual_membership} | 预测: {status} | "
                          f"结果: {success_str} | 置信度: {conf:.3f}")

                target_success_rate = np.mean([r['attack_success'] for r in node_target_results])
                print(f"    {target_node} 平均成功率: {target_success_rate:.3f}")

            edge_epsilon_results = []
            edge_element_success_rates = {}

            for target_idx, target_edge in enumerate(edge_targets):
                edge_str = f"({target_edge[0]},{target_edge[1]})"
                print(f"  边目标 {target_idx + 1}/{len(edge_targets)}: {edge_str}")
                edge_target_results = []
                edge_element_success_rates[target_edge] = []

                for attack_round in range(attacks_per_target):
                    current_attack += 1
                    result = self.execute_mia_attack(target_edge, 'edge', epsilon, reference_graph)
                    all_results.append(result)
                    edge_target_results.append(result)
                    edge_epsilon_results.append(result)
                    edge_element_success_rates[target_edge].append(result['attack_success'])

                    status = "✓" if result['prediction'] == 1 else "✗"
                    success_str = "成功" if result['attack_success'] == 1 else "失败"
                    actual_membership = "在" if result['actual_membership'] == 1 else "不在"
                    release_type = "原始图" if result['release_type'] == 'original' else "合成图"
                    conf = result.get('correct_confidence', 0.5)

                    print(f"    [{current_attack:3d}/{total_attacks}] 第{attack_round + 1:2d}次 | "
                          f"发布: {release_type:4s} | 实际: {actual_membership} | 预测: {status} | "
                          f"结果: {success_str} | 置信度: {conf:.3f}")

                target_success_rate = np.mean([r['attack_success'] for r in edge_target_results])
                print(f"    {edge_str} 平均成功率: {target_success_rate:.3f}")

            end_epsilon_time = time.time()
            epsilon_duration = end_epsilon_time - start_epsilon_time
            timing_logs.append({
                'attack_type': 'node+edge',
                'epsilon': epsilon,
                'phase': 'evaluation',
                'duration_seconds': epsilon_duration
            })

            if node_element_success_rates:
                node_avg_success_rates = [np.mean(successes) for successes in node_element_success_rates.values()]
                node_avg_success = np.mean(node_avg_success_rates)
                node_max_success = np.max(node_avg_success_rates)
                node_min_success = np.min(node_avg_success_rates)
                print(f"  节点攻击成功率统计:")
                print(
                    f"    平均成功率: {node_avg_success:.3f}, 最大: {node_max_success:.3f}, 最小: {node_min_success:.3f}")

            if edge_element_success_rates:
                edge_avg_success_rates = [np.mean(successes) for successes in edge_element_success_rates.values()]
                edge_avg_success = np.mean(edge_avg_success_rates)
                edge_max_success = np.max(edge_avg_success_rates)
                edge_min_success = np.min(edge_avg_success_rates)
                print(f"  边攻击成功率统计:")
                print(
                    f"    平均成功率: {edge_avg_success:.3f}, 最大: {edge_max_success:.3f}, 最小: {edge_min_success:.3f}")

            print(f"\n📊 攻击优势双指标对比 (ε={epsilon}):")

            results_eps = [r for r in all_results if r['epsilon'] == epsilon]

            in_member_results = [r for r in results_eps if r['actual_membership'] == 1]
            out_member_results = [r for r in results_eps if r['actual_membership'] == 0]

            tpr = np.mean([r['prediction'] for r in in_member_results]) if in_member_results else 0.0
            fpr = np.mean([r['prediction'] for r in out_member_results]) if out_member_results else 0.0
            classic_advantage = tpr - fpr

            avg_conf_in = np.mean([r['correct_confidence'] for r in in_member_results]) if in_member_results else 0.5
            avg_conf_out = np.mean([r['correct_confidence'] for r in out_member_results]) if out_member_results else 0.5
            confidence_advantage = (avg_conf_in + avg_conf_out) / 2.0

            print(f"Classic Advantage: {classic_advantage:8.3f}")
            print(f"Confidence Advantage: {confidence_advantage:8.3f}")

            print(f"  🔹 本组 (ε={epsilon}) 耗时: {epsilon_duration:.2f} 秒")

        end_total_time = time.time()
        total_duration = end_total_time - start_total_time

        print("\n" + "=" * 80)
        print(f"MIA攻击评估完成")
        print(f"📌 总耗时: {total_duration:.2f} 秒 ({total_duration / 60:.2f} 分钟)")
        print("=" * 80)

        final_table = self.calculate_final_statistics_table(all_results)

        if not final_table.empty:
            final_table['_total_execution_time_s'] = total_duration
            final_table['_avg_attack_time_s'] = total_duration / max(len(all_results), 1)

        results_file = self.save_final_table(final_table, output_path, output_prefix)

        timing_df = pd.DataFrame(timing_logs)
        if output_prefix:
            timing_file = os.path.join(output_path, f"{output_prefix}_timing.csv")
        else:
            timing_file = os.path.join(output_path, f"timing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        timing_df.to_csv(timing_file, index=False)
        print(f"⏱️  时间日志已保存至: {timing_file}")

        print("最终统计表格:")
        print(final_table.to_string(index=False))

        return final_table, all_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='成员推断攻击(MIA)')

    parser.add_argument('--original_graph', type=str, required=True,
                        help='原始图文件路径')
    parser.add_argument('--reference_graph', type=str, required=True,
                        help='参考图文件路径')
    parser.add_argument('--dpgs_module', type=str, required=True,
                        help='DPGS模块文件路径')
    parser.add_argument('--dpgs_function', type=str, required=True,
                        help='DPGS函数名称')

    parser.add_argument('--node_targets', type=int, required=True,
                        help='节点目标数量')
    parser.add_argument('--edge_targets', type=int, required=True,
                        help='边目标数量')
    parser.add_argument('--attacks_per_target', type=int, required=True,
                        help='每个目标的攻击次数')
    parser.add_argument('--epsilon_values', type=str, required=True,
                        help='隐私预算列表，用逗号分隔，例如: 0.01,1,999')

    parser.add_argument('--output_prefix', type=str, required=True,
                        help='输出文件前缀，用于生成带属性的文件名')

    args = parser.parse_args()

    epsilon_values = [float(x.strip()) for x in args.epsilon_values.split(',')]

    output_path = "./mia_results"
    os.makedirs(output_path, exist_ok=True)

    attacker = ExactMIAAttacker(
        original_graph_path=args.original_graph,
        dpgs_module_path=args.dpgs_module,
        dpgs_function_name=args.dpgs_function
    )

    final_table, all_results = attacker.run_mia_evaluation(
        epsilon_values=epsilon_values,
        reference_graph_path=args.reference_graph,
        node_target_count=args.node_targets,
        edge_target_count=args.edge_targets,
        attacks_per_target=args.attacks_per_target,
        output_path=output_path,
        output_prefix=args.output_prefix
    )