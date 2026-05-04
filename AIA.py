import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import random
import pickle
import sys
import os
from typing import Dict, List, Tuple, Any, Optional, Callable
import importlib.util
import argparse
from datetime import datetime


class StructuralAttributeIAttacker:

    def __init__(self,
                 original_graph_path: str,
                 dpgs_module_path: str,
                 dpgs_function_name: str = 'priv_graph'):

        self.original_graph = self.load_graph_from_txt(original_graph_path)
        self.dpgs_function = self.load_dpgs_function(dpgs_module_path, dpgs_function_name)

        print(f"原始图加载完成: {self.original_graph.number_of_nodes()} 节点, "
              f"{self.original_graph.number_of_edges()} 边")

        self.results = {}

        self.sensitive_attributes = [
            'degree',
            'core_number',
            'local_density',
            'neighbor_avg_degree',
            'triangles'
        ]

        self.edge_sensitive_attributes = [
            'edge_weight',
            'common_neighbors',
            'jaccard_coefficient',
            'preferential_attachment',
            'shortest_path_length'
        ]

        self.attribute_ranges = {}
        self.attribute_types = {}
        self._initialize_attribute_properties(self.original_graph)

        self._initialize_edge_attribute_properties(self.original_graph)

    def _initialize_attribute_properties(self, graph: nx.Graph):

        print("\n计算节点属性取值范围和类型...")

        for attr in self.sensitive_attributes:
            values = []

            if attr == 'degree':
                values = [graph.degree(node) for node in graph.nodes()]
                attr_type = 'discrete'
            elif attr == 'core_number':
                try:
                    from networkx.algorithms.core import core_number
                    core_nums = core_number(graph)
                    values = [core_nums.get(node, 0) for node in graph.nodes()]
                except:
                    values = [len(list(graph.neighbors(node))) // 2 for node in graph.nodes()]
                attr_type = 'discrete'
            elif attr == 'local_density':
                for node in graph.nodes():
                    neighbors = list(graph.neighbors(node))
                    if neighbors:
                        local_nodes = [node] + neighbors
                        subgraph = graph.subgraph(local_nodes)
                        possible_edges = len(local_nodes) * (len(local_nodes) - 1) / 2
                        actual_edges = subgraph.number_of_edges()
                        density = actual_edges / possible_edges if possible_edges > 0 else 0.0
                    else:
                        density = 0.0
                    values.append(density)
                attr_type = 'continuous'
            elif attr == 'neighbor_avg_degree':
                for node in graph.nodes():
                    neighbors = list(graph.neighbors(node))
                    if neighbors:
                        neighbor_degrees = [graph.degree(n) for n in neighbors]
                        avg_degree = np.mean(neighbor_degrees)
                    else:
                        avg_degree = 0.0
                    values.append(avg_degree)
                attr_type = 'continuous'
            elif attr == 'triangles':
                try:
                    triangles_dict = nx.triangles(graph)
                    values = [triangles_dict.get(node, 0) for node in graph.nodes()]
                except:
                    values = [0 for _ in graph.nodes()]
                attr_type = 'discrete'

            if values:
                unique_values = set(values)
                integer_count = sum(1 for v in values if abs(v - round(v)) < 1e-10)
                if integer_count / len(values) > 0.8:
                    attr_type = 'discrete'

                min_val = np.min(values)
                max_val = np.max(values)
                range_length = max_val - min_val

                if range_length < 1e-10:
                    range_length = 1.0

                self.attribute_ranges[attr] = {
                    'min': min_val,
                    'max': max_val,
                    'range': range_length,
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 0
                }

                self.attribute_types[attr] = attr_type

                print(f"  属性 '{attr}': 类型={attr_type}, 范围=[{min_val:.4f}, {max_val:.4f}], "
                      f"长度={range_length:.4f}, 均值={np.mean(values):.4f}")
            else:
                self.attribute_ranges[attr] = {
                    'min': 0.0,
                    'max': 1.0,
                    'range': 1.0,
                    'mean': 0.5,
                    'std': 0.2
                }
                if attr in ['degree', 'core_number', 'triangles']:
                    self.attribute_types[attr] = 'discrete'
                else:
                    self.attribute_types[attr] = 'continuous'
                print(f"  属性 '{attr}': 类型={self.attribute_types[attr]}, 使用默认范围 [0.0, 1.0]")

    def _initialize_edge_attribute_properties(self, graph: nx.Graph):

        print("\n计算边属性取值范围和类型...")

        if graph.number_of_edges() == 0:
            print("  图没有边，使用默认范围")
            for attr in self.edge_sensitive_attributes:
                self.attribute_ranges[attr] = {
                    'min': 0.0,
                    'max': 1.0,
                    'range': 1.0,
                    'mean': 0.5,
                    'std': 0.2
                }
                self.attribute_types[attr] = 'continuous'
            return

        for attr in self.edge_sensitive_attributes:
            values = []

            if attr == 'edge_weight':
                values = [1.0 for _ in graph.edges()]
                attr_type = 'discrete'
            elif attr == 'common_neighbors':
                for u, v in graph.edges():
                    common = len(set(graph.neighbors(u)) & set(graph.neighbors(v)))
                    values.append(common)
                attr_type = 'discrete'
            elif attr == 'jaccard_coefficient':
                for u, v in graph.edges():
                    neighbors_u = set(graph.neighbors(u))
                    neighbors_v = set(graph.neighbors(v))
                    union = len(neighbors_u | neighbors_v)
                    if union > 0:
                        jaccard = len(neighbors_u & neighbors_v) / union
                    else:
                        jaccard = 0.0
                    values.append(jaccard)
                attr_type = 'continuous'
            elif attr == 'preferential_attachment':
                for u, v in graph.edges():
                    pa = graph.degree(u) * graph.degree(v)
                    values.append(pa)
                attr_type = 'continuous'
            elif attr == 'shortest_path_length':
                for u, v in graph.edges():
                    values.append(1.0)
                attr_type = 'discrete'

            if values:
                min_val = np.min(values)
                max_val = np.max(values)
                range_length = max_val - min_val

                if range_length < 1e-10:
                    range_length = 1.0

                self.attribute_ranges[attr] = {
                    'min': min_val,
                    'max': max_val,
                    'range': range_length,
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 0
                }
                self.attribute_types[attr] = attr_type

                print(f"  边属性 '{attr}': 类型={attr_type}, 范围=[{min_val:.4f}, {max_val:.4f}], "
                      f"长度={range_length:.4f}, 均值={np.mean(values):.4f}")

    def load_graph_from_txt(self, file_path: str) -> nx.Graph:
        G = nx.Graph()
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        nodes = line.strip().split()
                        if len(nodes) >= 2:
                            G.add_edge(int(nodes[0]), int(nodes[1]))
            return G
        except Exception as e:
            print(f"加载图文件错误: {e}")
            return nx.Graph()

    def load_dpgs_function(self, module_path: str, function_name: str) -> Callable:
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
            return self._create_fallback_synthetic_graph(original_graph, epsilon)

    def _create_fallback_synthetic_graph(self, graph: nx.Graph, epsilon: float) -> nx.Graph:
        G_synthetic = nx.Graph()

        for node in graph.nodes():
            G_synthetic.add_node(node)

        original_edges = list(graph.edges())

        if epsilon < 0.1:
            edge_ratio = 0.1
        elif epsilon < 1.0:
            edge_ratio = 0.3
        elif epsilon < 10.0:
            edge_ratio = 0.6
        else:
            edge_ratio = 0.9

        num_edges_to_add = int(len(original_edges) * edge_ratio)

        if num_edges_to_add > 0:
            edges_to_add = random.sample(original_edges, min(num_edges_to_add, len(original_edges)))
            for u, v in edges_to_add:
                G_synthetic.add_edge(u, v)

        print(f"回退合成图: {G_synthetic.number_of_nodes()} 节点, {G_synthetic.number_of_edges()} 边")
        return G_synthetic

    def _extract_all_features_from_graph(self, graph: nx.Graph, node: int) -> Tuple[Dict[str, float], Dict[str, float]]:

        if not graph.has_node(node):
            return {}, {}

        all_features = {}

        all_features['degree'] = float(graph.degree(node))

        try:
            from networkx.algorithms.core import core_number
            core_nums = core_number(graph)
            all_features['core_number'] = float(core_nums.get(node, 0))
        except:
            all_features['core_number'] = float(len(list(graph.neighbors(node))) // 2)

        neighbors = list(graph.neighbors(node))
        if neighbors:
            local_nodes = [node] + neighbors
            subgraph = graph.subgraph(local_nodes)
            possible_edges = len(local_nodes) * (len(local_nodes) - 1) / 2
            actual_edges = subgraph.number_of_edges()
            all_features['local_density'] = float(actual_edges / possible_edges) if possible_edges > 0 else 0.0
        else:
            all_features['local_density'] = 0.0

        if neighbors:
            neighbor_degrees = [graph.degree(n) for n in neighbors]
            all_features['neighbor_avg_degree'] = float(np.mean(neighbor_degrees))
        else:
            all_features['neighbor_avg_degree'] = 0.0

        try:
            triangles = nx.triangles(graph, node)
            all_features['triangles'] = float(triangles)
        except:
            triangles_count = 0
            neighbor_set = set(neighbors)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if graph.has_edge(neighbors[i], neighbors[j]):
                        triangles_count += 1
            all_features['triangles'] = float(triangles_count)

        if len(neighbors) >= 2:
            neighbor_degrees = [graph.degree(n) for n in neighbors]
            all_features['neighbor_degree_std'] = float(np.std(neighbor_degrees))
        else:
            all_features['neighbor_degree_std'] = 0.0

        if neighbors:
            neighbor_degrees = [graph.degree(n) for n in neighbors]
            all_features['neighbor_max_degree'] = float(max(neighbor_degrees))
            all_features['neighbor_min_degree'] = float(min(neighbor_degrees))
        else:
            all_features['neighbor_max_degree'] = 0.0
            all_features['neighbor_min_degree'] = 0.0

        try:
            second_neighbors = set()
            for n in neighbors:
                second_neighbors.update(list(graph.neighbors(n)))
            second_neighbors.discard(node)
            all_features['second_neighbor_count'] = float(len(second_neighbors))
        except:
            all_features['second_neighbor_count'] = 0.0

        if neighbors:
            neighbor_connections = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if graph.has_edge(neighbors[i], neighbors[j]):
                        neighbor_connections += 1
            possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
            all_features['neighbor_connection_ratio'] = float(
                neighbor_connections / possible_connections) if possible_connections > 0 else 0.0
        else:
            all_features['neighbor_connection_ratio'] = 0.0

        sensitive_candidates = {}
        for attr in self.sensitive_attributes:
            if attr in all_features:
                sensitive_candidates[attr] = all_features[attr]

        return all_features, sensitive_candidates

    def _extract_edge_features_from_graph(self, graph: nx.Graph, edge: Tuple[int, int]) -> Tuple[
        Dict[str, float], Dict[str, float]]:

        u, v = edge
        if not graph.has_node(u) or not graph.has_node(v):
            return {}, {}

        all_features = {}

        all_features['edge_weight'] = 1.0 if graph.has_edge(u, v) else 0.0

        neighbors_u = set(graph.neighbors(u)) if graph.has_node(u) else set()
        neighbors_v = set(graph.neighbors(v)) if graph.has_node(v) else set()
        common_neighbors = len(neighbors_u & neighbors_v)
        all_features['common_neighbors'] = float(common_neighbors)

        union = len(neighbors_u | neighbors_v)
        if union > 0:
            jaccard = len(neighbors_u & neighbors_v) / union
        else:
            jaccard = 0.0
        all_features['jaccard_coefficient'] = float(jaccard)

        deg_u = graph.degree(u) if graph.has_node(u) else 0
        deg_v = graph.degree(v) if graph.has_node(v) else 0
        pa = deg_u * deg_v
        all_features['preferential_attachment'] = float(pa)

        try:
            if graph.has_edge(u, v):
                sp = 1.0
            else:
                try:
                    sp = nx.shortest_path_length(graph, source=u, target=v)
                    sp = float(sp)
                except:
                    sp = 3.0
        except:
            sp = 3.0
        all_features['shortest_path_length'] = float(sp)

        try:
            betweenness = 0.0
            all_features['edge_betweenness'] = betweenness
        except:
            all_features['edge_betweenness'] = 0.0

        sensitive_candidates = {}
        for attr in self.edge_sensitive_attributes:
            if attr in all_features:
                sensitive_candidates[attr] = all_features[attr]

        return all_features, sensitive_candidates

    def _select_sensitive_attribute(self, sensitive_candidates: Dict[str, float], is_edge: bool = False) -> Tuple[
        str, float]:

        if not sensitive_candidates:
            if is_edge:
                return 'common_neighbors', 0.0
            return 'degree', 0.0

        selected_attr = random.choice(list(sensitive_candidates.keys()))
        return selected_attr, sensitive_candidates[selected_attr]

    def _calculate_normalized_error(self, predicted_value: float, true_value: float,
                                    selected_attr: str) -> Tuple[float, float]:

        if selected_attr in self.attribute_ranges:
            attr_range = self.attribute_ranges[selected_attr]['range']

            if attr_range < 1e-10:
                attr_range = 1.0

            absolute_error = abs(predicted_value - true_value)

            normalized_error = absolute_error / attr_range

            return absolute_error, normalized_error
        else:
            absolute_error = abs(predicted_value - true_value)
            normalized_error = absolute_error / max(abs(true_value), 1e-8)
            return absolute_error, normalized_error

    def _get_threshold_for_attribute(self, selected_attr: str) -> float:
        if selected_attr in self.attribute_types:
            if self.attribute_types[selected_attr] == 'discrete':
                return 0.0
            else:
                return 0.05
        else:
            return 0.05

    def _generate_fake_all_features(self, target_node: int, reference_graph: nx.Graph) -> Dict[str, float]:

        all_nodes = list(reference_graph.nodes())
        if target_node in all_nodes:
            all_nodes.remove(target_node)

        if not all_nodes:
            fake_features = {}
            for attr in self.sensitive_attributes:
                if attr == 'degree':
                    fake_features[attr] = 0.0
                elif attr == 'core_number':
                    fake_features[attr] = 0.0
                elif attr == 'local_density':
                    fake_features[attr] = 0.0
                elif attr == 'neighbor_avg_degree':
                    fake_features[attr] = 0.0
                elif attr == 'triangles':
                    fake_features[attr] = 0.0
            for i in range(10):
                fake_features[f'feature_{i}'] = 0.0
            return fake_features

        template_node = random.choice(all_nodes)

        template_all_features, _ = self._extract_all_features_from_graph(reference_graph, template_node)

        if not template_all_features:
            fake_features = {}
            for attr in self.sensitive_attributes:
                if attr == 'degree':
                    fake_features[attr] = float(random.randint(1, 10))
                elif attr == 'core_number':
                    fake_features[attr] = float(random.randint(0, 5))
                elif attr == 'local_density':
                    fake_features[attr] = random.uniform(0.0, 0.5)
                elif attr == 'neighbor_avg_degree':
                    fake_features[attr] = float(random.randint(1, 10))
                elif attr == 'triangles':
                    fake_features[attr] = float(random.randint(0, 5))
            for i in range(10):
                fake_features[f'feature_{i}'] = random.random()
            return fake_features

        return template_all_features.copy()

    def _generate_fake_edge_features(self, target_edge: Tuple[int, int], reference_graph: nx.Graph) -> Dict[str, float]:

        all_edges = list(reference_graph.edges())
        if target_edge in all_edges:
            all_edges.remove(target_edge)

        if not all_edges:
            fake_features = {}
            for attr in self.edge_sensitive_attributes:
                if attr == 'edge_weight':
                    fake_features[attr] = 1.0
                elif attr == 'common_neighbors':
                    fake_features[attr] = 0.0
                elif attr == 'jaccard_coefficient':
                    fake_features[attr] = 0.0
                elif attr == 'preferential_attachment':
                    u, v = target_edge
                    deg_u = reference_graph.degree(u) if reference_graph.has_node(u) else 0
                    deg_v = reference_graph.degree(v) if reference_graph.has_node(v) else 0
                    fake_features[attr] = float(deg_u * deg_v)
                elif attr == 'shortest_path_length':
                    fake_features[attr] = 1.0
            return fake_features

        template_edge = random.choice(all_edges)

        template_all_features, _ = self._extract_edge_features_from_graph(reference_graph, template_edge)

        if not template_all_features:
            return self._generate_fake_edge_features(target_edge, reference_graph)

        return template_all_features.copy()

    def _get_non_sensitive_features(self, all_features: Dict[str, float], sensitive_attr: str) -> Dict[str, float]:

        non_sensitive_features = all_features.copy()
        if sensitive_attr in non_sensitive_features:
            del non_sensitive_features[sensitive_attr]
        return non_sensitive_features

    def _sample_training_graph(self, reference_graph: nx.Graph, target_node: int,
                               include_target: bool = True) -> nx.Graph:

        all_nodes = list(reference_graph.nodes())

        if include_target:
            other_nodes = [n for n in all_nodes if n != target_node]
        else:
            other_nodes = [n for n in all_nodes if n != target_node]

        if not other_nodes:
            return nx.Graph()

        sample_size = int(len(all_nodes) * random.uniform(0.6, 0.8))
        sample_size = max(10, min(sample_size, len(other_nodes)))

        sampled_other = random.sample(other_nodes, sample_size)

        G_train = nx.Graph()

        if include_target:
            G_train.add_node(target_node)
            train_nodes = [target_node] + sampled_other
        else:
            train_nodes = sampled_other

        G_train.add_nodes_from(train_nodes)

        for u, v in reference_graph.subgraph(train_nodes).edges():
            G_train.add_edge(u, v)

        return G_train

    def _sample_training_graph_for_edge(self, reference_graph: nx.Graph, target_edge: Tuple[int, int],
                                        include_target: bool = True) -> nx.Graph:

        u, v = target_edge
        all_nodes = list(reference_graph.nodes())

        nodes_to_sample = [u, v]
        other_nodes = [n for n in all_nodes if n not in [u, v]]

        if not other_nodes:
            return nx.Graph()

        sample_size = int(len(all_nodes) * random.uniform(0.6, 0.8))
        sample_size = max(5, min(sample_size, len(other_nodes)))

        sampled_other = random.sample(other_nodes, sample_size)

        G_train = nx.Graph()
        train_nodes = nodes_to_sample + sampled_other
        G_train.add_nodes_from(train_nodes)

        for u_node, v_node in reference_graph.subgraph(train_nodes).edges():
            G_train.add_edge(u_node, v_node)

        if not include_target and G_train.has_edge(u, v):
            G_train.remove_edge(u, v)

        return G_train

    def _execute_attribute_inference_attack_with_st(self, target_node: int, epsilon: float,
                                                    reference_graph: nx.Graph, st_value: int) -> Dict[str, Any]:

        target_all_features_real, sensitive_candidates = self._extract_all_features_from_graph(reference_graph,
                                                                                               target_node)

        if not sensitive_candidates:
            return {
                'success': False,
                'target_node': target_node,
                'target_type': 'node',
                'epsilon': epsilon,
                'st': st_value,
                'error': '无法提取目标节点的敏感属性候选'
            }

        selected_sensitive_attr, true_sensitive_value = self._select_sensitive_attribute(sensitive_candidates)

        if selected_sensitive_attr in self.attribute_ranges:
            attr_range_info = self.attribute_ranges[selected_sensitive_attr]
            attr_type = self.attribute_types.get(selected_sensitive_attr, 'continuous')
            threshold = self._get_threshold_for_attribute(selected_sensitive_attr)

            print(f"      [敏感属性选择] 本次攻击选择 '{selected_sensitive_attr}' 作为敏感属性")
            print(f"                    类型: {attr_type}, 阈值: {threshold}, "
                  f"真实值: {true_sensitive_value:.4f}, "
                  f"取值范围: [{attr_range_info['min']:.4f}, {attr_range_info['max']:.4f}]")
        else:
            print(
                f"      [敏感属性选择] 本次攻击选择 '{selected_sensitive_attr}' 作为敏感属性，真实值: {true_sensitive_value:.4f}")

        G_R = self._sample_training_graph(reference_graph, target_node, include_target=(st_value == 1))

        if G_R.number_of_nodes() < 10:
            return {
                'success': False,
                'target_node': target_node,
                'target_type': 'node',
                'epsilon': epsilon,
                'st': st_value,
                'selected_sensitive_attr': selected_sensitive_attr,
                'error': '训练图太小'
            }

        G_S = self.generate_synthetic_graph(G_R, epsilon)

        G_X = G_S

        node_all_features = {}
        node_sensitive_values = {}

        for node in G_X.nodes():
            all_feat, sens_cand = self._extract_all_features_from_graph(G_X, node)
            if all_feat and selected_sensitive_attr in sens_cand:
                node_all_features[node] = all_feat
                node_sensitive_values[node] = sens_cand[selected_sensitive_attr]

        if len(node_all_features) < 10:
            return {
                'success': False,
                'target_node': target_node,
                'target_type': 'node',
                'epsilon': epsilon,
                'st': st_value,
                'selected_sensitive_attr': selected_sensitive_attr,
                'error': '合成图中有效节点太少'
            }

        try:
            X_train = []
            y_train = []
            feature_names = None

            if node_all_features:
                first_node = next(iter(node_all_features))
                non_sensitive_feat = self._get_non_sensitive_features(node_all_features[first_node],
                                                                      selected_sensitive_attr)
                feature_names = list(non_sensitive_feat.keys())

            for node, all_feat in node_all_features.items():
                if node == target_node:
                    continue

                if selected_sensitive_attr not in all_feat:
                    continue

                non_sensitive_feat = self._get_non_sensitive_features(all_feat, selected_sensitive_attr)

                feature_vector = [non_sensitive_feat.get(name, 0.0) for name in feature_names]
                X_train.append(feature_vector)

                if selected_sensitive_attr in self.attribute_types and self.attribute_types[
                    selected_sensitive_attr] == 'discrete':
                    y_train.append(int(round(node_sensitive_values[node])))
                else:
                    y_train.append(node_sensitive_values[node])

            if len(X_train) < 5:
                return {
                    'success': False,
                    'target_node': target_node,
                    'target_type': 'node',
                    'epsilon': epsilon,
                    'st': st_value,
                    'selected_sensitive_attr': selected_sensitive_attr,
                    'error': '训练样本不足'
                }

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            if st_value == 1:
                target_non_sensitive_feat = self._get_non_sensitive_features(target_all_features_real,
                                                                             selected_sensitive_attr)
                info_source = "真实特征（从参考图获得）"
            else:
                fake_all_features = self._generate_fake_all_features(target_node, reference_graph)
                target_non_sensitive_feat = self._get_non_sensitive_features(fake_all_features, selected_sensitive_attr)
                info_source = "猜测特征（模拟未知情况）"

            if not target_non_sensitive_feat:
                return {
                    'success': False,
                    'target_node': target_node,
                    'target_type': 'node',
                    'epsilon': epsilon,
                    'st': st_value,
                    'selected_sensitive_attr': selected_sensitive_attr,
                    'error': '无法生成目标节点特征'
                }

            target_feature_vector = []
            for name in feature_names:
                target_feature_vector.append(target_non_sensitive_feat.get(name, random.random()))

            target_feature_scaled = scaler.transform([target_feature_vector])

            predicted_value = float(model.predict(target_feature_scaled)[0])

            if selected_sensitive_attr in self.attribute_types and self.attribute_types[
                selected_sensitive_attr] == 'discrete':
                predicted_value = float(round(predicted_value))
                if predicted_value < 0:
                    predicted_value = 0.0

            absolute_error, normalized_error = self._calculate_normalized_error(
                predicted_value, true_sensitive_value, selected_sensitive_attr)

            threshold = self._get_threshold_for_attribute(selected_sensitive_attr)

            if selected_sensitive_attr in self.attribute_types and self.attribute_types[
                selected_sensitive_attr] == 'discrete':
                is_correct = 1 if abs(predicted_value - true_sensitive_value) < 1e-10 else 0
            else:
                is_correct = 1 if normalized_error <= threshold else 0

            return {
                'success': True,
                'target_node': target_node,
                'target_type': 'node',
                'epsilon': epsilon,
                'st': st_value,
                'selected_sensitive_attr': selected_sensitive_attr,
                'attribute_type': self.attribute_types.get(selected_sensitive_attr, 'continuous'),
                'threshold': threshold,
                'true_value': true_sensitive_value,
                'predicted_value': predicted_value,
                'absolute_error': absolute_error,
                'normalized_error': normalized_error,
                'is_correct': is_correct,
                'info_source': info_source,
                'graph_sizes': {
                    'G_R': (G_R.number_of_nodes(), G_R.number_of_edges()),
                    'G_S': (G_S.number_of_nodes(), G_S.number_of_edges())
                },
                'training_samples': len(X_train),
                'feature_names': feature_names
            }

        except Exception as e:
            return {
                'success': False,
                'target_node': target_node,
                'target_type': 'node',
                'epsilon': epsilon,
                'st': st_value,
                'selected_sensitive_attr': selected_sensitive_attr,
                'error': f'模型训练或预测失败: {str(e)}'
            }

    def _execute_edge_attribute_inference_attack_with_st(self, target_edge: Tuple[int, int], epsilon: float,
                                                         reference_graph: nx.Graph, st_value: int) -> Dict[str, Any]:

        u, v = target_edge

        target_all_features_real, sensitive_candidates = self._extract_edge_features_from_graph(reference_graph,
                                                                                                target_edge)

        if not sensitive_candidates:
            return {
                'success': False,
                'target_edge': target_edge,
                'target_type': 'edge',
                'epsilon': epsilon,
                'st': st_value,
                'error': '无法提取目标边的敏感属性候选'
            }

        selected_sensitive_attr, true_sensitive_value = self._select_sensitive_attribute(sensitive_candidates,
                                                                                         is_edge=True)

        if selected_sensitive_attr in self.attribute_ranges:
            attr_range_info = self.attribute_ranges[selected_sensitive_attr]
            attr_type = self.attribute_types.get(selected_sensitive_attr, 'continuous')
            threshold = self._get_threshold_for_attribute(selected_sensitive_attr)

            print(f"      [边敏感属性选择] 本次攻击选择 '{selected_sensitive_attr}' 作为敏感属性")
            print(f"                    类型: {attr_type}, 阈值: {threshold}, "
                  f"真实值: {true_sensitive_value:.4f}, "
                  f"取值范围: [{attr_range_info['min']:.4f}, {attr_range_info['max']:.4f}]")
        else:
            print(
                f"      [边敏感属性选择] 本次攻击选择 '{selected_sensitive_attr}' 作为敏感属性，真实值: {true_sensitive_value:.4f}")

        G_R = self._sample_training_graph_for_edge(reference_graph, target_edge, include_target=(st_value == 1))

        if G_R.number_of_nodes() < 5 or G_R.number_of_edges() < 3:
            return {
                'success': False,
                'target_edge': target_edge,
                'target_type': 'edge',
                'epsilon': epsilon,
                'st': st_value,
                'selected_sensitive_attr': selected_sensitive_attr,
                'error': '训练图太小或边太少'
            }

        G_S = self.generate_synthetic_graph(G_R, epsilon)

        G_X = G_S

        edge_all_features = {}
        edge_sensitive_values = {}

        for edge_candidate in G_X.edges():
            all_feat, sens_cand = self._extract_edge_features_from_graph(G_X, edge_candidate)
            if all_feat and selected_sensitive_attr in sens_cand:
                edge_all_features[edge_candidate] = all_feat
                edge_sensitive_values[edge_candidate] = sens_cand[selected_sensitive_attr]

        if len(edge_all_features) < 5:
            return {
                'success': False,
                'target_edge': target_edge,
                'target_type': 'edge',
                'epsilon': epsilon,
                'st': st_value,
                'selected_sensitive_attr': selected_sensitive_attr,
                'error': '合成图中有效边太少'
            }

        try:
            X_train = []
            y_train = []
            feature_names = None

            if edge_all_features:
                first_edge = next(iter(edge_all_features))
                non_sensitive_feat = self._get_non_sensitive_features(edge_all_features[first_edge],
                                                                      selected_sensitive_attr)
                feature_names = list(non_sensitive_feat.keys())

            for edge_candidate, all_feat in edge_all_features.items():
                if edge_candidate == target_edge:
                    continue

                if selected_sensitive_attr not in all_feat:
                    continue

                non_sensitive_feat = self._get_non_sensitive_features(all_feat, selected_sensitive_attr)

                feature_vector = [non_sensitive_feat.get(name, 0.0) for name in feature_names]
                X_train.append(feature_vector)

                if selected_sensitive_attr in self.attribute_types and self.attribute_types[
                    selected_sensitive_attr] == 'discrete':
                    y_train.append(int(round(edge_sensitive_values[edge_candidate])))
                else:
                    y_train.append(edge_sensitive_values[edge_candidate])

            if len(X_train) < 5:
                return {
                    'success': False,
                    'target_edge': target_edge,
                    'target_type': 'edge',
                    'epsilon': epsilon,
                    'st': st_value,
                    'selected_sensitive_attr': selected_sensitive_attr,
                    'error': '训练样本不足'
                }

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            if st_value == 1:
                target_non_sensitive_feat = self._get_non_sensitive_features(target_all_features_real,
                                                                             selected_sensitive_attr)
                info_source = "真实特征（从参考图获得）"
            else:
                fake_all_features = self._generate_fake_edge_features(target_edge, reference_graph)
                target_non_sensitive_feat = self._get_non_sensitive_features(fake_all_features, selected_sensitive_attr)
                info_source = "猜测特征（模拟未知情况）"

            if not target_non_sensitive_feat:
                return {
                    'success': False,
                    'target_edge': target_edge,
                    'target_type': 'edge',
                    'epsilon': epsilon,
                    'st': st_value,
                    'selected_sensitive_attr': selected_sensitive_attr,
                    'error': '无法生成目标边特征'
                }

            target_feature_vector = []
            for name in feature_names:
                target_feature_vector.append(target_non_sensitive_feat.get(name, 0.0))

            target_feature_scaled = scaler.transform([target_feature_vector])

            predicted_value = float(model.predict(target_feature_scaled)[0])

            if selected_sensitive_attr in self.attribute_types and self.attribute_types[
                selected_sensitive_attr] == 'discrete':
                predicted_value = float(round(predicted_value))
                if predicted_value < 0:
                    predicted_value = 0.0

            absolute_error, normalized_error = self._calculate_normalized_error(
                predicted_value, true_sensitive_value, selected_sensitive_attr)

            threshold = self._get_threshold_for_attribute(selected_sensitive_attr)

            if selected_sensitive_attr in self.attribute_types and self.attribute_types[
                selected_sensitive_attr] == 'discrete':
                is_correct = 1 if abs(predicted_value - true_sensitive_value) < 1e-10 else 0
            else:
                is_correct = 1 if normalized_error <= threshold else 0

            return {
                'success': True,
                'target_edge': target_edge,
                'target_type': 'edge',
                'epsilon': epsilon,
                'st': st_value,
                'selected_sensitive_attr': selected_sensitive_attr,
                'attribute_type': self.attribute_types.get(selected_sensitive_attr, 'continuous'),
                'threshold': threshold,
                'true_value': true_sensitive_value,
                'predicted_value': predicted_value,
                'absolute_error': absolute_error,
                'normalized_error': normalized_error,
                'is_correct': is_correct,
                'info_source': info_source,
                'graph_sizes': {
                    'G_R': (G_R.number_of_nodes(), G_R.number_of_edges()),
                    'G_S': (G_S.number_of_nodes(), G_S.number_of_edges())
                },
                'training_samples': len(X_train),
                'feature_names': feature_names
            }

        except Exception as e:
            return {
                'success': False,
                'target_edge': target_edge,
                'target_type': 'edge',
                'epsilon': epsilon,
                'st': st_value,
                'selected_sensitive_attr': selected_sensitive_attr,
                'error': f'模型训练或预测失败: {str(e)}'
            }

    def calculate_attack_advantage(self, attack_results: List[Dict[str, Any]]) -> Tuple[float, Dict]:

        if not attack_results:
            return 0.0, {}

        results_st1 = [r for r in attack_results if r['success'] and r.get('st', 1) == 1]
        results_st0 = [r for r in attack_results if r['success'] and r.get('st', 0) == 0]

        if not results_st1 or not results_st0:
            print(f"警告: s_t=1结果数={len(results_st1)}, s_t=0结果数={len(results_st0)}")
            return 0.0, {'total': len(attack_results), 'successful_st1': len(results_st1),
                         'successful_st0': len(results_st0)}

        attr_stats = {}

        all_attrs = self.sensitive_attributes + self.edge_sensitive_attributes

        for attr in all_attrs:
            attr_results_st1 = [r for r in results_st1 if r.get('selected_sensitive_attr') == attr]
            attr_results_st0 = [r for r in results_st0 if r.get('selected_sensitive_attr') == attr]

            if attr_results_st1 and attr_results_st0:
                correct_st1 = sum(1 for r in attr_results_st1 if r.get('is_correct', 0) == 1)
                correct_st0 = sum(1 for r in attr_results_st0 if r.get('is_correct', 0) == 1)

                prob_correct_st1 = correct_st1 / len(attr_results_st1)
                prob_correct_st0 = correct_st0 / len(attr_results_st0)
                attack_advantage = prob_correct_st1 - prob_correct_st0

                attr_stats[attr] = {
                    'count_st1': len(attr_results_st1),
                    'count_st0': len(attr_results_st0),
                    'prob_correct_st1': prob_correct_st1,
                    'prob_correct_st0': prob_correct_st0,
                    'attack_advantage': attack_advantage,
                    'attribute_type': self.attribute_types.get(attr, 'continuous'),
                    'threshold': self._get_threshold_for_attribute(attr)
                }

        correct_st1 = sum(1 for r in results_st1 if r.get('is_correct', 0) == 1)
        prob_correct_st1 = correct_st1 / len(results_st1)

        correct_st0 = sum(1 for r in results_st0 if r.get('is_correct', 0) == 1)
        prob_correct_st0 = correct_st0 / len(results_st0)

        attack_advantage = prob_correct_st1 - prob_correct_st0

        normalized_errors_st1 = [r.get('normalized_error', 1.0) for r in results_st1]
        normalized_errors_st0 = [r.get('normalized_error', 1.0) for r in results_st0]

        stats = {
            'total_attacks': len(attack_results),
            'successful_attacks': len(results_st1) + len(results_st0),
            'successful_st1': len(results_st1),
            'successful_st0': len(results_st0),
            'correct_st1': correct_st1,
            'correct_st0': correct_st0,
            'prob_correct_st1': prob_correct_st1,
            'prob_correct_st0': prob_correct_st0,
            'attack_advantage': attack_advantage,
            'avg_normalized_error_st1': np.mean(normalized_errors_st1) if normalized_errors_st1 else 0,
            'avg_normalized_error_st0': np.mean(normalized_errors_st0) if normalized_errors_st0 else 0,
            'median_normalized_error_st1': np.median(normalized_errors_st1) if normalized_errors_st1 else 0,
            'median_normalized_error_st0': np.median(normalized_errors_st0) if normalized_errors_st0 else 0,
            'std_normalized_error_st1': np.std(normalized_errors_st1) if len(normalized_errors_st1) > 1 else 0,
            'std_normalized_error_st0': np.std(normalized_errors_st0) if len(normalized_errors_st0) > 1 else 0,
            'attribute_statistics': attr_stats
        }

        return attack_advantage, stats

    def run_aia_evaluation_with_st(self,
                                   epsilon_values: List[float],
                                   reference_graph_path: str,
                                   node_target_count: int = 10,
                                   edge_target_count: int = 5,
                                   attacks_per_target: int = 5,
                                   output_path: str = "./aia_results",
                                   output_prefix: str = None) -> Tuple[pd.DataFrame, Dict]:

        os.makedirs(output_path, exist_ok=True)

        reference_graph = self.load_graph_from_txt(reference_graph_path)
        print(f"参考图加载完成: {reference_graph.number_of_nodes()} 节点, "
              f"{reference_graph.number_of_edges()} 边")

        all_nodes = list(reference_graph.nodes())
        if len(all_nodes) < node_target_count:
            target_nodes = all_nodes
            print(f"警告: 只有 {len(all_nodes)} 个节点，少于请求的 {node_target_count} 个")
        else:
            target_nodes = random.sample(all_nodes, node_target_count)

        all_edges = list(reference_graph.edges())
        if len(all_edges) < edge_target_count:
            target_edges = all_edges
            print(f"警告: 只有 {len(all_edges)} 条边，少于请求的 {edge_target_count} 条")
        else:
            target_edges = random.sample(all_edges, edge_target_count)

        print(f"选择了 {len(target_nodes)} 个目标节点和 {len(target_edges)} 条目标边进行攻击")
        print(f"每个目标节点攻击 {attacks_per_target * 2} 次 (s_t=0和s_t=1各{attacks_per_target}次)")
        print(f"每条目标边攻击 {attacks_per_target * 2} 次 (s_t=0和s_t=1各{attacks_per_target}次)")
        print(f"节点敏感属性: {self.sensitive_attributes}")
        print(f"边敏感属性: {self.edge_sensitive_attributes}")

        all_attack_results = []
        detailed_results = {}

        table_data = []

        for epsilon in epsilon_values:
            print("\n" + "=" * 50)
            print(f"评估 epsilon = {epsilon}")
            print("=" * 50)

            epsilon_attack_results = []

            print(f"\n▶ 节点攻击部分")
            for target_idx, target_node in enumerate(target_nodes):
                print(f"\n  攻击目标节点 {target_idx + 1}/{len(target_nodes)}: {target_node}")

                target_results_st1 = []
                target_results_st0 = []

                print(f"    s_t=1 (目标在训练集中):")
                for attack_iter in range(attacks_per_target):
                    result = self._execute_attribute_inference_attack_with_st(target_node, epsilon,
                                                                              reference_graph, st_value=1)
                    target_results_st1.append(result)
                    epsilon_attack_results.append(result)
                    all_attack_results.append(result)

                    if result['success']:
                        status = "✓" if result.get('is_correct', 0) == 1 else "~"
                        error_str = f"{result.get('normalized_error', 1.0):.3f}"
                        info_source = result.get('info_source', '未知')
                        sensitive_attr = result.get('selected_sensitive_attr', '未知')
                        attr_type = result.get('attribute_type', 'continuous')
                        threshold = result.get('threshold', 0.3)
                        print(f"      第{attack_iter + 1:2d}次 | {status} | "
                              f"敏感属性: {sensitive_attr:15s} | "
                              f"类型: {attr_type:10s} | "
                              f"阈值: {threshold:.1f} | "
                              f"真值: {result['true_value']:8.4f} | "
                              f"预测: {result['predicted_value']:8.4f} | "
                              f"归一化误差: {error_str} | {info_source}")
                    else:
                        sensitive_attr = result.get('selected_sensitive_attr', '未知')
                        print(
                            f"      第{attack_iter + 1:2d}次 | ✗ | 敏感属性: {sensitive_attr} | 错误: {result.get('error', '未知')}")

                print(f"    s_t=0 (目标不在训练集中):")
                for attack_iter in range(attacks_per_target):
                    result = self._execute_attribute_inference_attack_with_st(target_node, epsilon,
                                                                              reference_graph, st_value=0)
                    target_results_st0.append(result)
                    epsilon_attack_results.append(result)
                    all_attack_results.append(result)

                    if result['success']:
                        status = "✓" if result.get('is_correct', 0) == 1 else "~"
                        error_str = f"{result.get('normalized_error', 1.0):.3f}"
                        info_source = result.get('info_source', '未知')
                        sensitive_attr = result.get('selected_sensitive_attr', '未知')
                        attr_type = result.get('attribute_type', 'continuous')
                        threshold = result.get('threshold', 0.3)
                        print(f"      第{attack_iter + 1:2d}次 | {status} | "
                              f"敏感属性: {sensitive_attr:15s} | "
                              f"类型: {attr_type:10s} | "
                              f"阈值: {threshold:.1f} | "
                              f"真值: {result['true_value']:8.4f} | "
                              f"预测: {result['predicted_value']:8.4f} | "
                              f"归一化误差: {error_str} | {info_source}")
                    else:
                        sensitive_attr = result.get('selected_sensitive_attr', '未知')
                        print(
                            f"      第{attack_iter + 1:2d}次 | ✗ | 敏感属性: {sensitive_attr} | 错误: {result.get('error', '未知')}")

                successful_st1 = [r for r in target_results_st1 if r['success']]
                successful_st0 = [r for r in target_results_st0 if r['success']]

                if successful_st1:
                    avg_error_st1 = np.mean([r.get('normalized_error', 1.0) for r in successful_st1])
                    success_count_st1 = sum(1 for r in successful_st1 if r.get('is_correct', 0) == 1)
                    success_rate_st1 = success_count_st1 / len(successful_st1)
                    print(
                        f"    s_t=1: 平均归一化误差={avg_error_st1:.3f}, 成功率={success_rate_st1:.3f} ({success_count_st1}/{len(successful_st1)})")

                if successful_st0:
                    avg_error_st0 = np.mean([r.get('normalized_error', 1.0) for r in successful_st0])
                    success_count_st0 = sum(1 for r in successful_st0 if r.get('is_correct', 0) == 1)
                    success_rate_st0 = success_count_st0 / len(successful_st0)
                    print(
                        f"    s_t=0: 平均归一化误差={avg_error_st0:.3f}, 成功率={success_rate_st0:.3f} ({success_count_st0}/{len(successful_st0)})")

            print(f"\n▶ 边攻击部分")
            for target_idx, target_edge in enumerate(target_edges):
                edge_str = f"({target_edge[0]},{target_edge[1]})"
                print(f"\n  攻击目标边 {target_idx + 1}/{len(target_edges)}: {edge_str}")

                target_results_st1 = []
                target_results_st0 = []

                print(f"    s_t=1 (目标边在训练集中):")
                for attack_iter in range(attacks_per_target):
                    result = self._execute_edge_attribute_inference_attack_with_st(target_edge, epsilon,
                                                                                   reference_graph, st_value=1)
                    target_results_st1.append(result)
                    epsilon_attack_results.append(result)
                    all_attack_results.append(result)

                    if result['success']:
                        status = "✓" if result.get('is_correct', 0) == 1 else "~"
                        error_str = f"{result.get('normalized_error', 1.0):.3f}"
                        info_source = result.get('info_source', '未知')
                        sensitive_attr = result.get('selected_sensitive_attr', '未知')
                        attr_type = result.get('attribute_type', 'continuous')
                        threshold = result.get('threshold', 0.3)
                        print(f"      第{attack_iter + 1:2d}次 | {status} | "
                              f"边敏感属性: {sensitive_attr:20s} | "
                              f"类型: {attr_type:10s} | "
                              f"阈值: {threshold:.1f} | "
                              f"真值: {result['true_value']:8.4f} | "
                              f"预测: {result['predicted_value']:8.4f} | "
                              f"归一化误差: {error_str} | {info_source}")
                    else:
                        sensitive_attr = result.get('selected_sensitive_attr', '未知')
                        print(
                            f"      第{attack_iter + 1:2d}次 | ✗ | 边敏感属性: {sensitive_attr} | 错误: {result.get('error', '未知')}")

                print(f"    s_t=0 (目标边不在训练集中):")
                for attack_iter in range(attacks_per_target):
                    result = self._execute_edge_attribute_inference_attack_with_st(target_edge, epsilon,
                                                                                   reference_graph, st_value=0)
                    target_results_st0.append(result)
                    epsilon_attack_results.append(result)
                    all_attack_results.append(result)

                    if result['success']:
                        status = "✓" if result.get('is_correct', 0) == 1 else "~"
                        error_str = f"{result.get('normalized_error', 1.0):.3f}"
                        info_source = result.get('info_source', '未知')
                        sensitive_attr = result.get('selected_sensitive_attr', '未知')
                        attr_type = result.get('attribute_type', 'continuous')
                        threshold = result.get('threshold', 0.3)
                        print(f"      第{attack_iter + 1:2d}次 | {status} | "
                              f"边敏感属性: {sensitive_attr:20s} | "
                              f"类型: {attr_type:10s} | "
                              f"阈值: {threshold:.1f} | "
                              f"真值: {result['true_value']:8.4f} | "
                              f"预测: {result['predicted_value']:8.4f} | "
                              f"归一化误差: {error_str} | {info_source}")
                    else:
                        sensitive_attr = result.get('selected_sensitive_attr', '未知')
                        print(
                            f"      第{attack_iter + 1:2d}次 | ✗ | 边敏感属性: {sensitive_attr} | 错误: {result.get('error', '未知')}")

                successful_st1 = [r for r in target_results_st1 if r['success']]
                successful_st0 = [r for r in target_results_st0 if r['success']]

                if successful_st1:
                    avg_error_st1 = np.mean([r.get('normalized_error', 1.0) for r in successful_st1])
                    success_count_st1 = sum(1 for r in successful_st1 if r.get('is_correct', 0) == 1)
                    success_rate_st1 = success_count_st1 / len(successful_st1)
                    print(
                        f"    s_t=1: 平均归一化误差={avg_error_st1:.3f}, 成功率={success_rate_st1:.3f} ({success_count_st1}/{len(successful_st1)})")

                if successful_st0:
                    avg_error_st0 = np.mean([r.get('normalized_error', 1.0) for r in successful_st0])
                    success_count_st0 = sum(1 for r in successful_st0 if r.get('is_correct', 0) == 1)
                    success_rate_st0 = success_count_st0 / len(successful_st0)
                    print(
                        f"    s_t=0: 平均归一化误差={avg_error_st0:.3f}, 成功率={success_rate_st0:.3f} ({success_count_st0}/{len(successful_st0)})")

            node_attack_results = [r for r in epsilon_attack_results if r.get('target_type') == 'node' and r['success']]
            edge_attack_results = [r for r in epsilon_attack_results if r.get('target_type') == 'edge' and r['success']]

            if node_attack_results:
                node_advantage, node_stats = self.calculate_attack_advantage(node_attack_results)

                node_unique_targets = len(set(r.get('target_node') for r in node_attack_results if 'target_node' in r))

                node_in_member = len([r for r in node_attack_results if r.get('st') == 1])
                node_out_member = len([r for r in node_attack_results if r.get('st') == 0])

                node_tpr = node_stats.get('prob_correct_st1', 0) if node_stats.get('successful_st1', 0) > 0 else 0
                node_fpr = node_stats.get('prob_correct_st0', 0) if node_stats.get('successful_st0', 0) > 0 else 0

                node_success_count = node_stats.get('correct_st1', 0) + node_stats.get('correct_st0', 0)
                node_success_rate = node_success_count / len(node_attack_results) if node_attack_results else 0

                node_target_success_rates = []
                for target in target_nodes:
                    target_results = [r for r in node_attack_results if r.get('target_node') == target]
                    if target_results:
                        target_success = sum(1 for r in target_results if r.get('is_correct', 0) == 1)
                        target_success_rate = target_success / len(target_results)
                        node_target_success_rates.append(target_success_rate)

                node_advantage_std = np.std(node_target_success_rates) if len(node_target_success_rates) > 1 else 0
                node_advantage_range = (max(node_target_success_rates) - min(
                    node_target_success_rates)) if node_target_success_rates else 0

                table_data.append({
                    'attack_type': 'node',
                    'epsilon': epsilon,
                    'unique_targets': node_unique_targets,
                    'total_attacks': len(node_attack_results),
                    'in_member_attacks': node_in_member,
                    'out_member_attacks': node_out_member,
                    'tpr': node_tpr,
                    'fpr': node_fpr,
                    'advantage': node_advantage,
                    'success_rate': node_success_rate,
                    'advantage_std': node_advantage_std,
                    'advantage_range': node_advantage_range,
                    '_total_execution_time_s': 0,
                    '_avg_attack_time_s': 0
                })

            if edge_attack_results:
                edge_advantage, edge_stats = self.calculate_attack_advantage(edge_attack_results)

                edge_unique_targets = len(set(r.get('target_edge') for r in edge_attack_results if 'target_edge' in r))

                edge_in_member = len([r for r in edge_attack_results if r.get('st') == 1])
                edge_out_member = len([r for r in edge_attack_results if r.get('st') == 0])

                edge_tpr = edge_stats.get('prob_correct_st1', 0) if edge_stats.get('successful_st1', 0) > 0 else 0
                edge_fpr = edge_stats.get('prob_correct_st0', 0) if edge_stats.get('successful_st0', 0) > 0 else 0

                edge_success_count = edge_stats.get('correct_st1', 0) + edge_stats.get('correct_st0', 0)
                edge_success_rate = edge_success_count / len(edge_attack_results) if edge_attack_results else 0

                edge_target_success_rates = []
                for target in target_edges:
                    target_results = [r for r in edge_attack_results if r.get('target_edge') == target]
                    if target_results:
                        target_success = sum(1 for r in target_results if r.get('is_correct', 0) == 1)
                        target_success_rate = target_success / len(target_results)
                        edge_target_success_rates.append(target_success_rate)

                edge_advantage_std = np.std(edge_target_success_rates) if len(edge_target_success_rates) > 1 else 0
                edge_advantage_range = (max(edge_target_success_rates) - min(
                    edge_target_success_rates)) if edge_target_success_rates else 0

                table_data.append({
                    'attack_type': 'edge',
                    'epsilon': epsilon,
                    'unique_targets': edge_unique_targets,
                    'total_attacks': len(edge_attack_results),
                    'in_member_attacks': edge_in_member,
                    'out_member_attacks': edge_out_member,
                    'tpr': edge_tpr,
                    'fpr': edge_fpr,
                    'advantage': edge_advantage,
                    'success_rate': edge_success_rate,
                    'advantage_std': edge_advantage_std,
                    'advantage_range': edge_advantage_range,
                    '_total_execution_time_s': 0,
                    '_avg_attack_time_s': 0
                })

            epsilon_stats = {
                'epsilon': epsilon,
                'attack_results': epsilon_attack_results,
                'target_nodes': target_nodes,
                'target_edges': target_edges
            }
            detailed_results[epsilon] = epsilon_stats

        result_df = pd.DataFrame(table_data)

        if not result_df.empty:
            if output_prefix:
                csv_path = os.path.join(output_path, f"{output_prefix}_statistics.csv")
            else:
                csv_path = os.path.join(output_path,
                                        f"AIA_{datetime.now().strftime('%Y%m%d_%H%M%S')}_statistics.csv")
            result_df.to_csv(csv_path, index=False)
            print(f"\n结果摘要已保存到: {csv_path}")

        if output_prefix:
            pkl_path = os.path.join(output_path, f"{output_prefix}_detailed.pkl")
        else:
            pkl_path = os.path.join(output_path,
                                    f"AIA_{datetime.now().strftime('%Y%m%d_%H%M%S')}_detailed.pkl")

        with open(pkl_path, 'wb') as f:
            pickle.dump(detailed_results, f)
        print(f"详细结果已保存到: {pkl_path}")

        print("\n" + "=" * 50)
        print("AIA评估结果摘要（包含节点和边攻击，s_t=0和s_t=1）")
        print("=" * 50)
        if not result_df.empty:
            print("\n" + result_df.to_string(index=False))
        else:
            print("没有有效的评估结果")

        return result_df, detailed_results

    def print_attack_summary(self):
        for attr in self.sensitive_attributes + self.edge_sensitive_attributes:
            if attr in self.attribute_ranges and attr in self.attribute_types:
                range_info = self.attribute_ranges[attr]
                attr_type = self.attribute_types[attr]
                threshold = self._get_threshold_for_attribute(attr)
                print(f"  {attr:20s}: 类型={attr_type}, 阈值={threshold}, "
                      f"范围=[{range_info['min']:.4f}, {range_info['max']:.4f}]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='属性推断攻击(AIA)')

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

    output_path = "./aia_results"
    os.makedirs(output_path, exist_ok=True)

    attacker = StructuralAttributeIAttacker(
        original_graph_path=args.original_graph,
        dpgs_module_path=args.dpgs_module,
        dpgs_function_name=args.dpgs_function
    )

    final_table, all_results = attacker.run_aia_evaluation_with_st(
        epsilon_values=epsilon_values,
        reference_graph_path=args.reference_graph,
        node_target_count=args.node_targets,
        edge_target_count=args.edge_targets,
        attacks_per_target=args.attacks_per_target,
        output_path=output_path,
        output_prefix=args.output_prefix
    )

    attacker.print_attack_summary()