import os
import numpy as np
import networkx as nx
import pandas as pd
import random
import importlib.util
import sys
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse
from scipy.stats import ks_2samp
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import jensenshannon

EPS = 1e-8


class GraphUtilityLossEvaluator:
    def __init__(self, original_graph_path: str, dpgs_module_path: str,
                 dpgs_function_name: str = 'create_bter_graph_from_adjacency_matrix'):

        self.original_graph = self.load_graph_from_txt(original_graph_path)

        self.dpgs_function = self.load_dpgs_function(dpgs_module_path, dpgs_function_name)

        self.scalar_tasks = [
            'node_count', 'edge_count', 'triangle_count', 'avg_degree',
            'diameter', 'avg_shortest_path', 'global_clustering',
            'avg_clustering', 'modularity', 'assortativity',
            'degree_variance'
        ]

        self.dist_tasks = ['degree_distribution', 'distance_distribution']

        self.struct_tasks = ['community_detection', 'eigenvector_centrality']

        self.all_tasks = self.scalar_tasks + self.dist_tasks + self.struct_tasks

        self.original_degree_variance = self.compute_degree_variance(self.original_graph)
        print(f"📊 original graph degree variance: {self.original_degree_variance:.6f}")

    def load_graph_from_txt(self, file_path: str) -> nx.Graph:
        G = nx.Graph()
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        u, v = map(int, line.strip().split()[:2])
                        G.add_edge(u, v)
            print(f"✅ Loaded original graph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
            return G
        except Exception as e:
            print(f"❌ Failed to load graph: {e}")
            return nx.Graph()

    def load_dpgs_function(self, module_path: str, function_name: str):
        try:
            spec = importlib.util.spec_from_file_location("dpgs_module", module_path)
            dpgs_module = importlib.util.module_from_spec(spec)
            original_argv = sys.argv.copy()
            sys.argv = [sys.argv[0]]
            spec.loader.exec_module(dpgs_module)
            sys.argv = original_argv
            func = getattr(dpgs_module, function_name)
            print(f"✅ Loaded DPGS synthesis function: {function_name}")
            return func
        except Exception as e:
            print(f"❌ Failed to load DPGS function: {e}")
            return lambda adj, eps: adj

    def graph_to_adj(self, G: nx.Graph) -> np.ndarray:
        nodes = sorted(G.nodes())
        n = len(nodes)
        adj = np.zeros((n, n))
        node2idx = {node: i for i, node in enumerate(nodes)}
        for u, v in G.edges():
            i, j = node2idx[u], node2idx[v]
            adj[i, j] = adj[j, i] = 1
        return adj

    def adj_to_graph(self, adj: np.ndarray) -> nx.Graph:
        G = nx.Graph()
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    G.add_edge(i, j)
        return G

    def generate_synthetic_graph(self, epsilon: float) -> nx.Graph:
        original_adj = self.graph_to_adj(self.original_graph)
        synthetic_adj = self.dpgs_function(original_adj, epsilon)
        return self.adj_to_graph(synthetic_adj)

    def compute_relative_error(self, val_r: float, val_s: float) -> float:
        numerator = abs(val_r - val_s)
        denominator = max(abs(val_r), EPS)
        return numerator / denominator

    def compute_dist_discrepancy(self, dist_r: np.ndarray, dist_s: np.ndarray) -> float:
        if len(dist_r) == 0 or len(dist_s) == 0:
            return 1.0
        ks_stat, _ = ks_2samp(dist_r, dist_s)
        return ks_stat

    def compute_struct_discrepancy(self, struct_r, struct_s, task_type: str) -> float:
        if len(struct_r) != len(struct_s):
            min_len = min(len(struct_r), len(struct_s))
            print(f"Warning: {task_type} length mismatch - r:{len(struct_r)} vs s:{len(struct_s)}, truncating to {min_len}")
            struct_r = struct_r[:min_len]
            struct_s = struct_s[:min_len]

        if task_type == 'community_detection':
            return 1.0 - normalized_mutual_info_score(struct_r, struct_s)
        elif task_type == 'eigenvector_centrality':
            mae = np.mean(np.abs(np.array(struct_r) - np.array(struct_s)))
            max_val = max(max(struct_r), max(struct_s), EPS)
            return mae / max_val
        return 1.0

    def compute_degree_variance(self, G: nx.Graph) -> float:
        degrees = [d for n, d in G.degree()]
        if len(degrees) == 0:
            return 0.0
        degree_variance = np.var(degrees)
        return degree_variance

    def compute_scalar_metrics(self, G: nx.Graph) -> Dict[str, float]:
        metrics = {}
        metrics['node_count'] = G.number_of_nodes()
        metrics['edge_count'] = G.number_of_edges()
        metrics['triangle_count'] = sum(nx.triangles(G).values()) // 3
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees) if degrees else 0
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            metrics['diameter'] = nx.diameter(G.subgraph(largest_cc))
        except:
            metrics['diameter'] = 0
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G.subgraph(largest_cc))
        except:
            metrics['avg_shortest_path'] = 0
        try:
            metrics['global_clustering'] = nx.transitivity(G)
        except:
            metrics['global_clustering'] = 0
        try:
            metrics['avg_clustering'] = nx.average_clustering(G)
        except:
            metrics['avg_clustering'] = 0
        try:
            communities = nx.community.louvain_communities(G)
            metrics['modularity'] = nx.community.modularity(G, communities)
        except:
            metrics['modularity'] = 0
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            metrics['assortativity'] = 0
        metrics['degree_variance'] = self.compute_degree_variance(G)
        return metrics

    def compute_dist_metrics(self, G: nx.Graph) -> Dict[str, list]:
        dists = {}
        dists['degree_distribution'] = [d for n, d in G.degree()]
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            paths = dict(nx.all_pairs_shortest_path_length(subG))
            dists['distance_distribution'] = [l for u in paths for v, l in paths[u].items() if u != v]
        except:
            dists['distance_distribution'] = [0]
        return dists

    def compute_struct_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        structs = {}
        sorted_nodes = sorted(G.nodes())

        try:
            communities = nx.community.louvain_communities(G)
            node_to_label = {}
            for i, comm in enumerate(communities):
                for n in comm:
                    node_to_label[n] = i
            structs['community_detection'] = [node_to_label[n] for n in sorted_nodes]
        except:
            structs['community_detection'] = [0] * len(sorted_nodes)

        try:
            ec = nx.eigenvector_centrality_numpy(G)
            structs['eigenvector_centrality'] = [ec[n] for n in sorted_nodes]
        except:
            structs['eigenvector_centrality'] = [0] * len(sorted_nodes)

        return structs

    def compute_utility_loss(self, synthetic_graph: nx.Graph) -> Dict[str, float]:
        G_r = self.original_graph
        G_s = synthetic_graph
        loss = {}

        r_scalar = self.compute_scalar_metrics(G_r)
        s_scalar = self.compute_scalar_metrics(G_s)
        for task in self.scalar_tasks:
            loss[task] = self.compute_relative_error(r_scalar[task], s_scalar[task])

        r_dist = self.compute_dist_metrics(G_r)
        s_dist = self.compute_dist_metrics(G_s)
        for task in self.dist_tasks:
            loss[task] = self.compute_dist_discrepancy(np.array(r_dist[task]), np.array(s_dist[task]))

        r_struct = self.compute_struct_metrics(G_r)
        s_struct = self.compute_struct_metrics(G_s)
        for task in self.struct_tasks:
            loss[task] = self.compute_struct_discrepancy(r_struct[task], s_struct[task], task)

        loss_values = [loss[t] for t in self.all_tasks]
        loss['overall_utility_loss'] = np.mean(loss_values)
        loss['degree_variance_raw'] = s_scalar['degree_variance']
        return loss

    def evaluate_single_epsilon(self, epsilon: float, n_runs: int = 5) -> Dict[str, Any]:
        print(f"\n🔹 Evaluating epsilon = {epsilon} | repeated runs {n_runs} time(s)")
        print("-" * 70)

        run_losses = []
        run_degree_variances = []

        for run in range(n_runs):
            synth_G = self.generate_synthetic_graph(epsilon)
            task_loss = self.compute_utility_loss(synth_G)
            run_losses.append(task_loss)
            run_degree_variances.append(task_loss['degree_variance_raw'])

            print(f"  Run {run + 1}/{n_runs} | overall utility loss: {task_loss['overall_utility_loss']:.4f} | synthetic graph degree variance: {task_loss['degree_variance_raw']:.6f}")

        avg_loss = {t: np.mean([rl[t] for rl in run_losses]) for t in self.all_tasks + ['overall_utility_loss']}
        std_loss = {t: np.std([rl[t] for rl in run_losses]) for t in self.all_tasks + ['overall_utility_loss']}

        avg_synth_degree_variance = np.mean(run_degree_variances)
        std_synth_degree_variance = np.std(run_degree_variances)

        result = {
            'epsilon': epsilon,
            'n_runs': n_runs,
            'avg_loss': avg_loss,
            'std_loss': std_loss,
            'original_degree_variance': self.original_degree_variance,
            'avg_synth_degree_variance': avg_synth_degree_variance,
            'std_synth_degree_variance': std_synth_degree_variance,
            'degree_variance_loss_avg': avg_loss['degree_variance'],
            'degree_variance_loss_std': std_loss['degree_variance']
        }

        print(f"\n📊 ε={epsilon} final results: ")
        print(f"  average overall utility loss: {avg_loss['overall_utility_loss']:.4f} (±{std_loss['overall_utility_loss']:.4f})")
        print(f"  original graph degree variance: {self.original_degree_variance:.6f}")
        print(f"  average synthetic graph degree variance: {avg_synth_degree_variance:.6f} (±{std_synth_degree_variance:.6f})")
        print(f"  degree variance relative error: {avg_loss['degree_variance']:.6f} (±{std_loss['degree_variance']:.6f})")
        return result

    def generate_final_table(self, all_results: List[Dict]) -> pd.DataFrame:
        table_rows = []
        for res in all_results:
            row = {
                'epsilon': res['epsilon'],
                'n_runs': res['n_runs'],
                'overall_utility_loss_avg': res['avg_loss']['overall_utility_loss'],
                'overall_utility_loss_std': res['std_loss']['overall_utility_loss'],
                'original_degree_variance': res['original_degree_variance'],
                'avg_synth_degree_variance': res['avg_synth_degree_variance'],
                'std_synth_degree_variance': res['std_synth_degree_variance'],
                'degree_variance_loss_avg': res['degree_variance_loss_avg'],
                'degree_variance_loss_std': res['degree_variance_loss_std']
            }
            for task in self.scalar_tasks:
                row[f'{task}_loss_avg'] = res['avg_loss'][task]
                row[f'{task}_loss_std'] = res['std_loss'][task]
            for task in self.dist_tasks:
                row[f'{task}_loss_avg'] = res['avg_loss'][task]
                row[f'{task}_loss_std'] = res['std_loss'][task]
            for task in self.struct_tasks:
                row[f'{task}_loss_avg'] = res['avg_loss'][task]
                row[f'{task}_loss_std'] = res['std_loss'][task]
            table_rows.append(row)
        return pd.DataFrame(table_rows)

    def save_results(self, df: pd.DataFrame, output_path: str, prefix: str):
        os.makedirs(output_path, exist_ok=True)
        filename = f"{prefix}_utility_loss.csv"
        path = os.path.join(output_path, filename)
        df.to_csv(path, index=False)
        print(f"\n💾 Results saved to: {path}")
        return path

    def run_evaluation(self, epsilon_list: List[float], n_runs: int = 5,
                       output_path: str = './utility_loss_results', prefix: str = 'exp'):
        start_time = time.time()
        print("\n" + "=" * 80)
        print("           Utility loss evaluation (paper formula + degree-variance extension)")
        print(f"total tasks: {len(self.all_tasks)} tasks (including degree variance) | epsilon list: {epsilon_list}")
        print("=" * 80)

        all_results = []
        for eps in epsilon_list:
            eps_result = self.evaluate_single_epsilon(eps, n_runs)
            all_results.append(eps_result)

        final_df = self.generate_final_table(all_results)
        self.save_results(final_df, output_path, prefix)

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"✅ Evaluation completed! Total elapsed time: {total_time:.2f}s | {total_time / 60:.2f}min")
        print("=" * 80)

        print("\n📌 Final summary of overall utility loss and degree-variance details: ")
        display_cols = ['epsilon', 'overall_utility_loss_avg', 'overall_utility_loss_std',
                        'original_degree_variance', 'avg_synth_degree_variance', 'degree_variance_loss_avg']
        print(final_df[display_cols].to_string(index=False))

        return final_df, all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility loss evaluation (paper formula + degree-variance extension)')
    parser.add_argument('--original_graph', required=True, type=str, help='Path to original graph file')
    parser.add_argument('--dpgs_module', required=True, type=str, help='Path to the DPGS synthesis module')
    parser.add_argument('--dpgs_func', required=True, type=str, help='DPGS synthesis function name')
    parser.add_argument('--epsilons', required=True, type=str, help='Comma-separated epsilon list, e.g., 0.1,1,10')
    parser.add_argument('--n_runs', default=5, type=int, help='Number of repeated runs per epsilon')
    parser.add_argument('--output_prefix', default='utility', type=str, help='Output file prefix')

    args = parser.parse_args()
    eps_list = [float(e.strip()) for e in args.epsilons.split(',')]

    evaluator = GraphUtilityLossEvaluator(
        original_graph_path=args.original_graph,
        dpgs_module_path=args.dpgs_module,
        dpgs_function_name=args.dpgs_func
    )

    evaluator.run_evaluation(
        epsilon_list=eps_list,
        n_runs=args.n_runs,
        output_path='./utility_loss_results',
        prefix=args.output_prefix
    )