

import sys
import os
from UTL import GraphUtilityLossEvaluator

SYNTHETIC_METHODS = {
    'DGG': {
        'module': './generators/DGG.py',
        'function': 'create_bter_graph_from_adjacency_matrix',
    },
    'DP1K': {
        'module': './generators/DP-1K.py',
        'function': 'generate_private_graph',
    },
    'PrivGraph': {
        'module': './generators/PrivGraph.py',
        'function': 'priv_graph',
    },
    'Tmf': {
        'module': './generators/Tmf.py',
        'function': 'top_m_filter',
    },
    'PrivHRG': {
        'module': './generators/PrivHRG.py',
        'function': 'privhrg_generate',
    },
    'PrivDPR': {
        'module': './generators/PrivDPR.py',
        'function': 'generate_synthetic_graph_from_adjmatrix',
    },
    'SKG': {
        'module': './generators/SKG.py',
        'function': 'dp_skg_from_matrix',
    }
}

def run_for_method(
    method_name: str,
    original_graph_path: str,
    epsilon_list: list,
    n_runs: int = 5
):

    method_info = SYNTHETIC_METHODS[method_name]
    module_path = method_info['module']
    func_name = method_info['function']

    print(f"\n\n==================================================")
    print(f"        Running: {method_name}")
    print(f"  Module: {module_path}")
    print(f"  Function: {func_name}")
    print(f"==================================================\n")

    # Initialize the utility-loss evaluator from UTL.py
    evaluator = GraphUtilityLossEvaluator(
        original_graph_path=original_graph_path,
        dpgs_module_path=module_path,
        dpgs_function_name=func_name
    )

    final_df, all_results = evaluator.run_evaluation(
        epsilon_list=epsilon_list,
        n_runs=n_runs,
        output_path='./utility_wikiV_loss_results',
        prefix=f"{method_name}_utility"
    )

    return final_df


def run_all_methods(
    original_graph_path: str,
    epsilon_list: list,
    n_runs: int = 5,
    run_only: list = None
):

    methods_to_run = run_only if run_only else list(SYNTHETIC_METHODS.keys())

    print(f"🚀 About to run utility-loss evaluation | method list: {methods_to_run}")
    print(f"📊 original graph: {original_graph_path}")
    print(f"🔒 Epsilon: {epsilon_list}")
    print(f"🔁 Runs per method: {n_runs}\n")

    for method in methods_to_run:
        if method not in SYNTHETIC_METHODS:
            print(f"⚠️  Skipping unknown method: {method}")
            continue

        run_for_method(
            method_name=method,
            original_graph_path=original_graph_path,
            epsilon_list=epsilon_list,
            n_runs=n_runs
        )

    print("\n✅ Utility-loss evaluation completed for all methods!")


if __name__ == "__main__":

    # ===================== Configure parameters here =====================
    ORIGINAL_GRAPH = "./data/Default.txt"  # Path to the original graph
    EPSILONS = [1,2,3,4,5,6,7,8,9,10,9999]       # Privacy budgets
    N_RUNS = 5                                   # Number of repeated runs
    # ===============================================================

    methods_from_cli = sys.argv[1:] if len(sys.argv) > 1 else None

    run_all_methods(
        original_graph_path=ORIGINAL_GRAPH,
        epsilon_list=EPSILONS,
        n_runs=N_RUNS,
        run_only=methods_from_cli
    )