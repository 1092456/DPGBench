#!/usr/bin/env python3

import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

SYNTHETIC_METHODS = {
    'PrivDPR': {
        'module': './generators/PrivDPR.py',
        'function': 'generate_synthetic_graph_from_adjmatrix',
    },
    'PrivHRG': {
        'module': './generators/PrivHRG.py',
        'function': 'privhrg_generate',
    },
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
    'SKG': {
        'module': './generators/SKG.py',
        'function': 'dp_skg_from_matrix',
    }
}

# Default experiment settings for each attack family.
DEFAULT_CONFIG = {
    'MIA': {
        'node_targets': 3,
        'edge_targets': 3,
        'attacks_per_target': 10,
        'epsilon_values': '0.01,1,999'
    },
    'AIA': {
        'node_targets': 3,
        'edge_targets': 3,
        'attacks_per_target': 20,
        'epsilon_values': '0.01,1,999'
    }
}


def setup_directories():
    """Create the output directories required by the benchmark runner."""
    dirs = ['data', 'generators', 'mia_results', 'aia_results', 'processed_data']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)


def run_xs(input_file, output_prefix):
    """Split the input graph into train and reference subgraphs."""
    print(f"\n{'=' * 60}")
    print("Step 1: Generate subgraphs")
    print(f"{'=' * 60}")

    output_file1 = f"{output_prefix}_in.txt"
    output_file2 = f"{output_prefix}_out.txt"

    cmd = [
        sys.executable, "XS.py",
        "--input", input_file,
        "--output1", output_file1,
        "--output2", output_file2
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        if Path(output_file1).exists() and Path(output_file2).exists():
            print("Subgraph generation completed successfully.")
            return output_file1, output_file2
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"Failed to run XS.py: {e}")
        return None, None


def _resolve_attack_config(attack_mode, args):
    """Resolve command-line overrides against the default attack configuration."""
    config = DEFAULT_CONFIG[attack_mode]
    node_targets = args.node_targets if args.node_targets is not None else config['node_targets']
    edge_targets = args.edge_targets if args.edge_targets is not None else config['edge_targets']
    attacks_per_target = args.attacks_per_target if args.attacks_per_target is not None else config['attacks_per_target']
    epsilon_values = args.epsilon_values if args.epsilon_values is not None else config['epsilon_values']
    return node_targets, edge_targets, attacks_per_target, epsilon_values


def _print_attack_config(synthetic_method, method_config, node_targets, edge_targets, attacks_per_target, epsilon_values):
    """Print the resolved attack configuration before launching an attack script."""
    print("\nAttack configuration:")
    print(f"  - Synthetic method: {synthetic_method}")
    print(f"  - Module file: {method_config['module']}")
    print(f"  - Function name: {method_config['function']}")
    print(f"  - Node targets: {node_targets}")
    print(f"  - Edge targets: {edge_targets}")
    print(f"  - Attacks per target: {attacks_per_target}")
    print(f"  - Privacy budgets: {epsilon_values}")


def run_mia(original_graph, reference_graph, synthetic_method, data_name, args):
    """Launch the membership inference attack evaluation."""
    print(f"\n{'=' * 60}")
    print("Step 2: Run MIA attack")
    print(f"{'=' * 60}")

    method_config = SYNTHETIC_METHODS.get(synthetic_method)
    if not method_config:
        print(f"Unknown synthetic graph method: {synthetic_method}")
        return False

    node_targets, edge_targets, attacks_per_target, epsilon_values = _resolve_attack_config('MIA', args)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"MIA_{data_name}_{synthetic_method}_N{node_targets}_E{edge_targets}_A{attacks_per_target}_{timestamp}"

    cmd = [
        sys.executable, "MIA.py",
        "--original_graph", original_graph,
        "--reference_graph", reference_graph,
        "--dpgs_module", method_config['module'],
        "--dpgs_function", method_config['function'],
        "--node_targets", str(node_targets),
        "--edge_targets", str(edge_targets),
        "--attacks_per_target", str(attacks_per_target),
        "--epsilon_values", epsilon_values,
        "--output_prefix", output_filename
    ]

    print(f"Command: {' '.join(cmd)}")
    _print_attack_config(synthetic_method, method_config, node_targets, edge_targets, attacks_per_target, epsilon_values)
    print("\nStarting MIA evaluation; live output follows.\n")
    print("-" * 60)

    try:
        subprocess.run(cmd, check=True)
        print("-" * 60)
        print("MIA evaluation completed.")
        print(f"  Result files: mia_results/{output_filename}_*.csv")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to run MIA.py: {e}")
        return False


def run_aia(original_graph, reference_graph, synthetic_method, data_name, args):
    """Launch the attribute inference attack evaluation."""
    print(f"\n{'=' * 60}")
    print("Step 2: Run AIA attack")
    print(f"{'=' * 60}")

    method_config = SYNTHETIC_METHODS.get(synthetic_method)
    if not method_config:
        print(f"Unknown synthetic graph method: {synthetic_method}")
        return False

    node_targets, edge_targets, attacks_per_target, epsilon_values = _resolve_attack_config('AIA', args)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"AIA_{data_name}_{synthetic_method}_N{node_targets}_E{edge_targets}_A{attacks_per_target}_{timestamp}"

    cmd = [
        sys.executable, "AIA.py",
        "--original_graph", original_graph,
        "--reference_graph", reference_graph,
        "--dpgs_module", method_config['module'],
        "--dpgs_function", method_config['function'],
        "--node_targets", str(node_targets),
        "--edge_targets", str(edge_targets),
        "--attacks_per_target", str(attacks_per_target),
        "--epsilon_values", epsilon_values,
        "--output_prefix", output_filename
    ]

    print(f"Command: {' '.join(cmd)}")
    _print_attack_config(synthetic_method, method_config, node_targets, edge_targets, attacks_per_target, epsilon_values)
    print("\nStarting AIA evaluation; live output follows.\n")
    print("-" * 60)

    try:
        subprocess.run(cmd, check=True)
        print("-" * 60)
        print("AIA evaluation completed.")
        print(f"  Result files: aia_results/{output_filename}_*.csv")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to run AIA.py: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run graph attack evaluations for DPGBench.')

    parser.add_argument('--attack_mode', type=str, required=True,
                        choices=['MIA', 'AIA'],
                        help='Attack mode: MIA for membership inference or AIA for attribute inference.')

    parser.add_argument('--data_name', type=str, required=True,
                        help='Dataset name, for example: Wiki-Vote.')

    parser.add_argument('--synthetic_method', type=str, default='PrivDPR',
                        choices=list(SYNTHETIC_METHODS.keys()),
                        help=f'Synthetic graph method. Default: PrivDPR. Options: {", ".join(SYNTHETIC_METHODS.keys())}.')

    # Optional arguments override the default settings for the selected attack mode.
    parser.add_argument('--node_targets', type=int, default=None,
                        help='Number of node targets. Default: MIA=3, AIA=3.')

    parser.add_argument('--edge_targets', type=int, default=None,
                        help='Number of edge targets. Default: MIA=3, AIA=3.')

    parser.add_argument('--attacks_per_target', type=int, default=None,
                        help='Number of attacks per target. Default: MIA=10, AIA=20.')

    parser.add_argument('--epsilon_values', type=str, default=None,
                        help='Comma-separated privacy budgets, e.g., 0.01,1,999. Default: MIA="0.01,1,999", AIA="0.01,1,999".')

    args = parser.parse_args()

    setup_directories()

    input_file = f"data/{args.data_name}.txt"
    output_prefix = f"processed_data/{args.data_name}_{args.synthetic_method}"

    # Validate that the input graph exists before preprocessing.
    if not Path(input_file).exists():
        print(f"Original graph file does not exist: {input_file}")
        print(f"Please place {args.data_name}.txt under the data/ directory.")
        return

    print(f"\n{'#' * 60}")
    print("Experiment configuration:")
    print(f"{'#' * 60}")
    print(f"Attack mode: {args.attack_mode}")
    print(f"Dataset: {args.data_name}")
    print(f"Synthetic graph method: {args.synthetic_method}")
    print(f"{'#' * 60}\n")

    train_graph, test_graph = run_xs(input_file, output_prefix)

    if not train_graph or not test_graph:
        print("Subgraph generation failed.")
        return

    if args.attack_mode == 'MIA':
        success = run_mia(train_graph, test_graph, args.synthetic_method, args.data_name, args)
    else:
        success = run_aia(train_graph, test_graph, args.synthetic_method, args.data_name, args)

    if success:
        print(f"\n{'#' * 60}")
        print("Experiment completed.")
        print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
