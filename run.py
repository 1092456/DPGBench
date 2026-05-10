#!/usr/bin/env python3


import os
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

# 默认参数配置
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

    dirs = ['data', 'generators', 'mia_results', 'aia_results', 'processed_data']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)


def run_xs(input_file, output_prefix):

    print(f"\n{'=' * 60}")
    print(f"步骤1: 生成子图")
    print(f"{'=' * 60}")

    output_file1 = f"{output_prefix}_in.txt"
    output_file2 = f"{output_prefix}_out.txt"

    cmd = [
        sys.executable, "XS.py",
        "--input", input_file,
        "--output1", output_file1,
        "--output2", output_file2
    ]

    print(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        if Path(output_file1).exists() and Path(output_file2).exists():
            print(f"✓ 子图生成成功")
            return output_file1, output_file2
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"✗ 运行XS.py失败: {e}")
        return None, None


def run_mia(original_graph, reference_graph, synthetic_method, data_name, args):
    print(f"\n{'=' * 60}")
    print(f"步骤2: 运行MIA攻击")
    print(f"{'=' * 60}")

    method_config = SYNTHETIC_METHODS.get(synthetic_method)
    if not method_config:
        print(f"✗ 未知的合成图方法: {synthetic_method}")
        return False

    node_targets = args.node_targets if args.node_targets is not None else DEFAULT_CONFIG['MIA']['node_targets']
    edge_targets = args.edge_targets if args.edge_targets is not None else DEFAULT_CONFIG['MIA']['edge_targets']
    attacks_per_target = args.attacks_per_target if args.attacks_per_target is not None else DEFAULT_CONFIG['MIA'][
        'attacks_per_target']
    epsilon_values = args.epsilon_values if args.epsilon_values is not None else DEFAULT_CONFIG['MIA']['epsilon_values']

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

    print(f"执行命令: {' '.join(cmd)}")
    print(f"\n攻击配置:")
    print(f"  - 合成图方法: {synthetic_method}")
    print(f"  - 模块文件: {method_config['module']}")
    print(f"  - 函数名称: {method_config['function']}")
    print(f"  - 节点目标数: {node_targets}")
    print(f"  - 边目标数: {edge_targets}")
    print(f"  - 攻击次数/目标: {attacks_per_target}")
    print(f"  - 隐私预算: {epsilon_values}")
    print(f"\n开始运行MIA攻击，实时输出如下:\n")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print(f"✓ MIA攻击完成")
        print(f"  结果文件: mia_results/{output_filename}_*.csv")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 运行MIA.py失败: {e}")
        return False


def run_aia(original_graph, reference_graph, synthetic_method, data_name, args):

    print(f"\n{'=' * 60}")
    print(f"步骤2: 运行AIA攻击")
    print(f"{'=' * 60}")

    method_config = SYNTHETIC_METHODS.get(synthetic_method)
    if not method_config:
        print(f"✗ 未知的合成图方法: {synthetic_method}")
        return False

    node_targets = args.node_targets if args.node_targets is not None else DEFAULT_CONFIG['AIA']['node_targets']
    edge_targets = args.edge_targets if args.edge_targets is not None else DEFAULT_CONFIG['AIA']['edge_targets']
    attacks_per_target = args.attacks_per_target if args.attacks_per_target is not None else DEFAULT_CONFIG['AIA'][
        'attacks_per_target']
    epsilon_values = args.epsilon_values if args.epsilon_values is not None else DEFAULT_CONFIG['AIA']['epsilon_values']

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

    print(f"执行命令: {' '.join(cmd)}")
    print(f"\n攻击配置:")
    print(f"  - 合成图方法: {synthetic_method}")
    print(f"  - 模块文件: {method_config['module']}")
    print(f"  - 函数名称: {method_config['function']}")
    print(f"  - 节点目标数: {node_targets}")
    print(f"  - 边目标数: {edge_targets}")
    print(f"  - 攻击次数/目标: {attacks_per_target}")
    print(f"  - 隐私预算: {epsilon_values}")
    print(f"\n开始运行AIA攻击，实时输出如下:\n")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print(f"✓ AIA攻击完成")
        print(f"  结果文件: aia_results/{output_filename}_*.csv")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 运行AIA.py失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='图攻击实验运行脚本')


    parser.add_argument('--attack_mode', type=str, required=True,
                        choices=['MIA', 'AIA'],
                        help='攻击模式: MIA(成员推断) 或 AIA(属性推断)')

    parser.add_argument('--data_name', type=str, required=True,
                        help='数据集名称，例如: Wiki-Vote')


    parser.add_argument('--synthetic_method', type=str, default='PrivDPR',
                        choices=list(SYNTHETIC_METHODS.keys()),
                        help=f'合成图方法 (默认: PrivDPR, 可选: {", ".join(SYNTHETIC_METHODS.keys())})')

    # 可选参数（用于覆盖默认值）
    parser.add_argument('--node_targets', type=int, default=None,
                        help='节点目标数量 (默认: MIA=3, AIA=3)')

    parser.add_argument('--edge_targets', type=int, default=None,
                        help='边目标数量 (默认: MIA=3, AIA=3)')

    parser.add_argument('--attacks_per_target', type=int, default=None,
                        help='每个目标的攻击次数 (默认: MIA=10, AIA=20)')

    parser.add_argument('--epsilon_values', type=str, default=None,
                        help='隐私预算列表，用逗号分隔，例如: 0.01,1,999 (默认: MIA="0.01,1,999", AIA="0.01,1,999")')

    args = parser.parse_args()


    setup_directories()


    input_file = f"data/{args.data_name}.txt"
    output_prefix = f"processed_data/{args.data_name}_{args.synthetic_method}"

    # 检查原始图文件
    if not Path(input_file).exists():
        print(f"✗ 原始图文件不存在: {input_file}")
        print(f"   请将 {args.data_name}.txt 文件放在 data/ 目录下")
        return

    print(f"\n{'#' * 60}")
    print(f"实验配置:")
    print(f"{'#' * 60}")
    print(f"攻击模式: {args.attack_mode}")
    print(f"数据集: {args.data_name}")
    print(f"合成图方法: {args.synthetic_method}")
    print(f"{'#' * 60}\n")


    train_graph, test_graph = run_xs(input_file, output_prefix)

    if not train_graph or not test_graph:
        print("✗ 子图生成失败")
        return


    if args.attack_mode == 'MIA':
        success = run_mia(train_graph, test_graph, args.synthetic_method, args.data_name, args)
    else:
        success = run_aia(train_graph, test_graph, args.synthetic_method, args.data_name, args)

    if success:
        print(f"\n{'#' * 60}")
        print(f"✓ 实验完成!")
        print(f"{'#' * 60}")


if __name__ == "__main__":
    main()