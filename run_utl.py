#!/usr/bin/env python3
# run_utility_evaluation.py
# 效用损失批量评估脚本（适配 PrivDPR / PrivHRG / DGG / DP1K / PrivGraph / Tmf / SKG）

import sys
import os
from UTL import GraphUtilityLossEvaluator

# ====================== 你提供的合成方法映射表 ======================
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
# ====================================================================

def run_for_method(
    method_name: str,
    original_graph_path: str,
    epsilon_list: list,
    n_runs: int = 5
):
    """
    对单个合成方法运行效用损失评估
    完全复用 UTL.py 的逻辑
    """
    method_info = SYNTHETIC_METHODS[method_name]
    module_path = method_info['module']
    func_name = method_info['function']

    print(f"\n\n==================================================")
    print(f"        正在运行：{method_name}")
    print(f"  模块：{module_path}")
    print(f"  函数：{func_name}")
    print(f"==================================================\n")

    # 初始化效用损失评估器（来自 UTL.py）
    evaluator = GraphUtilityLossEvaluator(
        original_graph_path=original_graph_path,
        dpgs_module_path=module_path,
        dpgs_function_name=func_name
    )

    # 运行评估
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
    """
    批量运行所有/指定合成方法
    """
    methods_to_run = run_only if run_only else list(SYNTHETIC_METHODS.keys())

    print(f"🚀 即将运行效用损失评估 | 方法列表：{methods_to_run}")
    print(f"📊 原始图：{original_graph_path}")
    print(f"🔒 Epsilon：{epsilon_list}")
    print(f"🔁 每方法运行次数：{n_runs}\n")

    for method in methods_to_run:
        if method not in SYNTHETIC_METHODS:
            print(f"⚠️  跳过未知方法：{method}")
            continue

        run_for_method(
            method_name=method,
            original_graph_path=original_graph_path,
            epsilon_list=epsilon_list,
            n_runs=n_runs
        )

    print("\n✅ 所有方法的效用损失评估全部完成！")


if __name__ == "__main__":

    # ===================== 请在这里配置你的参数 =====================
    ORIGINAL_GRAPH = "./data/Default.txt"  # 你的原始图路径
    EPSILONS = [1,2,3,4,5,6,7,8,9,10,9999]       # 隐私预算
    N_RUNS = 5                                   # 重复次数
    # ===============================================================

    # 读取命令行传入的需要运行的方法
    methods_from_cli = sys.argv[1:] if len(sys.argv) > 1 else None

    # 开始运行
    run_all_methods(
        original_graph_path=ORIGINAL_GRAPH,
        epsilon_list=EPSILONS,
        n_runs=N_RUNS,
        run_only=methods_from_cli
    )