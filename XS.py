import random
import argparse


def generate_two_non_overlapping_subgraphs(input_file, output_file1, output_file2,
                                           total_nodes=80, split_ratio=0.5):
    """
    生成两个节点互不重复的子图，并重新编号为从0开始的连续节点

    参数:
    input_file: 输入图文件路径
    output_file1: 第一个子图输出文件路径
    output_file2: 第二个子图输出文件路径
    total_nodes: 总共要选择的节点数量
    split_ratio: 第一个子图占总节点的比例
    """

    # 读取原始图文件
    edges = []
    all_nodes = set()

    print(f"正在读取图文件: {input_file}")
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        node1 = int(parts[0])
                        node2 = int(parts[1])
                        edges.append((node1, node2))
                        all_nodes.add(node1)
                        all_nodes.add(node2)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    print(f"原始图信息:")
    print(f"  节点总数: {len(all_nodes)}")
    print(f"  边总数: {len(edges)}")

    # 计算每个子图的节点数量
    num_nodes1 = int(total_nodes * split_ratio)
    num_nodes2 = total_nodes - num_nodes1

    print(f"第一个子图节点数: {num_nodes1}")
    print(f"第二个子图节点数: {num_nodes2}")
    print(f"总共选择节点数: {total_nodes}")

    # 检查是否有足够的节点
    if len(all_nodes) < total_nodes:
        print(f"错误: 原始图只有 {len(all_nodes)} 个节点，无法选择 {total_nodes} 个不同节点")
        return

    # 随机选择节点（确保不重复）
    all_nodes_list = list(all_nodes)
    random.shuffle(all_nodes_list)

    # 选择第一个子图的节点
    selected_nodes1 = set(all_nodes_list[:num_nodes1])
    # 选择第二个子图的节点（从剩余节点中选择）
    selected_nodes2 = set(all_nodes_list[num_nodes1:num_nodes1 + num_nodes2])

    print(f"✓ 第一个子图选择了 {len(selected_nodes1)} 个节点")
    print(f"✓ 第二个子图选择了 {len(selected_nodes2)} 个节点")

    # 验证节点不重复
    common_nodes = selected_nodes1.intersection(selected_nodes2)
    if not common_nodes:
        print("✓ 验证通过：两个子图的节点完全互不相同")
    else:
        print(f"错误: 发现 {len(common_nodes)} 个重复节点")
        return

    # 为每个子图提取边并去重
    def extract_subgraph_edges(selected_nodes):
        """提取子图的边，并去除重复边"""
        subgraph_edges_set = set()  # 使用集合自动去重
        for node1, node2 in edges:
            if node1 in selected_nodes and node2 in selected_nodes:
                # 标准化边表示：确保较小的节点在前（可选，有助于进一步去重）
                edge = (min(node1, node2), max(node1, node2))
                subgraph_edges_set.add(edge)
        return list(subgraph_edges_set)

    subgraph_edges1 = extract_subgraph_edges(selected_nodes1)
    subgraph_edges2 = extract_subgraph_edges(selected_nodes2)

    print(f"\n提取边并去重后:")
    print(f"  第一个子图边数: {len(subgraph_edges1)}")
    print(f"  第二个子图边数: {len(subgraph_edges2)}")

    # ====================== 核心修改开始 ======================
    # 重新编号函数：保证节点数 = 设定值，保留所有选中节点
    def renumber_nodes(edges_list, selected_nodes, start_id=0):
        """
        重新编号节点，保证包含所有selected_nodes，从start_id开始连续编号
        """
        # 固定使用所有选中的节点，保证数量 = 设定值
        sorted_original_nodes = sorted(selected_nodes)
        node_mapping = {old: new + start_id for new, old in enumerate(sorted_original_nodes)}

        # 处理边
        renumbered_edges_set = set()
        for node1, node2 in edges_list:
            new1 = node_mapping[node1]
            new2 = node_mapping[node2]
            edge = (min(new1, new2), max(new1, new2))
            renumbered_edges_set.add(edge)

        renumbered_edges = sorted(renumbered_edges_set, key=lambda x: (x[0], x[1]))
        return renumbered_edges, node_mapping

    # 子图1 从 0 开始编号
    renumbered_edges1, mapping1 = renumber_nodes(subgraph_edges1, selected_nodes1, start_id=0)
    # 子图2 从 num_nodes1 开始编号（接在子图1后面）
    renumbered_edges2, mapping2 = renumber_nodes(subgraph_edges2, selected_nodes2, start_id=num_nodes1)
    # ====================== 核心修改结束 ======================

    # 显示统计信息
    print(f"\n第一个子图统计:")
    print(f"  原始节点数: {len(selected_nodes1)}")
    print(f"  重新编号后的节点数: {len(set(node for edge in renumbered_edges1 for node in edge))}")
    print(f"  去重后的边数: {len(renumbered_edges1)}")
    if mapping1:
        print(f"  节点编号范围: 0 到 {max(mapping1.values())}")

    print(f"\n第二个子图统计:")
    print(f"  原始节点数: {len(selected_nodes2)}")
    print(f"  重新编号后的节点数: {len(set(node for edge in renumbered_edges2 for node in edge))}")
    print(f"  去重后的边数: {len(renumbered_edges2)}")
    if mapping2:
        print(f"  节点编号范围: {min(mapping2.values())} 到 {max(mapping2.values())}")

    # 检查是否有自环边
    def check_self_loops(edges_list, graph_name):
        self_loops = [(n1, n2) for n1, n2 in edges_list if n1 == n2]
        if self_loops:
            print(
                f"  ⚠ {graph_name} 发现 {len(self_loops)} 条自环边: {self_loops[:5]}{'...' if len(self_loops) > 5 else ''}")
            return True
        return False

    has_self_loops1 = check_self_loops(renumbered_edges1, "子图1")
    has_self_loops2 = check_self_loops(renumbered_edges2, "子图2")

    # 移除自环边（可选）
    def remove_self_loops(edges_list):
        return [(n1, n2) for n1, n2 in edges_list if n1 != n2]

    if has_self_loops1 or has_self_loops2:
        print("正在移除自环边...")
        renumbered_edges1 = remove_self_loops(renumbered_edges1)
        renumbered_edges2 = remove_self_loops(renumbered_edges2)
        print(f"移除后: 子图1边数={len(renumbered_edges1)}, 子图2边数={len(renumbered_edges2)}")

    # 保存子图函数
    def save_subgraph(filename, renumbered_edges):
        try:
            with open(filename, 'w') as f:
                # 边已经按升序排列
                for node1, node2 in renumbered_edges:
                    f.write(f"{node1} {node2}\n")
            print(f"✓ 文件已保存: {filename}")
            return True
        except Exception as e:
            print(f"保存文件 {filename} 时出错: {e}")
            return False

    print(f"\n正在保存重新编号后的子图文件...")
    success1 = save_subgraph(output_file1, renumbered_edges1)
    success2 = save_subgraph(output_file2, renumbered_edges2)

    if success1 and success2:
        print(f"\n✓ 两个重新编号的子图已成功生成!")
        print(f"  文件1: {output_file1} ({len(renumbered_edges1)} 条边)")
        print(f"  文件2: {output_file2} ({len(renumbered_edges2)} 条边)")

        # 验证节点编号从0开始
        print(f"\n✓ 验证节点编号:")

        # 检查第一个子图
        nodes1 = set()
        for node1, node2 in renumbered_edges1:
            nodes1.add(node1)
            nodes1.add(node2)

        nodes2 = set()
        for node1, node2 in renumbered_edges2:
            nodes2.add(node1)
            nodes2.add(node2)

        min_node1 = min(nodes1) if nodes1 else 0
        max_node1 = max(nodes1) if nodes1 else 0
        min_node2 = min(nodes2) if nodes2 else 0
        max_node2 = max(nodes2) if nodes2 else 0

        print(f"  子图1节点范围: {min_node1} 到 {max_node1}")
        print(f"  子图2节点范围: {min_node2} 到 {max_node2}")

        if min_node1 == 0 and min_node2 == num_nodes1:
            print(f"  ✓ 两个子图节点编号连续衔接")
        else:
            print(f"  ⚠ 节点编号未从0开始，请检查")

    else:
        print(f"\n✗ 子图生成过程中出现问题")


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='生成两个节点不重叠的子图')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图文件路径')
    parser.add_argument('--output1', type=str, required=True,
                        help='第一个子图输出文件路径')
    parser.add_argument('--output2', type=str, required=True,
                        help='第二个子图输出文件路径')
    parser.add_argument('--total_nodes', type=int, default=1000,
                        help='总共要选择的节点数量')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help='第一个子图占总节点的比例')

    args = parser.parse_args()

    # 使用命令行参数运行
    generate_two_non_overlapping_subgraphs(
        input_file=args.input,
        output_file1=args.output1,
        output_file2=args.output2,
        total_nodes=args.total_nodes,
        split_ratio=args.split_ratio
    )