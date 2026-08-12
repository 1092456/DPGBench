import random
import argparse


def generate_two_non_overlapping_subgraphs(input_file, output_file1, output_file2,
                                           total_nodes=80, split_ratio=0.5):
    """
    Generate two node-disjoint subgraphs and relabel nodes to consecutive IDs starting from 0

    Args:
    input_file: Input graph file path
    output_file1: Output file path for the first subgraph
    output_file2: Output file path for the second subgraph
    total_nodes: Total number of nodes to sample
    split_ratio: Fraction of sampled nodes assigned to the first subgraph
    """

    # Read the original graph file
    edges = []
    all_nodes = set()

    print(f"Reading graph file: {input_file}")
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
        print(f"Error: file not found {input_file}")
        return
    except Exception as e:
        print(f"Error while reading file: {e}")
        return

    print(f"Original graph information:")
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Total edges: {len(edges)}")

    # Compute the number of nodes in each subgraph
    num_nodes1 = int(total_nodes * split_ratio)
    num_nodes2 = total_nodes - num_nodes1

    print(f"First subgraph nodes: {num_nodes1}")
    print(f"Second subgraph nodes: {num_nodes2}")
    print(f"Total selected nodes: {total_nodes}")

    # Check whether enough nodes are available
    if len(all_nodes) < total_nodes:
        print(f"Error: original graph has only {len(all_nodes)} nodes, cannot select {total_nodes} distinct nodes")
        return

    # Randomly select nodes without overlap
    all_nodes_list = list(all_nodes)
    random.shuffle(all_nodes_list)

    # Select nodes for the first subgraph
    selected_nodes1 = set(all_nodes_list[:num_nodes1])
    # Select nodes for the second subgraph from the remaining nodes
    selected_nodes2 = set(all_nodes_list[num_nodes1:num_nodes1 + num_nodes2])

    print(f"✓ First subgraph selected {len(selected_nodes1)} nodes")
    print(f"✓ Second subgraph selected {len(selected_nodes2)} nodes")

    # Verify that the selected node sets do not overlap
    common_nodes = selected_nodes1.intersection(selected_nodes2)
    if not common_nodes:
        print("✓ Validation passed: the two subgraphs have disjoint node sets")
    else:
        print(f"Error: found {len(common_nodes)} overlapping nodes")
        return

    # Extract and deduplicate edges for each subgraph
    def extract_subgraph_edges(selected_nodes):
        """Extract subgraph edges and remove duplicates"""
        subgraph_edges_set = set()  # Use a set to deduplicate edges automatically
        for node1, node2 in edges:
            if node1 in selected_nodes and node2 in selected_nodes:
                # Normalize edge order by placing the smaller node first to aid deduplication
                edge = (min(node1, node2), max(node1, node2))
                subgraph_edges_set.add(edge)
        return list(subgraph_edges_set)

    subgraph_edges1 = extract_subgraph_edges(selected_nodes1)
    subgraph_edges2 = extract_subgraph_edges(selected_nodes2)

    print(f"\nAfter edge extraction and deduplication:")
    print(f"  First subgraph edges: {len(subgraph_edges1)}")
    print(f"  Second subgraph edges: {len(subgraph_edges2)}")


    def renumber_nodes(edges_list, selected_nodes, start_id=0):
        sorted_original_nodes = sorted(selected_nodes)
        node_mapping = {old: new + start_id for new, old in enumerate(sorted_original_nodes)}

        renumbered_edges_set = set()
        for node1, node2 in edges_list:
            new1 = node_mapping[node1]
            new2 = node_mapping[node2]
            edge = (min(new1, new2), max(new1, new2))
            renumbered_edges_set.add(edge)

        renumbered_edges = sorted(renumbered_edges_set, key=lambda x: (x[0], x[1]))
        return renumbered_edges, node_mapping

    renumbered_edges1, mapping1 = renumber_nodes(subgraph_edges1, selected_nodes1, start_id=0)
    renumbered_edges2, mapping2 = renumber_nodes(subgraph_edges2, selected_nodes2, start_id=num_nodes1)

    # Display summary statistics
    print(f"\nFirst subgraph statistics:")
    print(f"  Original node count: {len(selected_nodes1)}")
    print(f"  Relabeled node count: {len(set(node for edge in renumbered_edges1 for node in edge))}")
    print(f"  Deduplicated edge count: {len(renumbered_edges1)}")
    if mapping1:
        print(f"  Node ID range: 0 to {max(mapping1.values())}")

    print(f"\nSecond subgraph statistics:")
    print(f"  Original node count: {len(selected_nodes2)}")
    print(f"  Relabeled node count: {len(set(node for edge in renumbered_edges2 for node in edge))}")
    print(f"  Deduplicated edge count: {len(renumbered_edges2)}")
    if mapping2:
        print(f"  Node ID range: {min(mapping2.values())} to {max(mapping2.values())}")

    def check_self_loops(edges_list, graph_name):
        self_loops = [(n1, n2) for n1, n2 in edges_list if n1 == n2]
        if self_loops:
            print(
                f"  ⚠ {graph_name} found {len(self_loops)} self-loop edges: {self_loops[:5]}{'...' if len(self_loops) > 5 else ''}")
            return True
        return False

    has_self_loops1 = check_self_loops(renumbered_edges1, "subgraph 1")
    has_self_loops2 = check_self_loops(renumbered_edges2, "subgraph 2")

    def remove_self_loops(edges_list):
        return [(n1, n2) for n1, n2 in edges_list if n1 != n2]

    if has_self_loops1 or has_self_loops2:
        print("Removing self-loop edges...")
        renumbered_edges1 = remove_self_loops(renumbered_edges1)
        renumbered_edges2 = remove_self_loops(renumbered_edges2)
        print(f"After removal: subgraph 1edge count={len(renumbered_edges1)}, subgraph 2edge count={len(renumbered_edges2)}")

    def save_subgraph(filename, renumbered_edges):
        try:
            with open(filename, 'w') as f:
                for node1, node2 in renumbered_edges:
                    f.write(f"{node1} {node2}\n")
            print(f"✓ File saved: {filename}")
            return True
        except Exception as e:
            print(f"Error saving file {filename} : {e}")
            return False

    print(f"\nSaving relabeled subgraph files...")
    success1 = save_subgraph(output_file1, renumbered_edges1)
    success2 = save_subgraph(output_file2, renumbered_edges2)

    if success1 and success2:
        print(f"\n✓ Two relabeled subgraphs were generated successfully!")
        print(f"  File 1: {output_file1} ({len(renumbered_edges1)} edges)")
        print(f"  File 2: {output_file2} ({len(renumbered_edges2)} edges)")

        print(f"\n✓ Validate node IDs:")

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

        print(f"  subgraph 1node range: {min_node1} to {max_node1}")
        print(f"  subgraph 2node range: {min_node2} to {max_node2}")

        if min_node1 == 0 and min_node2 == num_nodes1:
            print(f"  ✓ The two subgraph node ID ranges are consecutive")
        else:
            print(f"  ⚠ Node IDs do not start from 0; please check")

    else:
        print(f"\n✗ Subgraph generation encountered a problem")


if __name__ == "__main__":
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description='Generate two node-disjoint subgraphs')
    parser.add_argument('--input', type=str, required=True,
                        help='Input graph file path')
    parser.add_argument('--output1', type=str, required=True,
                        help='Output file path for the first subgraph')
    parser.add_argument('--output2', type=str, required=True,
                        help='Output file path for the second subgraph')
    parser.add_argument('--total_nodes', type=int, default=1000,
                        help='Total number of nodes to sample')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help='Fraction of sampled nodes assigned to the first subgraph')

    args = parser.parse_args()

    generate_two_non_overlapping_subgraphs(
        input_file=args.input,
        output_file1=args.output1,
        output_file2=args.output2,
        total_nodes=args.total_nodes,
        split_ratio=args.split_ratio
    )