import networkx as nx
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import os

def trace_to_graph(trace: List[int]) -> nx.DiGraph:
    """Converts a trace into a NetworkX directed graph.
    
    This function creates a directed graph where:
    - Nodes represent program locations
    - Edges represent transitions between locations
    - The graph preserves the sequential order of the trace
    
    Args:
        trace: List of integers representing the sequence of nodes in the trace
        
    Returns:
        A NetworkX directed graph where edges represent transitions between nodes
    """
    G = nx.DiGraph()
    if not trace:
        return G
    
    # Add all nodes first to ensure they exist
    for node in trace:
        G.add_node(node)
    
    # Add edges between consecutive nodes
    for i in range(len(trace) - 1):
        G.add_edge(trace[i], trace[i+1])
    
    return G

def graph_to_trace(G: nx.DiGraph, start_node: Optional[int] = None) -> List[int]:
    """Converts a graph into a trace by following edges from a start node.
    
    This function creates a trace by:
    1. Starting at the specified node (or finding a suitable start)
    2. Following outgoing edges until reaching a node with no outgoing edges
    
    Args:
        G: NetworkX directed graph
        start_node: Optional starting node. If None, uses the first node with no incoming edges
        
    Returns:
        List of integers representing the sequence of nodes in the trace
    """
    if not G.nodes():
        return []
    
    if start_node is None:
        # Find nodes with no incoming edges
        start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not start_nodes:
            # If no nodes have zero in-degree, use the first node
            start_node = list(G.nodes())[0]
        else:
            start_node = start_nodes[0]
    
    trace = [start_node]
    current = start_node
    
    while G.out_degree(current) > 0:
        # Get the next node (assuming there's only one outgoing edge)
        next_node = list(G.successors(current))[0]
        trace.append(next_node)
        current = next_node
    
    return trace

def merge_graphs(graphs: List[nx.DiGraph]) -> nx.DiGraph:
    """Merges multiple graphs into a single graph.
    
    This function combines multiple graphs by:
    1. Adding all nodes from each graph
    2. Adding all edges from each graph
    3. Preserving the directed nature of the graphs
    
    Args:
        graphs: List of NetworkX directed graphs
        
    Returns:
        A single NetworkX directed graph containing all nodes and edges
    """
    merged = nx.DiGraph()
    
    for G in graphs:
        # Add all nodes and edges from each graph
        merged.add_nodes_from(G.nodes())
        merged.add_edges_from(G.edges())
    
    return merged

def get_graph_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """Returns statistics about the graph.
    
    This function computes various graph metrics including:
    - Node and edge counts
    - Directed/acyclic properties
    - Node degrees
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Dictionary containing graph statistics including:
        - num_nodes: Number of nodes
        - num_edges: Number of edges
        - is_directed: Whether the graph is directed
        - is_dag: Whether the graph is a DAG
        - has_cycles: Whether the graph contains cycles
        - in_degrees: Dictionary of node in-degrees
        - out_degrees: Dictionary of node out-degrees
    """
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'is_directed': G.is_directed(),
        'is_dag': nx.is_directed_acyclic_graph(G),
        'has_cycles': not nx.is_directed_acyclic_graph(G),
        'in_degrees': dict(G.in_degree()),
        'out_degrees': dict(G.out_degree())
    }

def get_edge_labels() -> Dict[int, str]:
    """Returns a mapping of edge IDs to their meanings in the test program.
    
    This function provides a mapping between edge IDs and their semantic meaning
    in the test program, including:
    - Entry points
    - Path conditions
    - Processing steps
    - Output types
    
    Returns:
        Dictionary mapping edge IDs to their semantic descriptions
    """
    return {
        5: "Common entry point",
        7: "Very negative path (< -10)",
        8: "Slightly negative path (-10 to -1)",
        9: "Zero path",
        10: "Very positive path (> 10)",
        11: "Slightly positive path (1 to 10)",
        12: "Common processing",
        13: "Common processing",
        14: "Very negative output",
        15: "Slightly negative output",
        16: "Zero output",
        17: "Common positive processing",
        18: "Very positive output",
        19: "Slightly positive output"
    }

def save_graph_visualization(G: nx.DiGraph, filename: str, output_dir: str = 'plots') -> None:
    """Saves a visualization of the graph to a file.
    
    This function creates a visual representation of the graph including:
    - Node layout using spring layout
    - Edge arrows showing direction
    - Node labels with edge meanings
    - Graph statistics in the title
    
    Args:
        G: NetworkX directed graph to visualize
        filename: Name of the output file (without extension)
        output_dir: Directory to save the plot in (default: 'plots')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with a larger size
    plt.figure(figsize=(15, 10))
    
    # Use spring layout for better node positioning
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.6)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    
    # Get edge labels
    edge_labels = get_edge_labels()
    
    # Create node labels with edge meanings
    node_labels = {node: f"{node}\n{edge_labels.get(node, '')}" 
                  for node in G.nodes()}
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    # Add title with graph statistics
    stats = get_graph_stats(G)
    plt.title(f'Graph Visualization\n'
             f'Nodes: {stats["num_nodes"]}, Edges: {stats["num_edges"]}\n'
             f'Is DAG: {stats["is_dag"]}, Has Cycles: {stats["has_cycles"]}')
    
    # Remove axis
    plt.axis('off')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), 
                bbox_inches='tight', dpi=300)
    plt.close() 