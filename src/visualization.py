import networkx as nx
import numpy as np
import dionysus as d
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import FancyBboxPatch

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

def get_program_name(program_path):
    """Extract program name from the full path.
    
    Args:
        program_path (str): Full path to the program
        
    Returns:
        str: Program name without path and extension
    """
    return os.path.splitext(os.path.basename(program_path))[0]

def create_figure_1_global_cfg(global_analyzer, program_path, show_plots=True):
    """
    Figure 1: Global Control Flow Graph Structure
    Shows the hierarchical branching with edge frequencies
    
    Args:
        global_analyzer: The global analyzer containing CFG data
        program_path (str): Path to the test program
        show_plots (bool): Whether to display the plot. Defaults to True.
    """
    if not global_analyzer.global_cfg:
        global_analyzer.build_global_cfg()
        
    # Create figure with more vertical space for titles
    fig = plt.figure(figsize=(12, 10))
    
    # Create main axes with padding for title
    ax = fig.add_subplot(111)
    program_name = get_program_name(program_path)
    ax.set_title(f'Global Control Flow Graph - {program_name}\n(Node colors indicate type, edge thickness shows frequency)', 
                fontsize=14, weight='bold', pad=40)
    
    # Get root and distances
    root = global_analyzer.all_traces[0][0]
    distances = nx.shortest_path_length(global_analyzer.global_cfg, source=root)
    
    # Group nodes by distance for layout
    levels = {}
    for node, dist in distances.items():
        if dist not in levels:
            levels[dist] = []
        levels[dist].append(node)
    
    # Create positions
    pos = {}
    for level, nodes in levels.items():
        for i, node in enumerate(sorted(nodes)):
            # Spread nodes horizontally within each level
            x_offset = (i - len(nodes)/2) * 1.5
            pos[node] = (level * 3, x_offset)
    
    # Draw nodes with colors based on their role
    node_colors = []
    node_sizes = []
    for node in global_analyzer.global_cfg.nodes():
        out_degree = global_analyzer.global_cfg.out_degree(node)
        in_degree = global_analyzer.global_cfg.in_degree(node)
        
        if out_degree > 1:  # Branch point
            node_colors.append('lightcoral')
            node_sizes.append(1200)
        elif in_degree > 1:  # Convergence point
            node_colors.append('lightgreen')  
            node_sizes.append(1200)
        else:  # Regular node
            node_colors.append('lightblue')
            node_sizes.append(800)
    
    nx.draw_networkx_nodes(global_analyzer.global_cfg, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8, ax=ax)
    
    # Draw edges with thickness based on frequency
    edges = list(global_analyzer.global_cfg.edges(data=True))
    max_freq = max([data['frequency'] for _, _, data in edges]) if edges else 1
    
    for u, v, data in edges:
        freq = data['frequency']
        width = 1 + 4 * (freq / max_freq)  # Scale width 1-5
        alpha = 0.6 + 0.4 * (freq / max_freq)  # Higher frequency = more opaque
        nx.draw_networkx_edges(global_analyzer.global_cfg, pos, [(u, v)], width=width, 
                             alpha=alpha, edge_color='gray', ax=ax)
    
    # Add labels
    nx.draw_networkx_labels(global_analyzer.global_cfg, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Add edge frequency labels
    edge_labels = {(u, v): str(data['frequency']) 
                  for u, v, data in global_analyzer.global_cfg.edges(data=True) if data['frequency'] > 1}
    nx.draw_networkx_edge_labels(global_analyzer.global_cfg, pos, edge_labels, font_size=10, 
                                font_color='red', ax=ax)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                  markersize=12, label='Branch Points'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=12, label='Convergence Points'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=10, label='Regular Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add distance labels
    ax.axis('off')
    
    plt.savefig(f'figures/{program_name}_global_cfg_structure.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

def create_figure_2_persistence_diagram(global_analyzer, program_path, show_plots=True):
    """
    Figure 2: Global CFG Barcode Plot
    Shows the barcodes for H0 and H1 features
    
    Args:
        global_analyzer: The global analyzer containing CFG data
        program_path (str): Path to the test program
        show_plots (bool): Whether to display the plot. Defaults to True.
    """
    if not global_analyzer.global_cfg:
        global_analyzer.build_global_cfg()
    diagrams = global_analyzer.analyze_global_persistence()
    
    # Create figure with more vertical space for titles
    fig = plt.figure(figsize=(12, 8))
    program_name = get_program_name(program_path)
    fig.suptitle(f'Global CFG Barcodes - {program_name}', fontsize=16, fontweight='bold', y=1)
    
    # Create subplots with more space between them
    gs = plt.GridSpec(1, 2, figure=fig, wspace=0.3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    
    # H0
    if diagrams and len(diagrams) > 0 and diagrams[0]:
        d.plot.plot_bars(diagrams[0], ax=ax0)
        ax0.set_title('H₀ Barcode', pad=20)
    else:
        ax0.set_title('H₀ Barcode (None)', pad=20)
    
    # H1
    if diagrams and len(diagrams) > 1 and diagrams[1]:
        d.plot.plot_bars(diagrams[1], ax=ax1, color='red')
        ax1.set_title('H₁ Barcode', pad=20)
    else:
        ax1.set_title('H₁ Barcode (None)', pad=20)
    
    ax0.set_xlabel('Filtration Value')
    ax1.set_xlabel('Filtration Value')
    
    plt.savefig(f'figures/{program_name}_global_cfg_barcode.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

def create_all_figures(global_analyzer, program_path, show_plots=True):
    """Create all visualization figures for the program analysis.
    
    Args:
        global_analyzer: The global analyzer containing CFG data
        program_path (str): Path to the test program
        show_plots (bool): Whether to display the plots. Defaults to True.
    """
    create_figure_1_global_cfg(global_analyzer, program_path, show_plots)
    create_figure_2_persistence_diagram(global_analyzer, program_path, show_plots) 