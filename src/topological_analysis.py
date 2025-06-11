#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import dionysus as d
from analysis import HybridTDAAnalyzer
from AFL_utils import AFL_Utils
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster

def safe_pd_distance(pd1, pd2, metric='bottleneck'):
    """Safely compute distance between two persistence diagrams."""
    if pd1 is None or len(pd1) == 0:
        if pd2 is None or len(pd2) == 0:
            return 0.0
        else:
            return float('inf')
    
    if pd2 is None or len(pd2) == 0:
        return float('inf')
    
    try:
        if metric == 'bottleneck':
            return d.bottleneck_distance(pd1, pd2)
        elif metric == 'wasserstein':
            return d.wasserstein_distance(pd1, pd2, q=2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    except Exception as e:
        print(f"Warning: Distance calculation failed ({e}), using feature count difference")
        return abs(len(pd1) - len(pd2))

def count_finite_features(pd):
    """Count features with finite death times."""
    if pd is None or len(pd) == 0:
        return 0
    return sum(1 for pt in pd if pt.death != float('inf'))

def analyze_pd_differences(pd1, pd2):
    """Analyze differences between two persistence diagrams."""
    h0_1 = len(pd1[0]) if len(pd1) > 0 and pd1[0] else 0
    h0_2 = len(pd2[0]) if len(pd2) > 0 and pd2[0] else 0
    h1_1 = len(pd1[1]) if len(pd1) > 1 and pd1[1] else 0
    h1_2 = len(pd2[1]) if len(pd2) > 1 and pd2[1] else 0
    
    h0_finite_1 = count_finite_features(pd1[0]) if len(pd1) > 0 else 0
    h0_finite_2 = count_finite_features(pd2[0]) if len(pd2) > 0 else 0
    h1_finite_1 = count_finite_features(pd1[1]) if len(pd1) > 1 else 0
    h1_finite_2 = count_finite_features(pd2[1]) if len(pd2) > 1 else 0
    
    return {
        'h0_total_diff': abs(h0_1 - h0_2),
        'h1_total_diff': abs(h1_1 - h1_2),
        'h0_finite_diff': abs(h0_finite_1 - h0_finite_2),
        'h1_finite_diff': abs(h1_finite_1 - h1_finite_2),
        'summary': f"H0: {h0_1}→{h0_2} (finite: {h0_finite_1}→{h0_finite_2}), H1: {h1_1}→{h1_2} (finite: {h1_finite_1}→{h1_finite_2})"
    }

def feature_based_distance(pd1, pd2):
    """Compute a simple feature-count-based distance."""
    diff = analyze_pd_differences(pd1, pd2)
    return diff['h0_total_diff'] + 2 * diff['h1_total_diff']

def detect_topological_equivalence(iteration_pds, novelty_threshold=0.1):
    """Detect topological equivalence between executions."""
    n_iterations = len(iteration_pds)
    if n_iterations < 2:
        return {"message": "Need at least 2 iterations for equivalence analysis"}
    
    equivalence_analysis = {
        'equivalent_pairs': [],
        'novel_but_equivalent': [],
        'unique_topology_changes': [],
        'distance_matrix': np.zeros((n_iterations, n_iterations)),
        'feature_distance_matrix': np.zeros((n_iterations, n_iterations)),
        'detailed_comparisons': []
    }
    
    # Track unique pairs to avoid duplicates
    seen_pairs = set()
    
    # Compute all pairwise distances
    for i in range(n_iterations):
        for j in range(n_iterations):
            if i != j:
                pd_i = iteration_pds[i]['diagrams']
                pd_j = iteration_pds[j]['diagrams']
                
                try:
                    h1_i = pd_i[1] if len(pd_i) > 1 and pd_i[1] else None
                    h1_j = pd_j[1] if len(pd_j) > 1 and pd_j[1] else None
                    bottleneck_dist = safe_pd_distance(h1_i, h1_j, 'bottleneck')
                except:
                    bottleneck_dist = float('inf')
                
                feature_dist = feature_based_distance(pd_i, pd_j)
                
                equivalence_analysis['distance_matrix'][i, j] = bottleneck_dist
                equivalence_analysis['feature_distance_matrix'][i, j] = feature_dist
                
                if i < j:  # Only process each pair once
                    pair_key = tuple(sorted([iteration_pds[i]['input_name'], iteration_pds[j]['input_name']]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        diff_analysis = analyze_pd_differences(pd_i, pd_j)
                        equivalence_analysis['detailed_comparisons'].append({
                            'input1': iteration_pds[i]['input_name'],
                            'input2': iteration_pds[j]['input_name'],
                            'iteration1': i,
                            'iteration2': j,
                            'bottleneck_distance': bottleneck_dist,
                            'feature_distance': feature_dist,
                            'differences': diff_analysis
                        })
                        
                        if feature_dist <= novelty_threshold:
                            equivalence_analysis['equivalent_pairs'].append({
                                'input1': iteration_pds[i]['input_name'],
                                'input2': iteration_pds[j]['input_name'],
                                'iteration1': i,
                                'iteration2': j,
                                'feature_distance': feature_dist,
                                'differences': diff_analysis['summary']
                            })
    
    # Analyze novelty vs equivalence
    baseline_pd = iteration_pds[0]['diagrams']
    seen_novel = set()
    
    for i in range(1, n_iterations):
        current_pd = iteration_pds[i]['diagrams']
        current_input = iteration_pds[i]['input_name']
        
        baseline_distance = feature_based_distance(baseline_pd, current_pd)
        baseline_diff = analyze_pd_differences(baseline_pd, current_pd)
        
        equivalent_to_previous = []
        for j in range(1, i):
            feature_dist = equivalence_analysis['feature_distance_matrix'][i, j]
            if feature_dist <= novelty_threshold:
                equivalent_to_previous.append({
                    'iteration': j,
                    'input_name': iteration_pds[j]['input_name'],
                    'distance': feature_dist
                })
        
        if baseline_distance > novelty_threshold:
            if equivalent_to_previous and current_input not in seen_novel:
                seen_novel.add(current_input)
                equivalence_analysis['novel_but_equivalent'].append({
                    'current_input': current_input,
                    'current_iteration': i,
                    'baseline_distance': baseline_distance,
                    'baseline_changes': baseline_diff['summary'],
                    'equivalent_to': equivalent_to_previous
                })
            elif not equivalent_to_previous and current_input not in seen_novel:
                seen_novel.add(current_input)
                equivalence_analysis['unique_topology_changes'].append({
                    'input': current_input,
                    'iteration': i,
                    'baseline_distance': baseline_distance,
                    'baseline_changes': baseline_diff['summary'],
                    'h0_change': baseline_diff['h0_total_diff'],
                    'h1_change': baseline_diff['h1_total_diff']
                })
    
    return equivalence_analysis

def visualize_analysis(equivalence_analysis, iteration_pds):
    """Visualize the topological analysis results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Feature distance matrix heatmap
    feature_matrix = equivalence_analysis['feature_distance_matrix']
    im = ax1.imshow(feature_matrix, cmap='viridis', vmin=0, vmax=np.max(feature_matrix))
    ax1.set_title('Feature Count Distance Matrix\n(H0 diff + 2×H1 diff)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Iteration')
    
    input_names = [pd['input_name'].replace('.txt', '') for pd in iteration_pds]
    ax1.set_xticks(range(len(input_names)))
    ax1.set_yticks(range(len(input_names)))
    ax1.set_xticklabels(input_names, rotation=45)
    ax1.set_yticklabels(input_names)
    
    n = len(iteration_pds)
    for i in range(n):
        for j in range(n):
            color = "white" if feature_matrix[i, j] > np.max(feature_matrix)/2 else "black"
            ax1.text(j, i, f'{feature_matrix[i, j]:.0f}',
                    ha="center", va="center", color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax1)
    
    # 2. H1 feature count progression
    h1_counts = []
    for pd_data in iteration_pds:
        h1_count = len(pd_data['diagrams'][1]) if len(pd_data['diagrams']) > 1 and pd_data['diagrams'][1] else 0
        h1_counts.append(h1_count)
    
    ax2.plot(range(len(h1_counts)), h1_counts, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of H1 Features')
    ax2.set_title('H1 Feature Count Progression')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(input_names)))
    ax2.set_xticklabels(input_names, rotation=45)
    
    for i, count in enumerate(h1_counts):
        ax2.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # 3. Equivalent pairs network
    real_equivalents = [pair for pair in equivalence_analysis['equivalent_pairs'] 
                       if pair['input1'] != pair['input2']]
    
    if real_equivalents:
        G = nx.Graph()
        for pair in real_equivalents:
            G.add_edge(pair['input1'].replace('.txt', ''), 
                      pair['input2'].replace('.txt', ''), 
                      weight=pair['feature_distance'])
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax3, with_labels=True, node_color='lightblue', 
                node_size=1000, font_size=8)
        
        edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax3, font_size=8)
        
        ax3.set_title('Topologically Equivalent Inputs\n(Connected = feature distance ≤ threshold)')
    else:
        ax3.text(0.5, 0.5, 'No equivalent pairs found\n(all inputs have unique topology)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Topologically Equivalent Inputs')
    
    # 4. Summary statistics
    ax4.axis('off')
    summary_text = f"""Topological Evolution Summary:

Total iterations: {len(iteration_pds)}
H1 features: {' → '.join(map(str, h1_counts))}

Equivalent pairs: {len(real_equivalents)}
Novel but equivalent: {len(equivalence_analysis['novel_but_equivalent'])}
Unique topology changes: {len(equivalence_analysis['unique_topology_changes'])}

Detailed Progression:"""
    
    for i, pd_data in enumerate(iteration_pds):
        input_name = pd_data['input_name'].replace('.txt', '')
        h1_count = h1_counts[i]
        h1_change = f" (+{h1_count})" if i > 0 and h1_count > h1_counts[i-1] else ""
        summary_text += f"\n{i+1}. {input_name}: {h1_count} H1{h1_change}"
    
    if equivalence_analysis['novel_but_equivalent']:
        summary_text += "\n\nNovel but Equivalent:"
        for item in equivalence_analysis['novel_but_equivalent']:
            equiv_names = [eq['input_name'].replace('.txt', '') for eq in item['equivalent_to']]
            summary_text += f"\n• {item['current_input'].replace('.txt', '')} ≡ {equiv_names}"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('figures/topological_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def cluster_programs(equivalence_analysis, iteration_pds, threshold=0.5):
    """Cluster programs based on their topological features."""
    feature_matrix = equivalence_analysis['feature_distance_matrix']
    condensed = squareform(feature_matrix)
    Z = linkage(condensed, method='ward')
    
    plt.figure(figsize=(12, 6))
    dend = dendrogram(Z, 
                      labels=[pd['input_name'].replace('.txt', '') for pd in iteration_pds],
                      leaf_rotation=45,
                      leaf_font_size=10,
                      color_threshold=threshold)
    
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.legend()
    
    plt.title('Hierarchical Clustering of Programs\nBased on Topological Features')
    plt.xlabel('Programs')
    plt.ylabel('Distance')
    plt.tight_layout()
    
    plt.savefig('figures/program_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')
    print('\nCluster assignments (threshold =', threshold, "):")
    for label, pd in zip(cluster_labels, iteration_pds):
        print(f"  {pd['input_name'].replace('.txt', '')}: Cluster {label}")
    
    return {
        'linkage_matrix': Z,
        'dendrogram': dend,
        'feature_matrix': feature_matrix,
        'cluster_labels': cluster_labels
    }

def analyze_program(program_path, input_files, novelty_threshold=0.5, generate_figures=True):
    """Analyze a program's execution using topological data analysis.
    
    Args:
        program_path (str): Path to the program to analyze
        input_files (list): List of input files to use for analysis
        novelty_threshold (float): Threshold for detecting topological equivalence
        generate_figures (bool): Whether to generate and save figures
    
    Returns:
        tuple: (analyzer, equivalence_analysis, clustering_results)
    """
    print("=== Starting Topological Analysis ===")
    
    if not os.path.exists(program_path):
        raise FileNotFoundError(f"Program not found: {program_path}")
    
    analyzer = HybridTDAAnalyzer()
    afl_utils = AFL_Utils(program_path)
    
    # Process each input and store PDs
    for i, input_file in enumerate(input_files, 1):
        input_path = os.path.join('test_inputs', input_file)
        if os.path.exists(input_path):
            print(f"\nProcessing iteration {i}: {input_file}")
            edges, nodes = afl_utils.run_showmap(input_path)
            if edges:
                analyzer.analyze_execution_hybrid(edges, input_file)
                h0_count = len(analyzer.iteration_pds[-1]['diagrams'][0]) if analyzer.iteration_pds[-1]['diagrams'][0] else 0
                h1_count = len(analyzer.iteration_pds[-1]['diagrams'][1]) if len(analyzer.iteration_pds[-1]['diagrams']) > 1 and analyzer.iteration_pds[-1]['diagrams'][1] else 0
                print(f"Stored PDs: H0={h0_count}, H1={h1_count}")
            else:
                print(f"Warning: No edges found in {input_file}")
    
    # Deduplicate iteration_pds by input_name
    unique_pds = []
    seen_names = set()
    for pd in analyzer.iteration_pds:
        if pd['input_name'] not in seen_names:
            unique_pds.append(pd)
            seen_names.add(pd['input_name'])
    analyzer.iteration_pds = unique_pds
    
    # Analyze topological equivalence
    print("\n=== Topological Equivalence Analysis ===")
    equivalence_analysis = detect_topological_equivalence(analyzer.iteration_pds, novelty_threshold=novelty_threshold)
    
    # Print detailed results
    print("\nDetailed Pairwise Comparisons:")
    for comp in equivalence_analysis['detailed_comparisons']:
        print(f"  {comp['input1']} vs {comp['input2']}:")
        print(f"    Feature distance: {comp['feature_distance']}")
        print(f"    Changes: {comp['differences']['summary']}")
    
    print(f"\nEquivalent Pairs (feature distance ≤ {novelty_threshold}):")
    real_pairs = [p for p in equivalence_analysis['equivalent_pairs'] if p['input1'] != p['input2']]
    if real_pairs:
        for pair in real_pairs:
            print(f"  {pair['input1']} ≡ {pair['input2']} (distance: {pair['feature_distance']})")
            print(f"    {pair['differences']}")
    else:
        print("  No equivalent pairs found - all inputs have unique topological signatures")
    
    print("\nNovel but Equivalent Cases:")
    if equivalence_analysis['novel_but_equivalent']:
        for item in equivalence_analysis['novel_but_equivalent']:
            equiv_names = [eq['input_name'] for eq in item['equivalent_to']]
            print(f"  {item['current_input']} is novel vs baseline (distance: {item['baseline_distance']})")
            print(f"    Changes from baseline: {item['baseline_changes']}")
            print(f"    But equivalent to: {equiv_names}")
    else:
        print("  No novel but equivalent cases found")
    
    print("\nUnique Topology Changes:")
    for item in equivalence_analysis['unique_topology_changes']:
        print(f"  {item['input']}: distance {item['baseline_distance']} from baseline")
        print(f"    Changes: {item['baseline_changes']}")
        print(f"    H0 change: {item['h0_change']}, H1 change: {item['h1_change']}")
    
    # Generate figures if requested
    if generate_figures:
        print("\nGenerating figures...")
        os.makedirs('figures', exist_ok=True)
        visualize_analysis(equivalence_analysis, analyzer.iteration_pds)
        
        print("\n=== Program Clustering Analysis ===")
        clustering_results = cluster_programs(equivalence_analysis, analyzer.iteration_pds, threshold=novelty_threshold)
    else:
        clustering_results = None
    
    return analyzer, equivalence_analysis, clustering_results 