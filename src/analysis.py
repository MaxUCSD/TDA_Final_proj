import networkx as nx
import numpy as np
import dionysus as d
import matplotlib.pyplot as plt
from graph_utils import trace_to_graph, get_graph_stats
from AFL_utils import AFL_Utils
import os

class TDAnalyzer:
    """Analyzes program execution traces using Topological Data Analysis (TDA).
    
    This class implements zigzag persistence homology to analyze the topological
    structure of program execution traces. It computes persistent homology features
    (H0 and H1) to identify connected components and cycles in the execution flow.
    
    Attributes:
        G (nx.DiGraph): The graph representation of the trace
        root (int): The root node of the trace
        distance_from_root (dict): Mapping of nodes to their distance from root
        filtration (d.Filtration): The simplicial filtration for persistence
        zigzag (d.ZigzagPersistence): The zigzag persistence computation
        diagrams (list): The persistence diagrams (H0, H1)
        cells (list): The persistence cells
    """

    def __init__(self):
        """Initializes a new TDAnalyzer instance with empty attributes."""
        self.G = None
        self.root = None
        self.distance_from_root = None
        self.filtration = None
        self.zigzag = None
        self.diagrams = None
        self.cells = None

    def analyze_trace(self, trace):
        """Analyzes a given trace using TDA.
        
        This method:
        1. Converts the trace to a graph
        2. Computes distances from root
        3. Builds a simplicial filtration
        4. Computes zigzag persistent homology
        
        Args:
            trace (list): List of integers representing the execution trace
        """
        self.G = trace_to_graph(trace)
        self.root = trace[0]
        self.distance_from_root = nx.shortest_path_length(self.G, source=self.root)

        # Print distances from root
        print("\n--- Node Distances from Root ---")
        for node, dist in sorted(self.distance_from_root.items()):
            print(f"  Node {node}: distance = {dist}")

        # Build and check filtration
        self._build_filtration()
        print("\nChecking if filtration is a valid simplicial complex...")
        d.is_simplicial(self.filtration, report=True)
        print("Check complete.")

        # Compute birth times and zigzag persistence
        times = self._compute_birth_times()
        print("\nComputing zigzag persistent homology...")
        self.zigzag, self.diagrams, self.cells = d.zigzag_homology_persistence(self.filtration, times)

        # Print H0 diagram points
        print("\n--- H0 Diagram Points (Birth, Death) ---")
        if len(self.diagrams) > 0 and self.diagrams[0]:
            h0_diagram = self.diagrams[0]
            for pt in h0_diagram:
                print(f"  ({pt.birth}, {pt.death})")
        else:
            print("  No H0 features found in the trace")

        # Print H1 diagram points
        print("\n--- H1 Diagram Points (Birth, Death) ---")
        if len(self.diagrams) > 1 and self.diagrams[1]:
            h1_diagram = self.diagrams[1]
            for pt in h1_diagram:
                print(f"  ({pt.birth}, {pt.death})")
        else:
            print("  No H1 features found in the trace (no cycles detected)")

    def analyze_afl_trace(self, afl_utils: AFL_Utils, input_file_path: str):
        """Analyzes a trace from AFL showmap output.
        
        Args:
            afl_utils (AFL_Utils): AFL utilities instance
            input_file_path (str): Path to the input file to analyze
        """
        edges, nodes = afl_utils.run_showmap(input_file_path)
        if not edges:
            print("No trace data available")
            return
        
        # Convert AFL edge IDs to a trace
        trace = edges
        self.analyze_trace(trace)

    def _build_filtration(self):
        """Builds the simplicial filtration from the graph.
        
        Creates 0-simplices (nodes) and 1-simplices (edges) for the filtration.
        """
        all_simplices = []
        for node in self.G.nodes():
            all_simplices.append(d.Simplex([int(node)]))
        for u, v in self.G.edges():
            all_simplices.append(d.Simplex([int(u), int(v)]))
        self.filtration = d.Filtration(all_simplices)

    def _compute_birth_times(self):
        """Computes birth times for the filtration.
        
        Birth times are based on the distance from root for each simplex.
        
        Returns:
            list: List of birth times for each simplex in the filtration
        """
        times = []
        for s in self.filtration:
            if len(s) == 1:
                node = s[0]
                birth_time = self.distance_from_root[node]
            elif len(s) == 2:
                u, v = s[0], s[1]
                birth_time = max(self.distance_from_root[u], self.distance_from_root[v])
            times.append([birth_time])
        return times

    def get_graph_stats(self):
        """Returns statistics about the analyzed graph.
        
        Returns:
            dict: Dictionary containing graph statistics
            
        Raises:
            ValueError: If analyze_trace() or analyze_afl_trace() hasn't been called
        """
        if not self.G:
            raise ValueError("Must run analyze_trace() or analyze_afl_trace() before getting stats")
        return get_graph_stats(self.G)

class GlobalCFGAnalyzer:
    """Analyzes the global Control Flow Graph (CFG) from multiple execution traces.
    
    This class builds and analyzes a global CFG that combines information from
    multiple execution traces, identifying common paths, branch points, and
    convergence points.
    
    Attributes:
        global_cfg (nx.DiGraph): The global control flow graph
        all_traces (list): List of all execution traces
        trace_metadata (dict): Mapping of input names to their traces
    """
    
    def __init__(self):
        """Initializes a new GlobalCFGAnalyzer instance with empty attributes."""
        self.global_cfg = None
        self.all_traces = []
        self.trace_metadata = {}
        
    def add_trace(self, trace, input_name):
        """Adds a trace to the global CFG.
        
        Args:
            trace (list): The execution trace to add
            input_name (str): Name/identifier for this trace
        """
        self.all_traces.append(trace)
        self.trace_metadata[input_name] = trace
        
    def build_global_cfg(self):
        """Builds the global control flow graph from all traces.
        
        Creates a directed graph where edges are weighted by their frequency
        across all traces.
        
        Returns:
            nx.DiGraph: The constructed global CFG
        """
        import networkx as nx
        
        self.global_cfg = nx.DiGraph()
        
        # Add all edges from all traces
        for trace in self.all_traces:
            for i in range(len(trace) - 1):
                edge = (trace[i], trace[i+1])
                
                # Add edge and track frequency
                if self.global_cfg.has_edge(trace[i], trace[i+1]):
                    self.global_cfg[trace[i]][trace[i+1]]['frequency'] += 1
                else:
                    self.global_cfg.add_edge(trace[i], trace[i+1], frequency=1)
        
        return self.global_cfg
    
    def analyze_global_topology(self):
        """Analyzes the topology of the global CFG.
        
        Computes and prints:
        - Node and edge counts
        - Distances from root
        - Node degrees
        - Edge frequencies
        
        Returns:
            dict: Mapping of nodes to their distances from root
        """
        if not self.global_cfg:
            self.build_global_cfg()
            
        # Compute distances from root (first node of first trace)
        root = self.all_traces[0][0] if self.all_traces else None
        if not root:
            return None
            
        distances = nx.shortest_path_length(self.global_cfg, source=root)
        
        print("\n=== GLOBAL CFG ANALYSIS ===")
        print(f"Total nodes: {self.global_cfg.number_of_nodes()}")
        print(f"Total edges: {self.global_cfg.number_of_edges()}")
        print(f"Root node: {root}")
        
        print("\nNode distances from root:")
        for node, dist in sorted(distances.items()):
            out_degree = self.global_cfg.out_degree(node)
            in_degree = self.global_cfg.in_degree(node)
            print(f"  Node {node}: distance={dist}, in_degree={in_degree}, out_degree={out_degree}")
            
        print("\nEdge frequencies:")
        for u, v, data in self.global_cfg.edges(data=True):
            print(f"  {u} → {v}: frequency={data['frequency']}")
            
        return distances
    
    def analyze_global_persistence(self):
        """Computes zigzag persistent homology on the global CFG.
        
        Creates a filtration based on distance from root and computes
        zigzag persistent homology features (H0 and H1).
        
        Returns:
            list: Persistence diagrams for H0 and H1
        """
        import dionysus as d
        
        if not self.global_cfg:
            self.build_global_cfg()
            
        # Get root and distances
        root = self.all_traces[0][0]
        distances = nx.shortest_path_length(self.global_cfg, source=root)
        
        print("\n=== GLOBAL CFG ZIGZAG PERSISTENT HOMOLOGY ===")
        
        # Build filtration based on distance from root
        simplices = []
        
        # Add nodes (0-simplices)
        for node in self.global_cfg.nodes():
            simplex = d.Simplex([int(node)])
            simplices.append((distances[node], simplex))
        
        # Add edges (1-simplices)  
        for u, v in self.global_cfg.edges():
            simplex = d.Simplex([int(u), int(v)])
            birth_time = max(distances[u], distances[v])
            simplices.append((birth_time, simplex))
        
        # Sort by filtration time
        simplices.sort(key=lambda x: x[0])
        
        # Create filtration
        filtration = d.Filtration([s[1] for s in simplices])
        
        # Compute birth times for zigzag - each time needs to be a list
        times = [[s[0]] for s in simplices]
        
        # Compute zigzag persistence
        zigzag, diagrams, cells = d.zigzag_homology_persistence(filtration, times)
        
        # Print results
        print(f"H0 (Connected Components) - {len(diagrams[0])} features:")
        for i, pt in enumerate(diagrams[0]):
            print(f"  Feature {i}: birth={pt.birth}, death={pt.death}")
            
        if len(diagrams) > 1:
            print(f"H1 (Cycles) - {len(diagrams[1])} features:")
            for i, pt in enumerate(diagrams[1]):
                print(f"  Feature {i}: birth={pt.birth}, death={pt.death}")
        else:
            print("H1 (Cycles) - 0 features")
            
        return diagrams
    
    def analyze_execution_in_global_context(self, execution_trace, input_name):
        """Analyzes how a specific execution interacts with global topology.
        
        Computes:
        - Execution subgraph
        - Node distances in execution vs global context
        - Branch points and convergence points
        
        Args:
            execution_trace (list): The execution trace to analyze
            input_name (str): Name/identifier for this execution
            
        Returns:
            dict: Analysis results including subgraph, distances, and key points
        """
        if not self.global_cfg:
            self.build_global_cfg()
            
        print(f"\n=== EXECUTION ANALYSIS: {input_name} ===")
        print(f"Trace: {execution_trace}")
        
        # Extract subgraph for this execution
        execution_subgraph = self.global_cfg.subgraph(execution_trace)
        
        print(f"Execution subgraph: {execution_subgraph.number_of_nodes()} nodes, {execution_subgraph.number_of_edges()} edges")
        
        # Compute distances in execution subgraph
        root = execution_trace[0]
        exec_distances = nx.shortest_path_length(execution_subgraph, source=root)
        
        print("Node distances in execution:")
        for node in execution_trace:
            global_dist = nx.shortest_path_length(self.global_cfg, source=root)[node]
            exec_dist = exec_distances[node]
            print(f"  Node {node}: global_distance={global_dist}, execution_distance={exec_dist}")
        
        # Which branch points does this execution hit?
        branch_points = []
        for node in execution_trace:
            if self.global_cfg.out_degree(node) > 1:
                branch_points.append(node)
        print(f"Branch points traversed: {branch_points}")
        
        # Which convergence points does this execution hit?
        convergence_points = []
        for node in execution_trace:
            if self.global_cfg.in_degree(node) > 1:
                convergence_points.append(node)
        print(f"Convergence points traversed: {convergence_points}")
        
        return {
            'subgraph': execution_subgraph,
            'distances': exec_distances,
            'branch_points': branch_points,
            'convergence_points': convergence_points
        }

class HybridTDAAnalyzer:
    """Analyzes program execution using topological data analysis.
    
    This class implements:
    1. Individual trace analysis using persistence diagrams
    2. Global CFG analysis
    3. Topological equivalence analysis
    
    Attributes:
        individual_analyzer (TDAnalyzer): For individual trace analysis
        global_analyzer (GlobalCFGAnalyzer): For global CFG analysis
        iteration_pds (list): List of persistence diagrams at each iteration
    """
    
    def __init__(self):
        """Initializes a new HybridTDAAnalyzer instance."""
        self.individual_analyzer = TDAnalyzer()
        self.global_analyzer = GlobalCFGAnalyzer()
        self.iteration_pds = []  # Track PDs at each iteration
        
    def analyze_execution_hybrid(self, trace, input_name):
        """Performs analysis on a single execution trace.
        
        Args:
            trace (list): The execution trace to analyze
            input_name (str): Name/identifier for this execution
            
        Returns:
            dict: Global persistence diagrams
        """
        print(f"\n{'='*50}")
        print(f"ANALYSIS: {input_name}")
        print(f"{'='*50}")
        
        # 1. Individual trace analysis
        print("\n--- INDIVIDUAL TRACE ANALYSIS ---")
        self.individual_analyzer.analyze_trace(trace)
        
        # 2. Add to global CFG and analyze
        self.global_analyzer.add_trace(trace, input_name)
        self.global_analyzer.build_global_cfg()
        self.global_analyzer.analyze_global_topology()
        global_diagrams = self.global_analyzer.analyze_global_persistence()
        
        # Store PDs for this iteration
        self.iteration_pds.append({
            'input_name': input_name,
            'diagrams': global_diagrams
        })
        
        return global_diagrams 