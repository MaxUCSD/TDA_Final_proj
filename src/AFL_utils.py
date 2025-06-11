import subprocess
import os
import tempfile

class AFL_Utils:
    """Utilities for working with AFL (American Fuzzy Lop) execution traces.
    
    This class provides methods for running AFL's showmap tool and parsing its output
    to extract execution traces from programs. It handles the interaction with AFL
    and provides convenient access to trace data.
    
    Attributes:
        target_program (str): Path to the program being analyzed
        trace_text (str): Raw output from AFL showmap
        trace_edges (list): List of edge IDs from the trace
        trace_nodes (list): List of unique node IDs from the trace
    """
    
    def __init__(self, target_program: str):
        """Initializes a new AFL_Utils instance.
        
        Args:
            target_program (str): Path to the program to analyze with AFL
        """
        self.target_program = target_program
        self.trace_text = None
        self.trace_edges = None
        self.trace_nodes = None
        
    def __str__(self):
        """Returns a string representation of the AFL_Utils instance.
        
        Returns:
            str: Formatted string showing program path and trace information
        """
        status = []
        status.append(f"AFL_Utils for program: {self.target_program}")
        if self.trace_edges is not None:
            status.append(f"Last trace edges: {self.trace_edges}")
            status.append(f"Last trace nodes: {self.trace_nodes}")
            status.append(f"Trace length: {len(self.trace_edges)}")
        else:
            status.append("No trace data collected yet")
        return "\n".join(status)
        
    def run_showmap(self, input_file_path: str):
        """Runs AFL showmap on the target program with the given input.
        
        This method:
        1. Runs AFL showmap to collect edge coverage
        2. Parses the output to extract edge IDs
        3. Extracts unique node IDs from the edges
        
        Args:
            input_file_path (str): Path to the input file to use
            
        Returns:
            tuple: (list of edge IDs, list of node IDs)
        """
        self.trace_text = self._run_showmap(input_file_path)
        self.trace_edges = self._parse_afl_trace()
        self.trace_nodes = self._get_nodes_from_edges(self.trace_edges)
        print(f"{input_file_path} parsed edge IDs: {self.trace_edges}")
        return self.trace_edges, self.trace_nodes
    
    def _run_showmap(self, input_file_path: str):
        """Runs AFL showmap and captures its output.
        
        This method:
        1. Creates a temporary file for AFL output
        2. Runs AFL showmap with the given input
        3. Reads and returns the output
        
        Args:
            input_file_path (str): Path to the input file
            
        Returns:
            str: Raw output from AFL showmap, or None if execution failed
        """
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            output_file = tmp.name
        with open(input_file_path, 'r') as f:
            input_value = f.read().strip()
        cmd = ['afl-showmap','-o', output_file,'-C','--',self.target_program,input_value]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with open(output_file, 'r') as f:
                trace_text = f.read()
            os.remove(output_file)
            return trace_text
        except (subprocess.CalledProcessError, FileNotFoundError):
            if os.path.exists(output_file):
                os.remove(output_file)
            return None

    def _parse_afl_trace(self):
        """Parses the AFL trace text into a list of edge IDs.
        
        This method processes the raw AFL output format:
        - Each line contains an edge ID and count (e.g., "5:1")
        - Only the edge IDs are extracted
        
        Returns:
            list: A list of integer edge IDs from the trace, or empty list if no trace
        """
        if not self.trace_text: return []
        edge_ids = []
        for line in self.trace_text.strip().split('\n'):
            if ':' not in line: continue
            edge_id, count = line.strip().split(':')
            edge_ids.append(int(edge_id))
        return edge_ids

    def _get_nodes_from_edges(self, edges):
        """Extracts unique node IDs from a list of edges.
        
        This method:
        1. Takes a list of edge IDs
        2. Extracts all unique node IDs
        3. Returns them in sorted order
        
        Args:
            edges (list): List of edge IDs from the trace
            
        Returns:
            list: Sorted list of unique node IDs present in the edges
        """
        if not edges: return []
        return sorted(set(edges))