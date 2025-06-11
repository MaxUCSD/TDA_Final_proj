# Topological Data Analysis for Program Traces

This project analyzes program execution traces using Topological Data Analysis (TDA), focusing on global zigzag persistence of the control flow graph (CFG) to cluster and compare program behaviors.

## Project Structure

```
tda_project/
├── src/
│   ├── analysis.py             # Core analysis classes (TDAnalyzer, GlobalCFGAnalyzer, HybridTDAAnalyzer)
│   ├── topological_analysis.py # High-level analysis pipeline and clustering
│   ├── visualization.py        # Visualization functions for figures and plots
│   ├── main.py                 # Main script for running the analysis
│   ├── AFL_utils.py            # Utilities for AFL trace collection
│   └── graph_utils.py          # Graph manipulation utilities
├── figures/                    # Generated visualization outputs
├── test_programs/              # Test programs for analysis
├── test_inputs/                # Test input files
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```
## Requirements

- Python 3.8+
- NetworkX
- NumPy
- Dionysus
- Matplotlib

Install dependencies with:
```bash
pip install -r requirements.txt
```

## AFL++ Setup

This project uses AFL++ for collecting program execution traces. Follow these steps to set up AFL++:

1. Install AFL++:
   ```bash
   # For macOS (using Homebrew)
   brew install aflplusplus

   # For Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y build-essential python3-dev automake git flex bison libglib2.0-dev libpixman-1-dev python3-setuptools
   git clone https://github.com/AFLplusplus/AFLplusplus.git
   cd AFLplusplus
   make distrib
   sudo make install
   ```

2. Verify installation:
   ```bash
   afl-showmap --version
   ```

3. Run AFL++ with existing programs:
   ```bash
   # Run AFL++ showmap on an existing program
   afl-showmap -o output.map -- .test_programs/test_program < tda_project/test_inputs/pos5.txt
   ```
