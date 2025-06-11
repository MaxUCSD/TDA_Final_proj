#!/usr/bin/env python3

import os
from topological_analysis import analyze_program

def setup_directories():
    """Ensures all necessary directories exist for the analysis.
    
    Creates the following directories if they don't exist:
    - figures: For storing visualization outputs
    - test_programs: For storing programs to analyze
    - test_inputs: For storing test input files
    """
    dirs = ['figures', 'test_programs', 'test_inputs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

if __name__ == '__main__':
    # Ensure directories exist
    setup_directories()
    
    # Define test inputs for each program
    input_files_loop_program = [
        'zero_loop.txt',      # Baseline: no loops
        'one_loop.txt',       # Adds one loop iteration
        'three_loop.txt',     # Multiple loop iterations
        'four_loop.txt',      # Multiple loop iterations
        'five_loop.txt',      # Multiple loop iterations
        'seven_loop.txt',     # Even more iterations
        'neg_five_loop.txt'   # Different path
    ]
    
    input_files_test_program = [
        'zero.txt', 'pos5.txt', 'pos20.txt', 
        'neg5.txt', 'neg20.txt'
    ]
    
    # Run analysis for each program
    program_path_loop_program = os.path.join('test_programs', 'loop_program')
    program_path_test_program = os.path.join('test_programs', 'test_program')
    
    try:
        print("\n=== Analyzing Loop Program ===")
        loop_results = analyze_program(program_path_loop_program, input_files_loop_program)
        
        print("\n=== Analyzing Test Program ===")
        test_results = analyze_program(program_path_test_program, input_files_test_program)
        
        print("\nAnalysis complete!")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please make sure the test program exists and is compiled.")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check that the test inputs contain valid traces.")
    except Exception as e:
        print(f"\nUnexpected error: {e}") 