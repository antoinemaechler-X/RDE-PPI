#!/usr/bin/env python3
"""
Script to run RDE-Linear pipeline with grouped train/test split.
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    else:
        print("FAILED")
        print("Error:")
        print(result.stderr)
        sys.exit(1)

def main():
    # Configuration
    result_file = "path/to/your/training/results.pkl"  # You need to specify this
    grouped_csv_path = "./data/complex_sequences_grouped_60.csv"
    output_dir = "./RDE_linear_skempi_grouped"
    
    print("RDE-Linear Grouped Pipeline")
    print("="*60)
    print(f"Result file: {result_file}")
    print(f"Grouped CSV: {grouped_csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Run entropy calculation (if needed)
    # This step is optional and depends on your setup
    # run_command(
    #     "python -m rde.linear.entropy",
    #     "Entropy calculation"
    # )
    
    # Step 2: Run calibration with grouped split
    run_command(
        f"python -m rde.linear.calibrate_grouped {result_file} --grouped_csv_path {grouped_csv_path} --num_folds 10 --output_dir {output_dir}",
        "Calibration with grouped train/test split"
    )
    
    # Step 3: Export parameters
    run_command(
        f"python -m rde.linear.export_params --result_dir {output_dir} --output data/rdelinear_params_grouped.csv",
        "Export parameters"
    )
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 