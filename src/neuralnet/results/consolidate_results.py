#!/usr/bin/env python3
"""
Consolidate all experiment results from CSV files into a single dataframe.

This script searches for all *_results_*.csv files under the experiments directory
and consolidates them into a single CSV file.

Requirements:
    - pandas

Usage:
    # Basic usage (uses default paths):
    python consolidate_results.py
    
    # Specify custom base path:
    python consolidate_results.py --base_path /path/to/experiments/
    
    # Specify custom output file:
    python consolidate_results.py --output /path/to/output.csv
    
    # Run quietly:
    python consolidate_results.py --quiet
"""

import os
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


def find_result_csvs(base_path):
    """
    Recursively find all result CSV files in the directory structure.
    
    Args:
        base_path: Base directory to search for CSV files
        
    Returns:
        List of paths to CSV files
    """
    csv_files = []
    base_path = Path(base_path)
    
    # Find all CSV files that match the pattern *_results_*.csv
    for csv_file in base_path.rglob("*_results_*.csv"):
        csv_files.append(csv_file)
    
    return csv_files


def consolidate_results(base_path, output_path=None, verbose=True):
    """
    Consolidate all result CSVs into a single dataframe.
    
    Args:
        base_path: Base directory containing experiment results
        output_path: Path to save consolidated CSV (optional)
        verbose: Print progress information
        
    Returns:
        Consolidated pandas DataFrame
    """
    # Find all CSV files
    csv_files = find_result_csvs(base_path)
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("No CSV files found!")
        return None
    
    # Read and consolidate all CSVs
    dfs = []
    for i, csv_file in enumerate(csv_files, 1):
        if verbose:
            print(f"Reading {i}/{len(csv_files)}: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Add metadata from file path
            # Extract experiment ID from path (e.g., shallow_matrixnet_20260120073018)
            parts = csv_file.parts
            for part in parts:
                if part.startswith("shallow_matrixnet_"):
                    df['experiment_id'] = part
                    break
            
            # Add file path for reference
            df['source_file'] = str(csv_file)
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    if len(dfs) == 0:
        print("No valid CSV files could be read!")
        return None
    
    # Concatenate all dataframes
    consolidated_df = pd.concat(dfs, ignore_index=True)
    
    if verbose:
        print(f"\nConsolidated {len(dfs)} CSV files")
        print(f"Total rows: {len(consolidated_df)}")
        print(f"\nColumns: {list(consolidated_df.columns)}")
        print(f"\nOperations: {consolidated_df['operation'].unique()}")
        print(f"Dimensions: {sorted(consolidated_df['dim'].unique())}")
        print(f"Training samples: {sorted(consolidated_df['train_sample'].unique())}")
        print(f"Tolerances: {sorted(consolidated_df['tolerance'].unique())}")
    
    # Save if output path provided
    if output_path:
        consolidated_df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nSaved consolidated results to: {output_path}")
    
    return consolidated_df


def main():
    """Main function to run the consolidation script."""
    parser = argparse.ArgumentParser(
        description="Consolidate experiment results from multiple CSV files"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/home/rahul3/scratch/2026/experiments/shallownetwork/",
        help="Base directory containing experiment results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for consolidated CSV (default: auto-generate with timestamp)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent
        args.output = output_dir / f"consolidated_results_{timestamp}.csv"
    
    # Run consolidation
    df = consolidate_results(
        base_path=args.base_path,
        output_path=args.output,
        verbose=not args.quiet
    )
    
    if df is not None:
        print(f"\n✓ Successfully consolidated {len(df)} rows")
        print(f"✓ Output saved to: {args.output}")
    else:
        print("\n✗ Consolidation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
