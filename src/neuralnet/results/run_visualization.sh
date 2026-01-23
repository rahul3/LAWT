#!/bin/bash
#
# Convenient script to run the visualization
# Usage: ./run_visualization.sh

# Python interpreter with required packages
PYTHON_EXEC="/project/6042579/rahul3/icprai_2026/slurm_tmpdir/env/bin/python"

# Default paths
CSV_PATH="/home/rahul3/projects/def-sbrugiap/rahul3/icprai_2026/results/shallow_nn_results.csv"
OUTPUT_DIR="./plots"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================"
echo "Shallow NN Results Visualization"
echo "========================================"
echo ""
echo "CSV Path: $CSV_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Check if CSV exists
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found at $CSV_PATH"
    echo "Please ensure the consolidated results CSV exists."
    exit 1
fi

# Run the visualization script
echo "Generating visualizations..."
$PYTHON_EXEC "$SCRIPT_DIR/visualize_results.py" \
    --csv_path "$CSV_PATH" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Visualization complete!"
echo "Check the plots in: $OUTPUT_DIR"
echo "========================================"
