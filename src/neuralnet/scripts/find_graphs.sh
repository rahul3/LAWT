#!/bin/bash

# Check if a directory path is provided
if [ $# -eq 0 ]; then
    echo "Please provide a directory path."
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Set the search directory
search_dir="$1"

# Set the output file
output_file="graph_file_list.txt"

# Check if the directory exists
if [ ! -d "$search_dir" ]; then
    echo "Error: Directory '$search_dir' does not exist."
    exit 1
fi

# Find all .log files and save their full paths to the output file
find "$search_dir" -type f -name "trai*.png" > "$output_file"

echo "Full paths of .log files have been saved to $output_file"
