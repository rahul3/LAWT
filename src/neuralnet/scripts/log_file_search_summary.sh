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
output_file="log_files_content_summary.txt"

# Check if the directory exists
if [ ! -d "$search_dir" ]; then
    echo "Error: Directory '$search_dir' does not exist."
    exit 1
fi

# Clear the output file if it exists
> "$output_file"

# Find all .log files and process each one
find "$search_dir" -type f -name "*.log" | while read -r log_file; do
    echo "Processing: $log_file" >> "$output_file"
    echo "--------------------" >> "$output_file"
    
    # Extract line 3
    echo "Line 3:" >> "$output_file"
    sed -n '3p' "$log_file" >> "$output_file"
    echo "" >> "$output_file"
    
    # Extract the last line
    echo "Last line:" >> "$output_file"
    tail -n 1 "$log_file" >> "$output_file"
    
    echo "" >> "$output_file"
    echo "" >> "$output_file"
done

echo "Extracted content from log files has been saved to $output_file"