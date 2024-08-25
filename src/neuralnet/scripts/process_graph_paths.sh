#!/bin/bash

# Check if input and output file names are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

input_file="$1"
output_file="$2"

# Create CSV header
echo "operation_type,matrix_type,dimension,path" > "$output_file"

# Process each line in the input file
while IFS= read -r line; do
    # Extract operation type
    operation_type=$(echo "$line" | sed -n 's/.*\/experiments\/\([^/]*\)\/.*/\1/p')

    # Extract matrix type
    matrix_type=$(echo "$line" | sed -n 's/.*\/experiments\/[^/]*\/\([^/]*\)\/.*/\1/p')

    # Extract dimension
    dimension=$(echo "$line" | sed -n 's/.*\/dim_\([0-9]*\)\/.*/\1/p')

    # Construct CSV line
    echo "$operation_type,$matrix_type,$dimension,$line" >> "$output_file"
done < "$input_file"

echo "CSV file has been created: $output_file"