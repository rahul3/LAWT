#!/bin/bash

num_epochs=$1
validation_interval=$2

# Configurable parameters
distributions=("gaussian" "uniform")
matrix_types=(
    "wigner"
    "orthogonal"
    "toeplitz"
    "hankel"
    "stochastic"
    "circulant"
    "band"
    "positive_definite"
    "m_matrix"
    "p_matrix"
    "z_matrix"
    "h_matrix"
    "hadamard"
    "general"
)

operations=("square" "exponential" "sign")
dimensions=(2 3 4 5)

# Loop over distributions, matrix types, and dimensions
for operation in "${operations[@]}"; do
    for matrix_type in "${matrix_types[@]}"; do
        for dim in "${dimensions[@]}"; do
            echo "Running train.py with operation=$operation, matrix_type=$matrix_type, dim=$dim"
            # python train.py --operation "$operation" --matrix_type "$matrix_type" --dim "$dim" --num_epochs "$num_epochs" --validation_interval "$validation_interval"
            python train.py --operation "$operation" --matrix_type "$matrix_type" --dim "$dim"
        done
    done
done