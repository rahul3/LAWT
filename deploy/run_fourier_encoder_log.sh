#!/bin/bash
#SBATCH --account=def-sbrugiap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=2-00:00
#SBATCH --mail-user=rahul.padmanabhan@mail.concordia.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/rahul3/scratch/2026/slurm_output_fourier_encoder_log/%x_%j.out
#SBATCH --error=/home/rahul3/scratch/2026/slurm_output_fourier_encoder_log/%x_%j.err

# Run the bashrc first
. ~/.bashrc
export FOURIER_ENCODER_TRAIN="/home/rahul3/projects/def-sbrugiap/rahul3/icprai_2026/LAWT/src/neuralnet/transformer_wo_embedding/train_enc_fourier_log.py"
export FOURIER_ENCODER_DUMP_PATH="/home/rahul3/scratch/experiments_fourier_encoder_log"
# export PYTHON_REQUIREMENTS="/home/rahul3/projects/def-sbrugiap/rahul3/icprai_2026/LAWT/requirements.txt"

echo "FOURIER_ENCODER_DUMP_PATH: $FOURIER_ENCODER_DUMP_PATH"
# echo "PYTHON_REQUIREMENTS: $PYTHON_REQUIREMENTS"

printf '%*s\n' 25 '' | tr ' ' '*'
echo "Using the ${CLUSTER} cluster"
printf '%*s\n' 25 '' | tr ' ' '*'

module load python/3.10
module load scipy-stack
export SLURM_TMPDIR="/home/rahul3/projects/def-sbrugiap/rahul3/icprai_2026/slurm_tmpdir"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"

# virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip

# echo "Starting to install requirements...."
# pip install --no-index -r $PYTHON_REQUIREMENTS

# pip install --no-index torch torchvision torchaudio
# pip install --no-index 'numpy<2.0'


# Experiment details
# EXP_NAME="shallownetwork"
# EXP_ID="$(date +"%Y%m%d%H%M")"
# OPERATION="log"
# MATRIX_TYPE="general"
# MAX_EPOCHS=100

# echo "Experiment name: $EXP_NAME"
# echo "Experiment ID: $EXP_ID"
# echo "Max epochs: $MAX_EPOCHS"
# echo "Operation: $OPERATION"

echo "Displaying the result of 'which python' command:"
which python

echo "Running with srun"
echo "Python version:"
python --version
PYTHON_EXEC=$(which python)
echo "Python path: $PYTHON_EXEC"


# EXP_ID="$MATRIX_TYPE_${OPERATION}_${ENCODING_NAME}_dim_${MATRIX_DIM}_$(date +"%Y%m%d%H%M")"

srun $PYTHON_EXEC $FOURIER_ENCODER_TRAIN 
