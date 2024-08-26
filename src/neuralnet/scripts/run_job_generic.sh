#!/bin/bash
#SBATCH --account=def-sbrugiap
#SBATCH --gres=gpu:1
# #SBATCH --gres=gpu:v100:1
# #SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --mail-user=rahul.padmanabhan@mail.concordia.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/rahul3/temp/slurm_output/%x_%j.out
#SBATCH --error=/home/rahul3/temp/slurm_output/%x_%j.err

# Run the bashrc first
. ~/.bashrc

export LAWT_PATH="/home/rahulpadmanabhan/projects/ws1/LAWT"
export LAWT_TRAIN="${LAWT_PATH}/train.py"

if [ "$(hostname)" = "node1" ]; then
    printf '%*s\n' 25 '' | tr ' ' '*'
    echo "Using local nodel1"
    printf '%*s\n' 25 '' | tr ' ' '*'
    echo "Setting up node1"
    export LAWT_PATH="/home/rahulpadmanabhan/projects/ws1/LAWT"
    export LAWT_TRAIN="${LAWT_PATH}/train.py"
    export LAWT_DUMP_PATH="/home/rahulpadmanabhan/projects/ws1/experiments"
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/home/rahulpadmanabhan/projects/software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/home/rahulpadmanabhan/projects/software/miniconda3/etc/profile.d/conda.sh" ]; then
            . "/home/rahulpadmanabhan/projects/software/miniconda3/etc/profile.d/conda.sh"
        else
            export PATH="/home/rahulpadmanabhan/projects/software/miniconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<

    conda activate dlenv_310
elif [ "$(hostname)" = "node2" ]; then
    printf '%*s\n' 25 '' | tr ' ' '*'
    echo "Using local node2"
    printf '%*s\n' 25 '' | tr ' ' '*'
    export LAWT_DUMP_PATH="/home/rahulpadmanabhan/Development/ws1/masters_thesis_2/experiments"
elif [ "${CLUSTER:-}" = "beluga" ] || [ "${CLUSTER:-}" = "graham" ]; then
    printf '%*s\n' 25 '' | tr ' ' '*'
    echo "Using the ${CLUSTER} cluster"
    printf '%*s\n' 25 '' | tr ' ' '*'
    export LAWT_PATH="/home/rahul3/projects/def-sbrugiap/rahul3/LAWT"
    export LAWT_TRAIN="${LAWT_PATH}/train.py"
    export LAWT_DUMP_PATH="/home/rahul3/scratch/experiments"
    export PYTHON_REQUIREMENTS="/home/rahul3/projects/def-sbrugiap/rahul3/requirements.txt"

    module load python/3.10
    module load scipy-stack
    export SLURM_TMPDIR="/home/rahul3/projects/def-sbrugiap/rahul3/slurm_tmpdir"

    virtualenv --no-download $SLURM_TMPDIR/env
    source $SLURM_TMPDIR/env/bin/activate
    pip install --no-index --upgrade pip

    echo "Starting to install requirements...."
    pip install --no-index -r $PYTHON_REQUIREMENTS

else
    export LAWT_PATH="/home/rahul3/projects/def-sbrugiap/rahul3/LAWT"
    export LAWT_TRAIN="${LAWT_PATH}/train.py"
    export LAWT_DUMP_PATH="/home/rahul3/scratch/experiments"
fi

echo "LAWT_PATH: $LAWT_PATH"
echo "LAWT_TRAIN: $LAWT_TRAIN"
echo "LAWT_DUMP_PATH: $LAWT_DUMP_PATH"
echo "PYTHON_REQUIREMENTS: $PYTHON_REQUIREMENTS"

# Experiment details
EXP_NAME="matrix_logarithm"
EXP_ID="202408252325"
OPERATION="matrix_logarithm"
MAX_EPOCHS=100

echo "Experiment name: $EXP_NAME"
echo "Experiment ID: $EXP_ID"
echo "Max epochs: $MAX_EPOCHS"
echo "Operation: $OPERATION"

echo "Displaying the result of 'which python' command:"
which python

if [ "${CLUSTER:-}" = "beluga" ] || [ "${CLUSTER:-}" = "graham" ]; then
    echo "Python version:"
    python --version
    PYTHON_EXEC=$(which python)
    echo "Python path: $PYTHON_EXEC"

    srun $PYTHON_EXEC $LAWT_TRAIN --dump_path "${LAWT_DUMP_PATH}" --save_periodic 0 --fp16 true --amp 2 --accumulate_gradients 1 --clip_grad_norm 5 --enc_emb_dim 512 --dec_emb_dim 512 --n_enc_layers 8 --n_dec_layers 1 --n_enc_heads 8 --n_dec_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --optimizer 'adam_warmup,warmup_updates=10000,lr=0.0001' --batch_size 64 --batch_size_eval 128 --max_len 200 --epoch_size 300000 --max_epoch $MAX_EPOCHS --num_workers 10 --export_data false --reload_data '' --reload_size '-1' --batch_load false --env_name numeric --tasks numeric --env_base_seed '-1' --eval_size 10000 --min_dimension 5 --max_dimension 5 --max_input_coeff 5 --operation $OPERATION --generator gaussian --rectangular false --output_encoding 'floatsymbol,2' --input_encoding 'floatsymbol,2' --max_output_len 80 --float_tolerance '0.05' --more_tolerance '0.02,0.01,0.005' --eval_norm d1 --eval_verbose 0 --beam_eval 1 --stopping_criterion 'valid_numeric_beam_acc,60' --validation_metrics valid_numeric_beam_acc --exp_name "${EXP_NAME}" --exp_id "${EXP_ID}"
else
    python $LAWT_TRAIN --dump_path "${LAWT_DUMP_PATH}" --save_periodic 0 --fp16 true --amp 2 --accumulate_gradients 1 --clip_grad_norm 5 --enc_emb_dim 512 --dec_emb_dim 512 --n_enc_layers 8 --n_dec_layers 1 --n_enc_heads 8 --n_dec_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --optimizer 'adam_warmup,warmup_updates=10000,lr=0.0001' --batch_size 64 --batch_size_eval 128 --max_len 200 --epoch_size 300000 --max_epoch $MAX_EPOCHS --num_workers 10 --export_data false --reload_data '' --reload_size '-1' --batch_load false --env_name numeric --tasks numeric --env_base_seed '-1' --eval_size 10000 --min_dimension 5 --max_dimension 5 --max_input_coeff 5 --operation $OPERATION --generator gaussian --rectangular false --output_encoding 'floatsymbol,2' --input_encoding 'floatsymbol,2' --max_output_len 80 --float_tolerance '0.05' --more_tolerance '0.02,0.01,0.005' --eval_norm d1 --eval_verbose 0 --beam_eval 1 --stopping_criterion 'valid_numeric_beam_acc,60' --validation_metrics valid_numeric_beam_acc --exp_name "${EXP_NAME}" --exp_id "${EXP_ID}"
    echo "Experiment path: ${LAWT_DUMP_PATH}/${EXP_NAME}/${EXP_ID}"
fi


