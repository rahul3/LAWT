#!/bin/bash

# Run the bashrc first
. ~/.bashrc

export LAWT_PATH="/home/rahulpadmanabhan/projects/ws1/LAWT"
export LAWT_TRAIN="${LAWT_PATH}/train.py"

if [ "$(hostname)" = "node1" ]; then
    echo "Setting up node1"
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
    export LAWT_DUMP_PATH="/home/rahulpadmanabhan/Development/ws1/masters_thesis_2/experiments"
else
    export LAWT_DUMP_PATH="/path/to/default_directory"
fi

echo "LAWT_PATH: $LAWT_PATH"
echo "LAWT_TRAIN: $LAWT_TRAIN"
echo "LAWT_DUMP_PATH: $LAWT_DUMP_PATH"

# Experiment details
EXP_NAME="matrix_logarithm"
EXP_ID="202408251204"
MAX_EPOCHS=100

echo "Experiment name: $EXP_NAME"
echo "Experiment ID: $EXP_ID"
echo "Max epochs: $MAX_EPOCHS"

echo "Displaying the result of 'which python' command:"
which python

python $LAWT_TRAIN --dump_path "${LAWT_DUMP_PATH}" --save_periodic 0 --fp16 true --amp 2 --accumulate_gradients 1 --clip_grad_norm 5 --enc_emb_dim 512 --dec_emb_dim 512 --n_enc_layers 8 --n_dec_layers 1 --n_enc_heads 8 --n_dec_heads 8 --dropout 0 --attention_dropout 0 --share_inout_emb true --sinusoidal_embeddings false --optimizer 'adam_warmup,warmup_updates=10000,lr=0.0001' --batch_size 64 --batch_size_eval 128 --max_len 200 --epoch_size 300000 --max_epoch $MAX_EPOCHS --num_workers 10 --export_data false --reload_data '' --reload_size '-1' --batch_load false --env_name numeric --tasks numeric --env_base_seed '-1' --eval_size 10000 --min_dimension 5 --max_dimension 5 --max_input_coeff 5 --operation matrix_exponential --generator gaussian --rectangular false --output_encoding 'floatsymbol,2' --input_encoding 'floatsymbol,2' --max_output_len 80 --float_tolerance '0.05' --more_tolerance '0.02,0.01,0.005' --eval_norm d1 --eval_verbose 0 --beam_eval 1 --stopping_criterion 'valid_numeric_beam_acc,60' --validation_metrics valid_numeric_beam_acc --exp_name "${EXP_NAME}" --exp_id "${EXP_ID}"

echo "Experiment path: ${LAWT_DUMP_PATH}/${EXP_NAME}/${EXP_ID}"

