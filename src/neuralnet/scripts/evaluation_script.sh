#!/bin/bash

PYTHON_EXEC="/home/rahulpadmanabhan/Development/envs/venv310/bin/python"
LAWT_TRAIN="/home/rahulpadmanabhan/Development/ws1/masters_thesis_2/LAWT/train.py"
LAWT_DUMP_PATH="/home/rahulpadmanabhan/Development/ws1/masters_thesis_2/experiments/evaluations"

echo "LAWT_DUMP_PATH: $LAWT_DUMP_PATH"
echo "LAWT_TRAIN: $LAWT_TRAIN"
echo "PYTHON_EXEC: $PYTHON_EXEC"

$PYTHON_EXEC $LAWT_TRAIN --eval_verbose 1 --eval_only true --eval_size 30000000 --dump_path $LAWT_DUMP_PATH --eval_from_exp $1


