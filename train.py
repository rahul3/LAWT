# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import argparse
import numpy as np
import torch
import os
import pickle

import src
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import bool_flag, initialize_exp
from src.model import check_model_params, build_modules
from src.envs import ENVS, build_env
from src.trainer import Trainer
from src.evaluator import Evaluator


np.seterr(all='raise')


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--enc_emb_dim", type=int, default=256,
                        help="Encoder embedding layer size")
    parser.add_argument("--dec_emb_dim", type=int, default=256,
                        help="Decoder embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_dec_layers", type=int, default=4,
                        help="Number of Transformer layers in the decoder")
    parser.add_argument("--n_enc_heads", type=int, default=8,
                        help="Number of Transformer encoder heads")
    parser.add_argument("--n_dec_heads", type=int, default=8,
                        help="Number of Transformer decoder heads")
    
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    
    # training parameters
    parser.add_argument("--env_base_seed", type=int, default=-1,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Maximum sequences length")
    parser.add_argument("--max_output_len", type=int, default=512,
                        help="max length of output, beam max size")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--eval_size", type=int, default=10000,
                        help="Size of valid and test samples")
    parser.add_argument("--batch_size_eval", type=int, default=128,
                        help="Number of sentences per batch during evaluation")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=300000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of CPU workers for DataLoader")

    # export data / reload it
    parser.add_argument("--export_data", type=bool_flag, default=False,
                        help="Export data and disable training.")
    parser.add_argument("--reload_data", type=str, default="",
                        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)")
    parser.add_argument("--reload_size", type=int, default=-1,
                        help="Reloaded training set size (-1 for everything)")
    parser.add_argument("--batch_load", type=bool_flag, default=False,
                        help="Load training set by batches (of size reload_size).")

    # environment parameters
    parser.add_argument("--env_name", type=str, default="numeric",
                        help="Environment name")
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    # tasks
    parser.add_argument("--tasks", type=str, default="numeric",
                        help="Tasks")

    # beam search configuration
    parser.add_argument("--beam_eval", type=bool_flag, default=True,
                        help="Evaluate with beam search decoding.")
    parser.add_argument("--beam_eval_train", type=int, default=0,
                        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--beam_length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--beam_early_stopping", type=bool_flag, default=True,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_from_exp", type=str, default="",
                        help="Path of experiment to use")
    parser.add_argument("--eval_data", type=str, default="",
                        help="Path of data to eval")
    parser.add_argument("--eval_distrib", type=str, default="",
                        help="distributions to test")
    parser.add_argument("--eval_verbose", type=int, default=0,
                        help="Export evaluation details")
    parser.add_argument("--eval_verbose_print", type=bool_flag, default=False,
                        help="Print evaluation details")

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--windows", type=bool_flag, default=False,
                        help="Windows version (no multiprocessing for eval)")
    
    # Metals support
    parser.add_argument("--metals", type=bool_flag, default=False,
                        help="Use Metals")
    
    return parser


def main(params):

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if params.cpu:
        assert not params.multi_gpu
    elif params.metals:
       if torch.backends.mps.is_available(): 
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            print (x)
    else:
        assert torch.cuda.is_available()
    src.utils.CUDA = not params.cpu and not params.metals
    src.utils.METALS = params.metals

    # build environment / modules / trainer / evaluator
    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_equations = 0

        while trainer.n_equations < trainer.epoch_size:

            # training steps
            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                if params.export_data:
                    trainer.export_data(task)
                else:
                    trainer.enc_dec_step(task)
                trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals()

        # print / JSON log
        # for k, v in scores.items():
        #    logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    if params.eval_only and params.eval_from_exp != "":
        # breakpoint()
        # read params from pickle
        pickle_file = params.eval_from_exp + "/params.pkl"
        evalulation_size = params.eval_size
        evaluation_verbose = params.eval_verbose
        exp_str = params.eval_from_exp
        assert os.path.isfile(pickle_file)
        pk = pickle.load(open(pickle_file, 'rb'))
        pickled_args = pk.__dict__
        del pickled_args['exp_id']
        del pickled_args['dump_path']
        del pickled_args['exp_name']
        
        for p in params.__dict__:
            if p in pickled_args:
                params.__dict__[p] = pickled_args[p]

        params.eval_only = True
        params.eval_size = evalulation_size
        params.eval_verbose = evaluation_verbose
        params.reload_model = exp_str + '/best-' + params.validation_metrics + '.pth'
        # print(params.reload_model)
        assert os.path.isfile(params.reload_model)
        if params.eval_data != "":
            params.eval_size = None
            params.reload_data = params.tasks + ',' + params.eval_data + ',' + params.eval_data + ',' + params.eval_data
        if params.eval_distrib != "":
            eval_distrib = params.eval_distrib.split(',')
            params.eigen_test_distribution = eval_distrib[0]
            params.additional_test_distributions = ";".join(eval_distrib[1:])
        params.is_slurm_job = False
        params.local_rank = -1

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        if params.exp_id == '':
            params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)


# Namespace(dump_path='/home/rahulpadmanabhan/Development/scratch', exp_name='ipython_debug', save_periodic=0, exp_id='ipython_001', fp16=True, amp=-1, enc_emb_dim=256, dec_emb_dim=256, n_enc_layers=4, n_dec_layers=4, n_enc_heads=8, n_dec_heads=8, dropout=0, attention_dropout=0, share_inout_emb=True, sinusoidal_embeddings=False, env_base_seed=-1, max_len=512, max_output_len=512, batch_size=32, eval_size=10000, batch_size_eval=128, optimizer='adam,lr=0.0001', clip_grad_norm=5, epoch_size=300000, max_epoch=100000, stopping_criterion='', validation_metrics='', accumulate_gradients=1, num_workers=10, export_data=False, reload_data='', reload_size=-1, batch_load=False, env_name='numeric', operation='invert_matrix', cotraining_tasks='TADMEFI', generator='uniform', max_input_coeff=10, min_input_coeff=-1, force_dim=False, first_dimension=5, second_dimension=5, min_dimension=5, max_dimension=5, rectangular=False, eigen_distribution='semicircle', eigen_test_distribution='semicircle', additional_test_distributions='', noisy_input=False, sigma=0.05, classic_eval=False, max_encoder_dimension=100, output_encoding='float,2,10', input_encoding='float,2,10', float_tolerance=0.1, coeff_tolerance=0.01, more_tolerance='', eval_norm='d1', tasks=['numeric'], beam_eval=True, beam_eval_train=0, beam_size=1, beam_length_penalty=1, beam_early_stopping=True, reload_model='', reload_checkpoint='', eval_only=False, eval_from_exp='', eval_data='', eval_distrib='', eval_verbose=0, eval_verbose_print=False, debug_slurm=False, debug=False, cpu=False, local_rank=-1, master_port=-1, windows=False, n_words=1328, eos_index=0, pad_index=1)