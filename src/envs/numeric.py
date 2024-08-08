# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger

import math
import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators
from src.dataset import EnvDataset

from torch.utils.data import DataLoader

from ..utils import bool_flag


SPECIAL_WORDS = ["<s>", "</s>", "<pad>", "(", ")"] + [
    f"<SPECIAL_{i}>" for i in range(10)
] + ['T', 'A', 'D', 'M', 'E', 'F', 'I']
logger = getLogger()


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class NumericEnvironment(object):
    """
    NumericEnvironment class represents an environment for generating pairs of matrices and their corresponding inverses.
    It provides methods for encoding and decoding matrices, generating expressions, and evaluating predictions.

    Args:
        params (argparse.Namespace): An argparse.Namespace object containing the parameters for the environment.

    Attributes:
        TRAINING_TASKS (set): A set of training tasks supported by the environment.
        TEST_TASKS (set): A set of test tasks supported by the environment.
        max_len (int): The maximum length of the input and output sequences.
        operation (str): The type of operation to perform on the matrices.
        input_encoding (str): The encoding scheme for the input matrices.
        output_encoding (str): The encoding scheme for the output matrices.
        gaussian_coeffs (bool): A flag indicating whether to use Gaussian coefficients for generating matrices.
        eval_norm (str): The norm to use for evaluating predictions.
        float_tolerance (float): The tolerance for comparing floating-point numbers.
        coeff_tolerance (float): The tolerance for comparing matrix coefficients.
        additional_tolerance (list): A list of additional tolerances for comparing matrix coefficients.
        test_sets (list): A list of additional test distributions.
        generator (object): The generator object for generating matrices.
        output_encoder (object): The encoder object for encoding output matrices.
        input_encoder (object): The encoder object for encoding input matrices.
        max_output_length (int): The maximum length of the output sequence.
        max_input_coeff (float): The maximum value of the input matrix coefficients.
        noisy_input (bool): A flag indicating whether to add noise to the input matrices.
        sigma (float): The standard deviation of the noise to be added to the input matrices.
        common_symbols (list): A list of common symbols used in the vocabulary.
        words (list): The vocabulary of the environment.
        id2word (dict): A dictionary mapping word indices to words.
        word2id (dict): A dictionary mapping words to word indices.
        n_words (int): The number of words in the vocabulary.
        eos_index (int): The index of the end-of-sequence token in the vocabulary.
        pad_index (int): The index of the padding token in the vocabulary.

    Methods:
        input_to_infix(lst): Converts the input sequence to an infix expression.
        output_to_infix(lst): Converts the output sequence to an infix expression.
        gen_expr(data_type, task, train): Generates pairs of matrices and inverses.
        decode_class(i): Decodes the class label into a string representation.
        code_class(xi, yi): Encodes the class label for a pair of matrices.
        check_prediction(src, tgt, hyp): Checks the prediction against the target and returns the evaluation metrics.
        create_train_iterator(task, data_path, params): Creates a train iterator for the environment.
        create_test_iterator(data_type, task, data_path, batch_size, params, size): Creates a test iterator for the environment.
        register_args(parser): Registers environment parameters with the argument parser.
    """

    TRAINING_TASKS = {"numeric"}
    TEST_TASKS = {"numeric"}

    def __init__(self, params):

        self.max_len = params.max_len
        self.operation = params.operation
        input_coder = params.input_encoding.split(',')
        output_coder = params.output_encoding.split(',')

        self.output_encoding = output_coder[0]
        self.input_encoding = input_coder[0]

        self.gaussian_coeffs = (params.generator == 'normal')
        self.eval_norm = params.eval_norm

        self.float_tolerance = params.float_tolerance
        self.coeff_tolerance = params.coeff_tolerance
        self.additional_tolerance = [
            float(x) for x in params.more_tolerance.split(",") if len(x) > 0
        ]
        if params.additional_test_distributions == "":
            self.test_sets = []
        else:
            self.test_sets = params.additional_test_distributions.split(';')

        if self.operation == "invert_matrix":
            self.generator = generators.InvertMatrix(params)
        elif self.operation == "matrix_vector":
            self.generator = generators.DotProduct(params)
        elif self.operation == "matrix_product":
            self.generator = generators.MatrixProduct(params)
        elif self.operation == "matrix_sum":
            self.generator = generators.MatrixAdd(params)
        elif self.operation == "eigenvalues":
            self.generator = generators.Eigenvalues(params)
        elif self.operation == "eigenvectors":
            self.generator = generators.Eigenvectors(params)
        elif self.operation == "syminverse":
            self.generator = generators.SymInverse(params)
        elif self.operation == "singularvalues":
            self.generator = generators.Singularvalues(params)
        elif self.operation == "singularvectors":
            self.generator = generators.Singularvectors(params)
        elif self.operation == "transpose":
            self.generator = generators.TransposeMatrix(params)
        elif self.operation == "cotraining":
            self.generator = generators.CoTraining(params)
        elif self.operation == "matrix_exponential":
            self.generator = generators.MatrixExponential(params)
        elif self.operation == "matrix_cube":
            self.generator = generators.MatrixCube(params)
        elif self.operation == "matrix_logarithm":
            self.generator = generators.MatrixLogarithm(params)
        elif self.operation == "matrix_sign":
            self.generator = generators.MatrixSign(params)
        elif self.operation == "matrix_sine":
            self.generator = generators.MatrixSine(params)
        elif self.operation == "matrix_cosine":
            self.generator = generators.MatrixCosine(params)
        elif self.operation == "matrix_fractional_power":
            self.generator = generators.MatrixCosine(params)
        else:
            logger.error(f"Unknown operation {self.operation}")

        if self.output_encoding == "FP15":
            self.output_encoder = encoders.FPSymbol(params, int(output_coder[1]), int(output_coder[2]))
        elif self.output_encoding == "float":
            self.output_encoder = encoders.Positional(params, int(output_coder[1]), int(output_coder[2]))
        elif self.output_encoding == "floatsymbol":
            self.output_encoder = encoders.FloatSymbol(params, int(output_coder[1]))
        else:
            logger.error(f"Unknown encoder {self.output_encoding}")

        if self.input_encoding == "FP15":
            self.input_encoder = encoders.FPSymbol(params, int(input_coder[1]), int(input_coder[2]))
        elif self.input_encoding == "float":
            self.input_encoder = encoders.Positional(params, int(input_coder[1]), int(input_coder[2]))
        elif self.input_encoding == "floatsymbol":
            self.input_encoder = encoders.FloatSymbol(params, int(output_coder[1]))
        else:
            logger.error(f"Unknown encoder {self.input_encoding}")

        assert (
            self.input_encoder.limit <= 0.0
            or params.max_input_coeff <= self.input_encoder.limit
        )

        self.max_output_length = params.max_output_len
        self.max_input_coeff = params.max_input_coeff
        self.noisy_input = params.noisy_input
        self.sigma = params.sigma
        
        # vocabulary
        self.common_symbols = ['+', '-', '10^', '.', '|']
        self.words = SPECIAL_WORDS + self.common_symbols + sorted(list(
            set(self.output_encoder.symbols + self.input_encoder.symbols)
        ))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"vocabulary: {len(self.word2id)} words")
        if len(self.word2id) < 1000:
            logger.info(f"words: {self.word2id}")

    def input_to_infix(self, lst):
        """
        Convert the encoded input sequence to an infix expression.
        
        Example (for the Positional Encoder):
        >>> matrix
        array([[-0.86939421, -0.54711437,  1.28583033],
            [ 0.22848608, -0.45579835,  1.23640795]])
        >>> encoded_matrix
        ['V2', 'V3', '-', '869', 'E-3', '-', '547', 'E-3', '+', '129', 'E-2', '+', '228', 'E-3', '-', '456', 'E-3', '+', '124', 'E-2']
        >>> env.input_to_infix(encoded_matrix)
        '[[-0.869 -0.547  1.29 ]\n [ 0.228 -0.456  1.24 ]]'
        """
        if self.operation == "cotraining":
            code = lst[0]
            ll = lst[1:]
        else:
            code = None
            ll = lst
        m = self.input_encoder.decode(ll)
        if m is None:
            return "Invalid"
        res = np.array2string(m)
        if code is None:
            return res
        return res + code 

    def output_to_infix(self, lst):
        """
        Convert the encoded output sequence to an infix expression.
        
        Current implementation is same as input_to_infix method. There is no difference between the two.
        """
        if self.operation == "cotraining":
            code = lst[0]
            ll = lst[1:]
        else:
            code = None
            ll = lst
        m = self.input_encoder.decode(ll)
        if m is None:
            return "Invalid"
        res = np.array2string(m)
        if code is None:
            return res
        return res + code

    def gen_expr(self, data_type=None, task=None, train=True):
        """
        Generate pairs of matrices and inverses
        Encode this as a prefix sentence
        """
        # TODO: What happens when you want to test and self.rng is not defined?
        gen = self.generator.generate(self.rng, self.gaussian_coeffs, self.output_encoder.limit, data_type)
        if gen is None:
            return None
        if self.operation == "cotraining":
            x_data, y_data, code = gen
        else:
            x_data, y_data = gen
        if self.noisy_input:
            n, p = np.shape(x_data)
            x_data = x_data + (self.max_input_coeff * self.sigma / math.sqrt(3.0)) * self.rng.randn(n, p)
        # encode input
        x = self.input_encoder.encode(x_data)
        # encode output
        y = self.output_encoder.encode(y_data)
        if self.operation == "cotraining":
            x = [code] + x
            y = [code] + y
        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None
        return x, y

    def decode_class(self, i):
        if self.operation == 'cotraining':
            return chr(i)
        return str(i // 100) + '-' + str(i % 100)

    def code_class(self, xi, yi):
        if self.operation == 'cotraining':
            return ord(xi[0])
        offset = 0
        nr_lines = int(xi[offset][1:]) * 100 + int(xi[offset + 1][1:])
        return nr_lines

    def check_prediction(self, src, tgt, hyp):
        if self.operation == "cotraining":
            if len(hyp) == 0 or len(src) == 0 or len(tgt) == 0:
                return -1.0, -1.0, -1.0, -1.0
            code = src[0]
            if hyp[0] != code or tgt[0] != code:
                return -1.0, -1.0, -1.0, -1.0
            h = hyp[1:]
            s = src[1:]
            t = tgt[1:]
        else:
            code = None
            h = hyp
            s = src
            t = tgt

        if len(h) == 0 or len(t) == 0:
            return -1.0, -1.0, -1.0, -1.0
        mat_hyp = self.output_encoder.decode(h)
        if mat_hyp is None:
            return -1.0, -1.0, -1.0, -1.0
        mat_tgt = self.output_encoder.decode(t)
        if np.shape(mat_hyp) != np.shape(mat_tgt):
            return -1.0, -1.0, -1.0, -1.0
        mat_src = self.input_encoder.decode(s)
        max_n, d1_n, d2_n, nb = self.generator.evaluate(mat_src, mat_tgt, mat_hyp, self.coeff_tolerance, code)
        if self.eval_norm == "d1":
            return d1_n, d2_n, max_n, nb
        elif self.eval_norm == "d2":
            return d2_n, d1_n, max_n, nb
        else:
            return max_n, d1_n, d2_n, nb

    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
            type = "train"
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"] + self.test_sets
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][1 if data_type == "valid" else 2]
            ),
            size=size,
            type=data_type,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--operation", type=str, default="invert_matrix", help="Operation performed"
        )
        parser.add_argument(
            "--cotraining_tasks", type=str, default="TADMEFI", help="Cotraining operations"
        )
        
        parser.add_argument(
            "--generator", type=str, default="uniform", help="Random generation of coefficients"
        )
        parser.add_argument(
            "--max_input_coeff", type=int, default=10, help="max coeff of input matrix"
        )
        parser.add_argument(
            "--min_input_coeff", type=int, default=-1, help="min coeff of input matrix, if -1 min = max"
        )
           
        # generation parameters
        parser.add_argument(
            "--force_dim", type=bool_flag, default=False, help="unique dimension in dataset"
        )
        parser.add_argument(
            "--first_dimension", type=int, default=5, help="first dimension if force_dim"
        )
        parser.add_argument(
            "--second_dimension", type=int, default=5, help="second dimension if force_dim"
        )

        parser.add_argument(
            "--min_dimension", type=int, default=5, help="minimum dimension of space"
        )
        parser.add_argument(
            "--max_dimension", type=int, default=5, help="maximum dimension of space"
        )
        parser.add_argument(
            "--rectangular", type=bool_flag, default=False, help="rectangular matrices allowed"
        )

        parser.add_argument(
            "--eigen_distribution", type=str, default="semicircle", help="eigenvalue distribution (semicircle, positive, uniform, gaussian, laplace)"
        )
        parser.add_argument(
            "--eigen_test_distribution", type=str, default="semicircle", help="eigenvalue test distribution (semicircle, positive, uniform, gaussian, laplace)"
        )
        parser.add_argument(
            "--additional_test_distributions", type=str, default="", help="additional test sets, sets separated by ; generators by ,"
        )
        parser.add_argument(
            "--noisy_input", type=bool_flag, default=False, help="add gaussian noise to input"
        )

        parser.add_argument(
            "--sigma",
            type=float,
            default=0.05,
            help="deviation of input noise (as a proportion of coeff deviation)",
        )
        parser.add_argument(
            "--classic_eval", type=bool_flag, default=False, help="evaluate inversion as other ops"
        )
        
        # representation parameters
        parser.add_argument(
            "--max_encoder_dimension",
            type=int,
            default=100,
            help="maximum dimension of space for encoder",
        )
        # FP15, precision, max_exp
        # float, precision, base_int
        # floatsymbol, precision
        parser.add_argument(
            "--output_encoding", type=str, default="float,2,1000", help="Encoder for output sequence"
        )
        parser.add_argument(
            "--input_encoding", type=str, default="float,2,1000", help="Encoder for input sequence"
        )

        # evaluation parameters
        parser.add_argument(
            "--float_tolerance",
            type=float,
            default=0.1,
            help="error tolerance for float results",
        )
        parser.add_argument(
            "--coeff_tolerance",
            type=float,
            default=0.01,
            help="error tolerance for nb of correct coefficients",
        )
        parser.add_argument(
            "--more_tolerance", type=str, default="", help="additional tolerance limits"
        )
        parser.add_argument(
            "--eval_norm", type=str, default="d1", help="norm to use for evaluation, max, d1 or d2"
        )

        # functions of matrices parameters
        parser.add_argument(
            "--p", type=float, default=0.5, help="For matrix fractional power"
        )


if __name__ == '__main__':
    import argparse
    from utils import bool_flag

    parser = argparse.ArgumentParser()
    NumericEnvironment.register_args(parser)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--input_encoding', type=str, default='float,2,1000')
    parser.add_argument('--output_encoding', type=str, default='float,2,1000')  
    parser.add_argument('--eval_norm', type=str, default='d1')
    parser.add_argument('--float_tolerance', type=float, default=0.1)
    parser.add_argument('--coeff_tolerance', type=float, default=0.01)
    parser.add_argument('--more_tolerance', type=str, default='')
    parser.add_argument('--additional_test_distributions', type=str, default='')
    parser.add_argument('--min_dimension', type=int, default=5)
    parser.add_argument('--max_dimension', type=int, default=5)
    parser.add_argument('--rectangular', type=bool_flag, default=False)
    parser.add_argument('--force_dim', type=bool_flag, default=False)
    parser.add_argument('--first_dimension', type=int, default=5)
    parser.add_argument('--second_dimension', type=int, default=5)
    parser.add_argument('--max_output_len', type=int, default=100)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--env_base_seed', type=int, default=0)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch_load', type=bool_flag, default=False)
    parser.add_argument('--num_workers', type=int, default=0) 
    parser.add_argument('--reload_size', type=int, default=-1)
    parser.add_argument('--n_gpu_per_node', type=int, default=1)

    params=parser.parse_args()
    print(params)

    env = NumericEnvironment(params)
