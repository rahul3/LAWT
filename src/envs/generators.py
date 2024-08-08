# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import numpy as np
import scipy
import math
from logging import getLogger

import scipy.linalg

logger = getLogger()


class Generator(ABC):
    """
    Abstract base class for generators.

    Attributes:
        min_dimension (int): The minimum dimension of the generated matrix.
        max_dimension (int): The maximum dimension of the generated matrix.
        rectangular (bool): Indicates whether the generated matrix should be rectangular.
        max_input_coeff (int): The maximum value for the input coefficients.
        min_input_coeff (int): The minimum value for the input coefficients.
        force_dim (bool): Indicates whether to force a specific dimension for the generated matrix.
        first_dimension (int): The first dimension of the generated matrix when force_dim is True.
        second_dimension (int): The second dimension of the generated matrix when force_dim is True.
    """

    def __init__(self, params):
        self.min_dimension = params.min_dimension
        self.max_dimension = params.max_dimension
        self.rectangular = params.rectangular
        assert self.min_dimension <= params.max_dimension
        self.max_input_coeff = params.max_input_coeff
        self.min_input_coeff = self.max_input_coeff if params.min_input_coeff <= 0 else params.min_input_coeff
        
        self.force_dim = params.force_dim
        self.first_dimension = params.first_dimension
        self.second_dimension = params.second_dimension

    def rand_matrix(self, rng, dim1, dim2, gaussian, max_coeff):
        """
        Generate a random matrix.

        Args:
            rng (RandomState): The random number generator.
            dim1 (int): The first dimension of the matrix.
            dim2 (int): The second dimension of the matrix.
            gaussian (bool): Indicates whether to generate the matrix using Gaussian distribution.
            max_coeff (int): The maximum value for the matrix coefficients.

        Returns:
            numpy.ndarray: The generated random matrix.
        """
        if gaussian:
            return np.array(max_coeff / math.sqrt(3.0) * rng.randn(dim1, dim2))
        else:
            return np.array(max_coeff * (2 * rng.rand(dim1, dim2) - 1))

    def gen_matrix(self, rng, gaussian):
        """
        Generate a matrix.

        Args:
            rng (RandomState): The random number generator.
            gaussian (bool): Indicates whether to generate the matrix using Gaussian distribution.

        Returns:
            numpy.ndarray: The generated matrix.
        """
        if self.force_dim:
            dim = self.first_dimension
            dim2 = self.second_dimension
        else:
            dim = rng.randint(self.min_dimension, self.max_dimension + 1)
            dim2 = rng.randint(self.min_dimension, self.max_dimension + 1) if self.rectangular else dim
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        matrix = self.rand_matrix(rng, dim, dim2, gaussian, max_coeff)
        return matrix

    @abstractmethod
    def generate(self, rng, gaussian, output_limit, type):
        """
        Abstract method to generate a matrix.

        Args:
            rng (RandomState): The random number generator.
            gaussian (bool): Indicates whether to generate the matrix using Gaussian distribution.
            output_limit (int): The maximum number of matrices to generate.
            type (str): The type of matrix to generate.

        Returns:
            list: The generated matrices.
        """
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp, prec, code):
        """
        Abstract method to evaluate the generated matrices.

        Args:
            src (numpy.ndarray): The source matrix.
            tgt (numpy.ndarray): The target matrix.
            hyp (numpy.ndarray): The generated matrix.
            prec (float): The precision value.
            code (str): The code snippet.

        Returns:
            float: The evaluation result.
        """
        pass
    



class TransposeMatrix(Generator):
    """
    A generator class for transposing matrices.

    This class provides methods to generate and evaluate transposed matrices.

    Args:
        params (dict): A dictionary of parameters for the generator.

    Attributes:
        params (dict): A dictionary of parameters for the generator.

    """

    def __init__(self, params):
        super().__init__(params)

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        """
        Generate a transposed matrix.

        Args:
            rng (numpy.random.Generator): The random number generator.
            gaussian (bool): Whether to generate a Gaussian matrix.
            output_limit (float, optional): The maximum absolute value of any element in the generated matrix.
                If set to a non-negative value, the generated matrix will be checked against this limit.
                Defaults to -1.0.
            type (str, optional): The type of matrix to generate. Defaults to None.

        Returns:
            tuple: A tuple containing the original matrix and its transposed version.

        """
        matrix = self.gen_matrix(rng, gaussian)
        result = matrix.T
        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        """
        Evaluate the transposed matrix.

        Args:
            src (numpy.ndarray): The source matrix.
            tgt (numpy.ndarray): The target matrix.
            hyp (numpy.ndarray): The hypothesis matrix.
            prec (float, optional): The precision threshold for evaluating the matrix.
                Defaults to 0.01.
            code (str, optional): The code for evaluation. Defaults to None.

        Returns:
            tuple: A tuple containing the evaluation metrics.

        """
        m = src.T - hyp
        s = src
        e = np.sum(np.abs(m) / (np.abs(src.T) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e




class InvertMatrix(Generator):
    """
    A generator class for generating inverted matrices.

    Args:
        params (object): The parameters for the generator.

    Attributes:
        classic_eval (bool): Flag indicating whether to use classic evaluation.

    Methods:
        generate(rng, gaussian, output_limit=-1.0, type=None):
            Generates an inverted matrix.

        evaluate(src, tgt, hyp, prec=0.01, code=None):
            Evaluates the generated inverted matrix.

    """

    def __init__(self, params):
        super().__init__(params)
        self.classic_eval = params.classic_eval

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        """
        Generates an inverted matrix.

        Args:
            rng (object): The random number generator.
            gaussian (bool): Flag indicating whether to use Gaussian distribution.
            output_limit (float, optional): The maximum absolute value of the elements in the inverted matrix.
                Defaults to -1.0, indicating no limit.
            type (object, optional): The type of the matrix. Defaults to None.

        Returns:
            tuple: A tuple containing the generated matrix and its inverse.

        """
        dim = rng.randint(self.min_dimension, self.max_dimension + 1)
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        matrix = self.rand_matrix(rng, dim, dim, gaussian, max_coeff)
        # no regularization for inversion, add a small value if necessary
        gamma = 0.0
        inverse = np.linalg.inv(matrix + gamma * np.eye(dim))
        # check result
        error = np.max(abs(matrix @ inverse - np.eye(dim)))
        if error > 1.0e-10:
            return None
        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(inverse))
            if max_coeff_y >= output_limit:
                return None
        return matrix, inverse

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        """
        Evaluates the generated inverted matrix.

        Args:
            src (object): The source matrix.
            tgt (object): The target matrix.
            hyp (object): The generated inverted matrix.
            prec (float, optional): The precision threshold for evaluation. Defaults to 0.01.
            code (object, optional): The code object. Defaults to None.

        Returns:
            tuple: A tuple containing the evaluation metrics.

        """
        if self.classic_eval:
            m = hyp - tgt
            s = tgt
            if np.max(np.abs(s)) == 0.0:
                if np.max(np.abs(m)) == 0.0:
                    return 0.0, 0.0, 0.0, 1.0
                else:
                    return -1.0, -1.0, -1.0, 0.0
            e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
            return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s),

        dim = np.shape(hyp)[0]
        m = src @ hyp - np.eye(dim)
        e = np.sum(np.abs(m) < prec) / m.size
        return np.max(np.abs(m)), np.sum(np.abs(m)) / dim, np.trace(m.T @ m) / dim, e


class DotProduct(Generator):
    def __init__(self, params):
        super().__init__(params)

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        if self.force_dim:
            dim = self.first_dimension
            dim2 = self.second_dimension
        else:
            dim = rng.randint(self.min_dimension, self.max_dimension + 1)
            dim2 = rng.randint(self.min_dimension, self.max_dimension + 1) if self.rectangular else dim
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        matrix = self.rand_matrix(rng, dim, dim2 + 1, gaussian, max_coeff)

        result = np.zeros((dim2, 1), dtype=float)
        for i in range(dim2):
            for j in range(dim):
                result[i, 0] += matrix[j, i] * matrix[j, dim2]

        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        m = hyp - tgt
        s = tgt
        if np.max(np.abs(s)) == 0.0:
            if np.max(np.abs(m)) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class MatrixProduct(Generator):
    def __init__(self, params):
        super().__init__(params)

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        if self.force_dim:
            dim = self.first_dimension
            dim2 = self.second_dimension
        else:
            dim = rng.randint(self.min_dimension, self.max_dimension + 1)
            dim2 = rng.randint(self.min_dimension, self.max_dimension + 1) if self.rectangular else dim
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        matrix = self.rand_matrix(rng, dim2, 2 * dim, gaussian, max_coeff)

        result = matrix[:, 0:dim].T @ matrix[:, dim:(2 * dim)]
        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        m = hyp - tgt
        s = tgt
        if np.max(np.abs(s)) == 0.0:
            if np.max(np.abs(m)) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class MatrixAdd(Generator):
    def __init__(self, params):
        super().__init__(params)

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        if self.force_dim:
            dim = self.first_dimension
            dim2 = self.second_dimension
        else:
            dim = rng.randint(self.min_dimension, self.max_dimension + 1)
            dim2 = rng.randint(self.min_dimension, self.max_dimension + 1) if self.rectangular else dim
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        matrix = self.rand_matrix(rng, dim2, 2 * dim, gaussian, max_coeff)
        result = matrix[:, 0:dim] + matrix[:, dim:(2 * dim)]
        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        m = hyp - tgt
        s = tgt
        if np.max(np.abs(s)) == 0.0:
            if np.max(np.abs(m)) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class Eigenvalues(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.eigen_distribution = params.eigen_distribution.split(',')
        self.eigen_test_distribution = params.eigen_test_distribution.split(',')
        assert len(self.eigen_distribution) > 0 and len(self.eigen_distribution) <= 5
        assert len(self.eigen_test_distribution) > 0 and len(self.eigen_test_distribution) <= 5
        self.test_sets = params.additional_test_distributions.split(';')
        self.test_distribs = {}
        for v in self.test_sets:
            self.test_distribs[v] = v.split(',')
            assert len(self.test_distribs[v]) > 0 and len(self.test_distribs[v]) <= 5

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        dim = rng.randint(self.min_dimension, self.max_dimension + 1)
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        a = self.rand_matrix(rng, dim, dim, gaussian, max_coeff)
        matrix = np.tril(a) + np.tril(a, -1).T
        dist = []
        if type == "train":
            dist = self.eigen_distribution
        elif type == "valid":
            dist = self.eigen_test_distribution
        else:
            dist = self.test_distribs[type]
        distrib = rng.choice(dist)
        if distrib in ["positive", "uniform", "gaussian", "laplace", "positive2"]:
            val, vec = np.linalg.eigh(matrix)
            sigma = math.sqrt(dim) * max_coeff / math.sqrt(3.0)
            if distrib == "positive":
                val = np.abs(val)
            elif distrib == "uniform":
                val = sigma * math.sqrt(3.0) * (2 * rng.rand(dim) - 1)
            elif distrib == "gaussian":
                val = sigma * rng.randn(dim)
            elif distrib == "laplace":
                val = rng.laplace(0, sigma / math.sqrt(2.0), dim)
            elif distrib == "positive2":
                val = np.abs(rng.laplace(0, sigma / math.sqrt(2.0), dim))
            matrix = vec.T @ np.diag(val) @ vec 
        elif distrib == "marcenko":
            m1 = self.rand_matrix(rng, dim, dim, gaussian, math.sqrt(max_coeff))
            matrix = m1.T @ m1
            
        eig = np.linalg.eigvals(matrix)
        result = np.flip(np.sort(eig)).reshape(-1, 1)

        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        m = hyp - tgt
        s = tgt
        if np.max(np.abs(s)) == 0.0:
            if np.max(np.abs(m)) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class Eigenvectors(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.eigen_distribution = params.eigen_distribution.split(',')
        self.eigen_test_distribution = params.eigen_test_distribution.split(',')
        assert len(self.eigen_distribution) > 0 and len(self.eigen_distribution) <= 5
        assert len(self.eigen_test_distribution) > 0 and len(self.eigen_test_distribution) <= 5
        self.test_sets = params.additional_test_distributions.split(';')
        self.test_distribs = {}
        for v in self.test_sets:
            self.test_distribs[v] = v.split(',')
            assert len(self.test_distribs[v]) > 0 and len(self.test_distribs[v]) <= 5


    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        dim = rng.randint(self.min_dimension, self.max_dimension + 1)
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        a = self.rand_matrix(rng, dim, dim, gaussian, max_coeff)
        matrix = np.tril(a) + np.tril(a, -1).T

        dist = []
        if type is "train":
            dist = self.eigen_distribution
        elif type == "valid":
            dist = self.eigen_test_distribution
        else:
            dist = self.test_distribs[type]
        distrib = rng.choice(dist)
        if distrib in ["positive", "uniform", "gaussian", "laplace","positive2"]:
            val, vec = np.linalg.eigh(matrix)
            sigma = math.sqrt(dim) * max_coeff / math.sqrt(3.0)
            if distrib == "positive":
                val = np.abs(val)
            elif distrib == "uniform":
                val = sigma * math.sqrt(3.0) * (2 * rng.rand(dim) - 1)
            elif distrib == "gaussian":
                val = sigma * rng.randn(dim)
            elif distrib == "laplace":
                val = rng.laplace(0, sigma / math.sqrt(2.0), dim)
            elif distrib == "positive2":
                val = np.abs(rng.laplace(0, sigma / math.sqrt(2.0), dim))           
            matrix = vec.T @ np.diag(val) @ vec 
        elif distrib ==  "marcenko":
            m1 = self.rand_matrix(rng, dim, dim, gaussian, math.sqrt(max_coeff))
            matrix = m1.T @ m1

        val, vec = np.linalg.eigh(matrix)
        result = np.vstack((val, vec))

        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        s = np.diag(hyp[0])
        vec = hyp[1:]
        m = (vec.T @ src @ vec) - s
        if np.max(np.abs(s)) == 0.0:
            if np.max(np.abs(m)) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class SymInverse(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.classic_eval = params.classic_eval
        self.eigen_distribution = params.eigen_distribution.split(',')
        self.eigen_test_distribution = params.eigen_test_distribution.split(',')
        assert len(self.eigen_distribution) > 0 and len(self.eigen_distribution) <= 5
        assert len(self.eigen_test_distribution) > 0 and len(self.eigen_test_distribution) <= 5
        self.test_sets = params.additional_test_distributions.split(';')
        self.test_distribs = {}
        for v in self.test_sets:
            self.test_distribs[v] = v.split(',')
            assert len(self.test_distribs[v]) > 0 and len(self.test_distribs[v]) <= 5


    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        dim = rng.randint(self.min_dimension, self.max_dimension + 1)
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        a = self.rand_matrix(rng, dim, dim, gaussian, max_coeff)
        matrix = np.tril(a) + np.tril(a, -1).T

        dist = []
        if type is "train":
            dist = self.eigen_distribution
        elif type == "valid":
            dist = self.eigen_test_distribution
        else:
            dist = self.test_distribs[type]
        distrib = rng.choice(dist)
        if distrib in ["positive", "uniform", "gaussian", "laplace", "positive2"]:
            val, vec = np.linalg.eigh(matrix)
            sigma = math.sqrt(dim) * max_coeff / math.sqrt(3.0)
            if distrib == "positive":
                val = np.abs(val)
            elif distrib == "uniform":
                val = sigma * math.sqrt(3.0) * (2 * rng.rand(dim) - 1)
            elif distrib == "gaussian":
                val = sigma * rng.randn(dim)
            elif distrib == "laplace":
                val = rng.laplace(0, sigma / math.sqrt(2.0), dim)
            elif distrib == "positive2":
                val = np.abs(rng.laplace(0, sigma / math.sqrt(2.0), dim))           
            matrix = vec.T @ np.diag(val) @ vec 
        elif distrib ==  "marcenko":
            m1 = self.rand_matrix(rng, dim, dim, gaussian, math.sqrt(max_coeff))
            matrix = m1.T @ m1

        gamma = 0.0
        inverse = np.linalg.inv(matrix + gamma * np.eye(dim))
        # check result
        error = np.max(abs(matrix @ inverse - np.eye(dim)))
        if error > 1.0e-10:
            return None
        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(inverse))
            if max_coeff_y >= output_limit:
                return None
        return matrix, inverse

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        if self.classic_eval:
            m = hyp - tgt
            s = tgt
            if np.max(np.abs(s)) == 0.0:
                if np.max(np.abs(m)) == 0.0:
                    return 0.0, 0.0, 0.0, 1.0
                else:
                    return -1.0, -1.0, -1.0, 0.0
            e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
            return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), 

        dim = np.shape(hyp)[0]
        m = src @ hyp - np.eye(dim)
        e = np.sum(np.abs(m) < prec) / m.size
        return np.max(np.abs(m)), np.sum(np.abs(m)) / dim, np.trace(m.T @ m) / dim, e


class Singularvalues(Generator):
    def __init__(self, params):
        super().__init__(params)
        
    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        if self.force_dim:
            dim = self.first_dimension
            dim2 = self.second_dimension
        else:
            dim = rng.randint(self.min_dimension, self.max_dimension + 1)
            dim2 = rng.randint(self.min_dimension, self.max_dimension + 1) if self.rectangular else dim
        max_coeff = rng.randint(self.min_input_coeff,self.max_input_coeff + 1)
        matrix = self.rand_matrix(rng, dim, dim2, gaussian, max_coeff)
        sing = np.linalg.svd(matrix, compute_uv=False)
        result = sing.reshape(-1, 1)

        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        m = hyp@ - tgt
        s = tgt
        if np.max(np.abs(s)) == 0.0:
            if np.max(np.abs(m)) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class Singularvectors(Generator):
    def __init__(self, params):
        super().__init__(params)

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        if self.force_dim:
            dim = self.first_dimension
            dim2 = self.second_dimension
        else:
            dim = rng.randint(self.min_dimension, self.max_dimension + 1)
            dim2 = rng.randint(self.min_dimension, self.max_dimension + 1) if self.rectangular else dim
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        matrix = self.rand_matrix(rng, dim, dim2, gaussian, max_coeff)
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        result = np.vstack((s, u, v.T))

        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        s = np.diag(hyp[0])
        dim = np.shape(src)[0] + 1
        u = hyp[1:dim]
        v = hyp[dim:]
        m = u.T @ src @ v - s
        if np.max(np.abs(s)) == 0.0:
            if np.max(np.abs(m)) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class CoTraining(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.tasks = params.cotraining_tasks
        self.subgen = {}
        for v in self.tasks:
            if v == 'T':
                self.subgen[v] = TransposeMatrix(params)
            elif v == 'A':
                self.subgen[v] = MatrixAdd(params)
            elif v == 'D':
                self.subgen[v] = DotProduct(params)
            elif v == 'M':
                self.subgen[v] = MatrixProduct(params)
            elif v == 'E':
                self.subgen[v] = Eigenvalues(params)
            elif v == 'F':
                self.subgen[v] = Eigenvectors(params)
            elif v == 'I':
                self.subgen[v] = InvertMatrix(params)
            else:
                logger.error(f"Unknown task code: {v}")

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        v = rng.choice(list(self.tasks))
        x , y = self.subgen[v].generate(rng, gaussian, output_limit, type)
        return x, y, v

    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        return self.subgen[code].evaluate(src, tgt, hyp, prec)
    

class MatrixCube(Generator):
    def __init__(self, params):
        super().__init__(params)

    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        if self.rectangular:
            return None
        if self.force_dim:
            dim = self.first_dimension
            dim2 = self.second_dimension
        else:
            dim = rng.randint(self.min_dimension, self.max_dimension + 1)
            dim2 = rng.randint(self.min_dimension, self.max_dimension + 1) if self.rectangular else dim
        max_coeff = rng.randint(self.min_input_coeff, self.max_input_coeff + 1)
        matrix = self.rand_matrix(rng, dim, dim2, gaussian, max_coeff)
        result = matrix @ matrix @ matrix  # cube
        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result
    
    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        m = hyp - tgt
        s = tgt
        if np.max(np.abs(s)) == 0.0:
            if np.max(np.abs(m)) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class MatrixExponential(Generator):
    """ A generator class for generating exponential matrices. """
    
    def __init__(self, params):
        super().__init__(params)
        
    def generate(self, rng, gaussian, output_limit=-1.0, type=None):
        matrix = self.gen_matrix(rng, gaussian)
        result = scipy.linalg.expm(matrix)
        if output_limit >= 0.0:
            max_coeff_y = np.max(np.abs(result))
            if max_coeff_y >= output_limit:
                return None
        return matrix, result
    
    
    def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
        m = hyp - tgt
        s = tgt
        e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
        return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e
    

class MatrixLogarithm(Generator):
        """ A generator class for generating logarithm matrices. """
        
        def __init__(self, params):
            super().__init__(params)
            
        def generate(self, rng, gaussian, output_limit=-1.0, type=None):
            matrix = self.gen_matrix(rng, gaussian)
            result = scipy.linalg.logm(matrix)
            if output_limit >= 0.0:
                max_coeff_y = np.max(np.abs(result))
                if max_coeff_y >= output_limit:
                    return None
            return matrix, result
        
        
        def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
            m = hyp - tgt
            s = tgt
            e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
            return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e
        

class MatrixSign(Generator):
        """
        A generator class for generating sign matrices.
        """
        
        def __init__(self, params):
            super().__init__(params)
            
        def generate(self, rng, gaussian, output_limit=-1.0, type=None):
            matrix = self.gen_matrix(rng, gaussian)
            result = scipy.linalg.signm(matrix)
            if output_limit >= 0.0:
                max_coeff_y = np.max(np.abs(result))
                if max_coeff_y >= output_limit:
                    return None
            return matrix, result
        
        
        def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
            m = hyp - tgt
            s = tgt
            e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
            return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e
        

class MatrixSine(Generator):
        """
        A generator class for generating sine matrices.
        """
        
        def __init__(self, params):
            super().__init__(params)
            
        def generate(self, rng, gaussian, output_limit=-1.0, type=None):
            matrix = self.gen_matrix(rng, gaussian)
            result = scipy.linalg.sinm(matrix)
            if output_limit >= 0.0:
                max_coeff_y = np.max(np.abs(result))
                if max_coeff_y >= output_limit:
                    return None
            return matrix, result
        
        
        def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
            m = hyp - tgt
            s = tgt
            e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
            return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e
        

class MatrixCosine(Generator):
        """
        A generator class for generating cosine matrices.
        """
        
        def __init__(self, params):
            super().__init__(params)
            
        def generate(self, rng, gaussian, output_limit=-1.0, type=None):
            matrix = self.gen_matrix(rng, gaussian)
            result = scipy.linalg.cosm(matrix)
            if output_limit >= 0.0:
                max_coeff_y = np.max(np.abs(result))
                if max_coeff_y >= output_limit:
                    return None
            return matrix, result
        
        
        def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
            m = hyp - tgt
            s = tgt
            e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
            return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e


class MatrixPthRoot(Generator):
        """
        A generator class for generating pth root matrices.
        """
        
        def __init__(self, params):
            super().__init__(params)
            self.p = params.p
            
        def generate(self, rng, gaussian, output_limit=-1.0, type=None):
            matrix = self.gen_matrix(rng, gaussian)
            result = scipy.linalg.fractional_matrix_power(matrix, 1/self.p)
            if output_limit >= 0.0:
                max_coeff_y = np.max(np.abs(result))
                if max_coeff_y >= output_limit:
                    return None
            return matrix, result
        
        
        def evaluate(self, src, tgt, hyp, prec=0.01, code=None):
            m = hyp - tgt
            s = tgt
            e = np.sum(np.abs(m) / (np.abs(s) + 1e-12) < prec) / m.size
            return np.max(np.abs(m)) / np.max(np.abs(s)), np.sum(np.abs(m)) / np.sum(np.abs(s)), np.trace(m.T @ m) / np.trace(s.T @ s), e