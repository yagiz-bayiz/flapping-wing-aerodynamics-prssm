"""
This code is a modification (matern 3/2 kernel is implemented) of th source code in PRSSM (Doerr et al., 2018
(PRSSM github)
"""

"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
@author: Andreas Doerr

The following snippets are derived from GPFlow V 1.0
  (https://github.com/GPflow/GPflow)
Copyright 2017 st--, Mark van der Wilk, licensed under the Apache License, Version 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

import tensorflow as tf
import numpy as np

from .tf_transform import forward, backward


class MATERN32:
    def __init__(self, input_dim, variance, lengthscales, dtype=tf.float64):
        with tf.name_scope('kern'):
            self.variance_unc = tf.Variable(backward(variance), dtype=dtype)
            self.variance = forward(self.variance_unc)

            self.lengthscales_unc = tf.Variable(backward(lengthscales) * tf.ones((input_dim,), dtype=dtype))
            self.lengthscales = forward(self.lengthscales_unc)

    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def Kdiag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    def K(self, X, X2=None):
        sqrt3d = (3 ** 0.5) * tf.sqrt(self.square_dist(X, X2) + 1e-8)
        kernel = self.variance * (1 + sqrt3d) * tf.exp(-sqrt3d)
        return kernel


def _jitter_cholesky(mat, jitter):
    mat = tf.matrix_set_diag(mat, tf.diag_part(mat) + jitter)
    return tf.cholesky(mat)


def cast_cholesky(mat, jitter=1e-8):
    """Compute cholesky decomposition, but first cast to float64."""
    dtype = mat.dtype
    if dtype == tf.float64:
        return _jitter_cholesky(mat, jitter=jitter)
    else:
        mat = tf.cast(mat, tf.float64)
        chol = _jitter_cholesky(mat, jitter=jitter)
        return tf.cast(chol, dtype)


def conditional(Xnew, X, kern, f, q_sqrt, Lm=None):
    # compute kernel stuff
    num_func = tf.shape(f)[1]
    Kmn = kern.K(X, Xnew)
    if Lm is None:
        Lm = cast_cholesky(kern.K(X), jitter=1e-8)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
    shape = tf.stack([num_func, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)

    # another backsubstitution in the unwhitened case
    A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)
        elif q_sqrt.get_shape().ndims == 3:
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(q_sqrt, A_tiled, transpose_a=True)
        else:
            raise ValueError("bad dimension for q_sqrt")

        fvar += tf.reduce_sum(tf.square(LTA), 1)

    return fmean, tf.transpose(fvar)


class GPModel(object):
    def __init__(self, in_dim, out_dim, num_points, gp_var, gp_len, zeta_mean,
                 zeta_pos, zeta_var, dtype=tf.float32):

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype
        self.num_points = num_points

        self.zeta_pos = tf.Variable(np.random.uniform(low=-zeta_pos,
                                                      high=zeta_pos,
                                                      size=(num_points, in_dim)),
                                    dtype=dtype)

        self.zeta_mean = tf.Variable(zeta_mean * np.random.rand(num_points, out_dim),
                                     dtype=dtype)

        zeta_var_unc = tf.Variable(backward(zeta_var * np.ones((num_points, out_dim))),
                                   dtype=self.dtype)
        self.zeta_var = forward(zeta_var_unc)
        self.zeta_std = tf.sqrt(self.zeta_var)

        self.kern = MATERN32(in_dim, gp_var, gp_len, dtype=dtype)

        kernel_matrix = self.kern.K(self.zeta_pos)
        self.cholesky = cast_cholesky(kernel_matrix, jitter=1e-8)

    def predict(self, Xnew):
        """Predict mean and variance at position Xnew."""
        Kmn = self.kern.K(self.zeta_pos, Xnew)

        # Compute the projection matrix A
        A = tf.matrix_triangular_solve(self.cholesky, Kmn, lower=True)

        # compute the covariance due to the conditioning
        fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([self.out_dim, 1])
        fvar = tf.tile(tf.expand_dims(fvar, 0), shape)

        # another backsubstitution in the unwhitened case
        A = tf.matrix_triangular_solve(tf.transpose(self.cholesky), A, lower=False)

        # construct the conditional mean
        fmean = tf.matmul(A, self.zeta_mean, transpose_a=True)

        if self.zeta_std is not None:
            if self.zeta_std.get_shape().ndims == 2:
                LTA = A * tf.expand_dims(tf.transpose(self.zeta_std), 2)
            elif self.zeta_std.get_shape().ndims == 3:
                A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([self.out_dim, 1, 1]))
                LTA = tf.matmul(self.zeta_std, A_tiled, transpose_a=True)
            else:
                raise ValueError("bad dimension for q_sqrt")

            fvar += tf.reduce_sum(tf.square(LTA), 1)

        return fmean, tf.transpose(fvar)

    def prior_kl(self):
        zeta_prior = tf.contrib.distributions.MultivariateNormalTriL(
            loc=tf.zeros((self.out_dim, self.num_points), dtype=self.dtype),
            scale_tril=tf.tile(tf.expand_dims(self.cholesky, 0), [self.out_dim, 1, 1]))

        zeta_dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=tf.transpose(self.zeta_mean),
            scale_diag=tf.transpose(self.zeta_std))

        return tf.reduce_sum(tf.contrib.distributions.kl_divergence(zeta_dist, zeta_prior))
