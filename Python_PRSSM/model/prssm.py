"""
This model is a re-implementation of PR-SSM (Doerr et al., 2018)
It is mostly based on the original code and adapted to fit into the existing framework
"""

import numpy as np
import tensorflow as tf
from .tf_transform import backward, forward
from .gp_tf import MATERN32, conditional
from .base_model import BaseModel


class PRSSM(BaseModel):

    def __init__(self, dim_u, dim_y, config):
        super(PRSSM, self).__init__(dim_u, dim_y, config)

    def _build_graph(self):
        self.ind_pnt_num = self.config['ind_pnt_num']
        self.samples = self.config['samples']

        with self.graph.as_default():

            # Variables
            self.zeta_pos = tf.Variable(np.random.uniform(low=-self.config['zeta_pos'],
                                                          high=self.config['zeta_pos'],
                                                          size=(self.ind_pnt_num, self.dim_x + self.dim_u)))
            self.zeta_mean = tf.Variable(self.config['zeta_mean'] * np.random.rand(self.ind_pnt_num, self.dim_x))
            zeta_var_unc = tf.Variable(backward(self.config['zeta_var'] * np.ones((self.ind_pnt_num, self.dim_x))))
            self.zeta_var = forward(zeta_var_unc)
            var_x_unc = tf.Variable(backward(np.asarray([self.config['var_x']] * self.dim_x)))
            self.var_x = forward(var_x_unc)
            var_y_unc = tf.Variable(backward(np.asarray([self.config['var_y']] * self.dim_y)))
            self.var_y = forward(var_y_unc)
            self.kern = MATERN32(self.dim_u + self.dim_x, self.config['gp_var'], np.asarray([self.config['gp_len']] * (self.dim_x+self.dim_u)))
            self.var_dict = {'process noise': self.var_x,
                             'observation noise': self.var_y,
                             'kernel variance': self.kern.variance,
                             'kernel lengthscales': self.kern.lengthscales,
                             'IP pos': self.zeta_pos,
                             'IP mean': self.zeta_mean,
                             'IP var': self.zeta_var}

    def _model(self):
        sample_in = self.sample_in
        sample_out = self.sample_out
        with self.graph.as_default():
            # Loop init
            x_array = tf.TensorArray(dtype=tf.float64, size=self.seq_len_tf, clear_after_read=False)
            x_0 = self._recog_model(sample_in, sample_out)
            x_array = x_array.write(0, x_0)

            u_array = tf.TensorArray(dtype=tf.float64, size=self.seq_len_tf,
                                     clear_after_read=False)
            u_dub = tf.transpose(sample_in, perm=[1, 0, 2])
            u_dub = tf.tile(tf.expand_dims(u_dub, axis=2), [1, 1, self.samples, 1])
            u_array = u_array.unstack(u_dub)

            # Loop
            u_final, x_final, t_final = tf.while_loop(
                lambda u, x, t: t < self.seq_len_tf - 1,
                self._loop_body,
                [u_array, x_array, 0], parallel_iterations=1)

            x_final = tf.transpose(x_final.stack(), perm=[1, 0, 2, 3])
            y_final = x_final[:, :, :, :self.dim_y]

            lik_seq_length_tf = tf.cast(tf.cast(self.seq_len_tf, tf.float32) * self.config['lik_seq_length_factor'],
                                        tf.int32)
            y_final_stat = y_final[:, -lik_seq_length_tf:, :, :]
            sample_out_stat = sample_out[:, -lik_seq_length_tf:, :]

            # Likelihood
            var_y_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.var_y, 0), 0), 0)
            var_full = tf.tile(var_y_exp, [self.batch_tf, lik_seq_length_tf, self.samples, 1])
            y_dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=y_final_stat, scale_diag=tf.sqrt(var_full))
            obs = tf.tile(tf.expand_dims(sample_out_stat, 2), [1, 1, self.samples, 1])
            log_probs = y_dist.log_prob(obs)

            # KL-Regularizer
            k_prior = self.kern.K(self.zeta_pos, self.zeta_pos)
            scale_prior = tf.tile(tf.expand_dims(tf.cholesky(k_prior), 0), [self.dim_x, 1, 1])
            zeta_prior = tf.contrib.distributions.MultivariateNormalTriL(
                loc=tf.zeros((self.dim_x, self.ind_pnt_num), dtype=tf.float64), scale_tril=scale_prior)
            zeta_dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.transpose(self.zeta_mean), scale_diag=tf.sqrt(tf.transpose(self.zeta_var)))
            kl_div = tf.contrib.distributions.kl_divergence(zeta_dist, zeta_prior)
            kl_div = tf.reshape(kl_div , (1, 1, tf.shape(kl_div)[0]))

            # Statistics
            self.pred_mean, pred_var = tf.nn.moments(y_final, axes=[2])
            self.pred_var = tf.add(pred_var, self.var_y)
            self.latent_mean, latent_var = tf.nn.moments(x_final, axes=[2])
            self.latent_var = tf.add(latent_var, self.var_x)

            # Statistics
            self.mse = tf.losses.mean_squared_error(labels=self.sample_out, predictions=self.pred_mean)
            self.sde = tf.abs(self.pred_mean - self.sample_out) / tf.sqrt(self.pred_var)

            # Training
            log_lik = tf.reduce_sum(log_probs)
            kl_reg = tf.reduce_sum(kl_div)
            elbo = log_lik - kl_reg
            self.loss = tf.negative(elbo)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
            self.train = optimizer.minimize(self.loss, colocate_gradients_with_ops=True)
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    def _loop_body(self, u, x, t):
        # read input
        u_t = u.read(t)
        x_t = x.read(t)
        in_t = tf.concat((x_t, u_t), axis=2)

        # gp
        in_t_reshape = tf.reshape(in_t, (self.batch_tf * self.samples, self.dim_u + self.dim_x))

        fmean, fvar = conditional(in_t_reshape, self.zeta_pos, self.kern,
                                  self.zeta_mean, tf.sqrt(self.zeta_var))

        fmean = tf.reshape(fmean, (self.batch_tf, self.samples, self.dim_x))
        fvar = tf.reshape(fvar, (self.batch_tf, self.samples, self.dim_x))
        fmean = tf.add(fmean, in_t[:, :, :self.dim_x])
        fvar = fvar + self.var_x

        # sample
        eps = tf.tile(tf.random_normal((self.batch_tf, self.samples, 1), dtype=tf.float64), [1, 1, self.dim_x])
        x_t = tf.add(fmean, tf.multiply(eps, tf.sqrt(fvar+1e-8)))
        x_out = x.write(t + 1, x_t)

        return u, x_out, t + 1

    def _recog_model(self, sample_in, sample_out):
        recog_len = self.config['recog_len']
        x_0 = None

        if 'recog_model' in self.config:
            recog = self.config['recog_model']
        else:
            recog = 'zeros'

        if recog == 'zeros':
            x_0 = tf.zeros((self.batch_tf, self.dim_x), dtype=tf.float64)
            x_0 = tf.tile(tf.expand_dims(x_0, axis=1), [1, self.samples, 1])

        if recog == 'output':
            x_0 = sample_out[:, 0, :]
            pad = tf.zeros((self.batch_tf, self.dim_x - self.dim_y), dtype=tf.float64)
            x_0 = tf.concat((x_0, pad), axis=1)
            x_0 = tf.tile(tf.expand_dims(x_0, axis=1), [1, self.samples, 1])

        if recog == 'conv':
            sample_uy = sample_in
            sample_uy = sample_uy[:, :recog_len, :]
            sample_uy = tf.cast(sample_uy, tf.float32)

            layer1 = tf.layers.conv1d(sample_uy, 5, 3, activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling1d(layer1, 2, 2)
            out1 = tf.reshape(pool1, [self.batch_tf, 35])
            dense2 = tf.layers.dense(out1, self.dim_x)
            dense2 = tf.cast(dense2, tf.float64)

            x_0 = tf.expand_dims(dense2, axis=1) + tf.zeros((self.batch_tf, self.samples, self.dim_x), dtype=tf.float64)

        if recog == 'rnn':
            sample_uy = sample_in
            sample_uy = sample_uy[:, :recog_len, :]
            rnn_recog = tf.nn.rnn_cell.LSTMCell(60, state_is_tuple=False)
            initial_state = rnn_recog.zero_state(self.batch_tf, dtype=tf.float64)
            _, recog_state = tf.nn.dynamic_rnn(rnn_recog,
                                               tf.reverse(sample_uy, axis=[1]),
                                               initial_state=initial_state,
                                               dtype=tf.float64,
                                               scope='RNN_recog')
            dense = tf.layers.dense(inputs=recog_state, units=self.dim_x)
            x_0 = tf.tile(tf.expand_dims(dense, axis=1), [1, self.samples, 1])

        assert x_0 is not None, 'invalid config for recognition model'
        return x_0
