import os
import numpy as np
import tensorflow as tf
import scipy.special
import scipy.io

class Outputs:

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.out_matdir = out_dir + 'matfiles/'
        self.ds = None
        self.model = None
        self.model_path = None
        self.trainer = None
        self.last_rmse = None
        self.predict_size = None
        self.dim_x = None
        if not os.path.exists(self.out_matdir):
            os.makedirs(self.out_matdir)

    def set_ds(self, ds):
        self.ds = ds

    def set_model(self, model, predict_size=None, dim_x=None):
        self.model = model
        self.model_path = self.out_dir + 'best.ckpt'
        self.predict_size = predict_size
        self.dim_x = dim_x

    def set_trainer(self, trainer):
        self.trainer = trainer

    def get_last_rmse(self):
        return self.last_rmse

    def create_all(self):
        assert self.model is not None
        assert self.ds is not None
        with self.model.graph.as_default():
            with tf.Session() as sess:
                self.model.saver.restore(sess, self.model_path)
                print("Generating outputs...")
                self._create_all(sess)

    def _create_all(self, sess):
        self.training_stats()
        self.prediction(sess)

    def training_stats(self):
        if self.trainer is not None:
            print("  training stats")
            scipy.io.savemat(self.out_matdir + 'training_loss.mat',
                             {'train_loss': self.trainer.train_all, 'test_loss': self.trainer.test_all})

    def prediction(self, sess):
        print("  Prediction")
        model = self.model
        ds = self.ds
        gpus = model.gpus
        num_gpus = len(gpus)

        mean_in = ds.mean_in
        mean_out = ds.mean_out
        std_in = ds.std_in
        std_out = ds.std_out

        # Train
        num_experiments_train = ds.train_in.shape[0]
        train_out_shape = ds.train_out.shape
        train_x_shape = list(train_out_shape)
        train_x_shape[2] = self.dim_x
        pred_train = np.empty(train_out_shape)
        var_train = np.empty(train_out_shape)
        pred_x_train = np.empty(train_x_shape)
        var_x_train = np.empty(train_x_shape)

        for i in range(int(num_experiments_train/num_gpus)):
            model.load_ds(sess, ds.train_in[i:(i+1)], ds.train_out[i:(i+1)])
            pred_train[i], var_train[i], pred_x_train[i], var_x_train[i] = \
                sess.run((model.pred_mean, model.pred_var, model.latent_mean, model.latent_var))

        var_x_train = np.sqrt(var_x_train)
        var_train = np.sqrt(var_train)
        gt_train = ds.train_out
        in_train = ds.train_in
        pos_train = ds.train_pos

        # Test
        num_experiments_test = ds.test_in.shape[0]
        test_out_shape = ds.test_out.shape
        test_x_shape = list(test_out_shape)
        test_x_shape[2] = self.dim_x

        pred_test = np.empty(test_out_shape)
        var_test = np.empty(test_out_shape)
        pred_x_test = np.empty(test_x_shape)
        var_x_test = np.empty(test_x_shape)

        for i in range(int(num_experiments_test/num_gpus)):
            model.load_ds(sess, ds.test_in[i:(i+1)], ds.test_out[i:(i+1)])
            pred_test[i], var_test[i], pred_x_test[i], var_x_test[i] = \
                sess.run((model.pred_mean, model.pred_var, model.latent_mean, model.latent_var))

        var_x_test = np.sqrt(var_x_test)
        var_test = np.sqrt(var_test)
        gt_test = ds.test_out
        in_test = ds.test_in
        pos_test = ds.test_pos

        scipy.io.savemat(self.out_matdir + 'predict_train_n_test.mat',
                         {'gp_mean_train': pred_train, 'gp_var_train': var_train, 'gt_train': gt_train,
                          'latent_mean_train': pred_x_train, 'latent_var_train': var_x_train, 'pos_train': pos_train,
                          'in_train': in_train, 'gp_mean_test': pred_test, 'gp_var_test': var_test, 'gt_test': gt_test,
                          'latent_mean_test': pred_x_test, 'latent_var_test': var_x_test, 'pos_test': pos_test,
                          'in_test': in_test, 'mean_in': mean_in, 'mean_out': mean_out, 'std_in': std_in,
                          'std_out': std_out})
