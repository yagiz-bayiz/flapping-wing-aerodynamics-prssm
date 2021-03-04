import sys
import numpy as np
import tensorflow as tf


class BaseModel:
    def __init__(self, dim_u, dim_y, config, dtype=tf.float64):
        self.dim_u = dim_u
        self.dim_y = dim_y
        self.config = config
        self.dtype = dtype
        self.graph = tf.Graph()
        self._build_ds_pipeline()
        self._build_graph()
        self._model()

    def _build_ds_pipeline(self):
        self.dim_x = self.config['dim_x']
        self.gpus = self.config['gpus']

        with self.graph.as_default():
            self.data_in = tf.placeholder(self.dtype, shape=[None, None, self.dim_u])  # [ds_size, seq_len, dim]
            self.data_out = tf.placeholder(self.dtype, shape=[None, None, self.dim_y])  # [ds_size, seq_len, dim]
            self.repeats = tf.placeholder(tf.int64)
            self.condition = tf.placeholder(tf.bool)
            dataset = tf.data.Dataset.from_tensor_slices((self.data_in, self.data_out))
            dataset = dataset.repeat(self.repeats).shuffle(self.config['shuffle'])
            dataset = dataset.batch(self.config['batch_size'])
            dataset = dataset.prefetch(buffer_size=1)
            self.dataset_iterator = dataset.make_initializable_iterator()
            self.sample_in, self.sample_out = self.dataset_iterator.get_next()  # [batch_size, seq_len, dim]
            self.totalbatch_tf = tf.shape(self.sample_in)[0]  # size of current batch
            self.seq_len_tf = tf.shape(self.sample_in)[1]  # seq_len of current batch
            self.batch_tf = tf.div(self.totalbatch_tf, len(self.gpus))

    def _build_graph(self):
        pass

    def _model(self):
        pass

    def load_ds(self, sess, data_in, data_out, repeats=1):
        sess.run(self.dataset_iterator.initializer,
                 feed_dict={self.data_in: data_in,
                            self.data_out: data_out,
                            self.repeats: repeats})

    @staticmethod
    def make_parallel(fn, gpus, num_outputs, **kwargs):
        print(gpus)
        num_gpus = len(gpus)
        in_splits = {}
        for k, v in kwargs.items():
            in_splits[k] = tf.split(v, num_gpus)

        out_ = {}
        out_return = [[]]*num_outputs
        for i in range(num_gpus):
            gpu = gpus[i]
            with tf.device(gpu):
                with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                    out_[i] = fn(**{k: v[i] for k, v in in_splits.items()})

        for k in range(num_outputs):
            out_return[k] = tf.concat([out_[i][k] for i in range(num_gpus)], axis=0)
        return [x for x in out_return]

    @staticmethod
    def run(sess, tensors, feed_dict=None, show_progress=False, res_all=None):
        while True:
            try:
                res = sess.run(tensors, feed_dict=feed_dict)
                if show_progress:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                if not isinstance(res, tuple):
                    res = (res,)
                if res_all is None:
                    res_all = [i for i in res]
                    for i in range(len(res)):
                        if res_all[i] is not None:
                            res_all[i] = np.atleast_1d(res_all[i])
                else:
                    for i, item in enumerate(res):
                        if item is not None:
                            item = np.atleast_1d(item)
                            res_all[i] = np.concatenate(
                                (res_all[i], item), axis=0)
            except tf.errors.OutOfRangeError:
                break
        if show_progress:
            print()
        return res_all
