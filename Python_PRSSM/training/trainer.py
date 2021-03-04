import numpy as np
import tensorflow as tf

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


class Trainer:

    def __init__(self, model, model_dir):
        self.model = model
        self.model_dir = model_dir
        self.out_dir = model.config['out_dir']
        self.train_all = []
        self.test_all = []

    def train(self, ds, epochs, retrain=False, test_data=False):
        print('\nTraining...\n')
        model = self.model
        with model.graph.as_default():
            config = tf.ConfigProto(device_count={'GPU': len(model.gpus)}, gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True,
                                    log_device_placement=True)
            # Pararellizing ops (default=2)
            # config.intra_op_parallelism_threads = 5
            # Executing ops in parallel (default=5)
            # config.inter_op_parallelism_threads = 10
            with tf.Session(config=config) as sess:

                if retrain:
                    model.saver.restore(sess, self.model_dir + 'model.ckpt')
                else:
                    sess.run(model.init)

                lowest_train = float('inf')
                for epoch in tqdm(range(epochs)):

                    # Train
                    model.load_ds(sess, ds.train_in_batch, ds.train_out_batch)
                    train_loss = model.run(sess, (model.train, model.loss))
                    train_loss = np.mean(train_loss[1])
                    self.train_all.append(train_loss)

                    # Test
                    if test_data:
                        model.load_ds(sess, ds.test_in_batch, ds.test_out_batch)
                        test_loss = model.run(sess, model.loss)
                        test_loss = np.mean(test_loss)
                        self.test_all.append(test_loss)
                    else:
                        test_loss = float('nan')

                    # Output
                    print('[{epoch:04}]: Train {train}, Test {test}'.format(
                        epoch=epoch, train=train_loss, test=test_loss))

                    # Save Best
                    if train_loss < lowest_train:
                        model.saver.save(sess, self.out_dir + '/best.ckpt')
                        lowest_train = train_loss

                # Save Last
                model.saver.save(sess, self.out_dir + '/model.ckpt')