"""
RNN model for crop classification
1: citrus
0: other
"""
# -*- coding: utf-8 -*-
# test should be on python 3, tensofflow should use GPU backend

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, cudnn_rnn
from os.path import join
import shutil
import glob


class LSTModel:
    def __init__(
            self,
            model_dir="./model_dir/HN-04",  #
            file="data2/citrus/0101_1231_17_1_CiOt_L_REG_TRAIN_18.npz",     #
            *,
            init_learning_rate=0.0001,  # 0.03
            training_steps=1000,
            batch_size=256,
            num_input=10,   #
            timesteps=365,  #
            num_classes=2,
            dropout=0.4,  #
            num_hidden=180,  # 320 -> 192
            num_layers=1,   # 1 -> 2
            display_step=5,
            decay=0.96,     # 0.93
    ):
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            os.makedirs(model_dir)
        else:
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.file = file
        self.init_learning_rate = init_learning_rate
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.num_input = num_input
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.display_step = display_step
        self.decay = decay
        features_test, labels_test, features_train, labels_train = self._load_data()
        self.features_train = features_train
        self.labels_train = labels_train
        self.features_test = features_test
        self.labels_test = labels_test

    def _data_norm(self, in_features):
        # input features shaoe [num_pixel, timestep, num_bands]
        # normalize every band: (x-miu)/max, [-0.5, 0.5]
        norm_mat = np.full(in_features.shape, 1e-4)
        norm_mat[:, :, 0] = 1/360.
        norm_mat[:, :, 1] = 1/8000.
        norm_mat[:, :, 2] = 1/90.
        new_features = norm_mat * in_features - 0.5

        assert new_features.shape == in_features.shape
        return new_features

    def _load_data(self):
        print("Initializing data")
        data = np.load(self.file)
        data_feat = data["features"]
        data_lab = data["labels"]

        # shuffle train data sets
        pos_idx = np.where(data_lab == 1)[0]
        train_idx_pos = pos_idx[: int(len(pos_idx) * 0.7)]
        neg_idx = np.where(data_lab == 0)[0]
        train_idx_neg = neg_idx[: int(len(neg_idx) * 0.7)]
        train_idx = list(train_idx_neg) + list(train_idx_pos)
        test_idx = np.delete(range(len(data_lab)), np.array(train_idx))

        # train_idx = random.sample(range(len(data_lab)), int(len(data_lab) * 0.8))

        random.shuffle(train_idx)
        random.shuffle(test_idx)

        # data reshape and normalization
        data_feat = data_feat.reshape(
            (-1, self.num_input, self.timesteps))
        data_feat = data_feat.transpose(0, 2, 1)
        data_feat = self._data_norm(data_feat)

        features_train = data_feat[train_idx, :, :]
        labels_train = data_lab[train_idx]
        labels_train = tf.one_hot(labels_train, self.num_classes)

        # test_idx = list(test_idx)
        features_test = data_feat[test_idx, :, :]
        labels_test = data_lab[test_idx]
        labels_test = tf.one_hot(labels_test, self.num_classes)

        # convert nan to 0, get rid of nan
        features_train = np.nan_to_num(features_train)
        features_test = np.nan_to_num(features_test)

        print("TRAINING TOTAL %d" % features_train.shape[0])
        print("TESTING TOTAL %d" % features_test.shape[0])
        return features_test, labels_test, features_train, labels_train

    def model_select(self, f1):
        max_idx = f1.index(max(f1))
        steps = max_idx * self.display_step
        if max_idx == 0:
            steps = 1
        model_list = os.listdir(self.model_dir)
        # print(model_list)
        print('OPTIMUM STEP ', steps)
        ch = '-' + str(steps) + '.'
        model = [m for m in model_list if ch in m]
        print("Final model files: ")
        print(model)
        # copy model file to package
        final_path = join(self.model_dir, 'final')
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.mkdir(final_path)
        for file in model:
            file = join(self.model_dir, file)
            shutil.copy(file, final_path)

        # change checkpoint parameter
        data = ''
        with open(join(self.model_dir, 'checkpoint'), 'r+') as f:
            for line in f.readlines():
                line = line.replace(
                    str(self.training_steps), str(steps)) + '\n'
                data += line
            f.close()

        with open(join(self.model_dir, 'checkpoint'), 'w') as f:
            f.writelines(data)
            f.close()

        shutil.copy(join(self.model_dir, 'checkpoint'), final_path)
        return final_path

    def RNN(self, x, drop, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        # here we use static_rnn which requires inputs to be a list of tensors
        x = tf.unstack(x, self.timesteps, 1)

        # For dynamic rnn, the require for input data should be in the shape of
        # (batch_size, timesteps, n_input) if time_major is False

        # Define a lstm cell with tensorflow
        # lstm_cell = rnn.LSTMCell(self.num_hidden, use_peepholes=True)
        # lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=(1 - drop))
        #
        # lstm_cell = cudnn_rnn.CudnnLSTM(self.num_layers, self.num_hidden,
        # dropout=drop)

        lstm_cell = rnn.LayerNormBasicLSTMCell(
            self.num_hidden,
            forget_bias=0.5,
            norm_gain=1.0,
            norm_shift=0.0,
            dropout_keep_prob=(1-drop)
        )
        # Get lstm cell output
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
        # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights["out"]) + biases["out"]

    def sess_run(self):
        print(
            '= = MODEL INFO = =', '\n'
            'model save dir: ', self.model_dir, '\n'
            'data file: ', self.file, '\n'
            'initial l_r: ', self.init_learning_rate, '\n'
            'dropout: ', self.dropout, '\n'
            'number of units: ', self.num_hidden, '\n'
            'number of layers: ', self.num_layers, '\n'
            'l_r decay rate: ', self.decay, '\n'
        )
        # now create computational graph
        num_batch = int(self.features_train.shape[0] / self.batch_size)

        # tf Graph input
        X = tf.placeholder(
            "float", [None, self.timesteps, self.num_input], name="input_x")
        Y = tf.placeholder("float", [None, self.num_classes], name="input_y")
        DROP_PROB = tf.placeholder(tf.float32, name="dropout_prob")
        # LEARN_RATE = tf.placeholder(tf.float32, name="learning_rate")

        # Define weights
        weights = {"out": tf.Variable(tf.random_normal(
            [self.num_hidden, self.num_classes]), name="weights_out")}
        biases = {"out": tf.Variable(tf.random_normal(
            [self.num_classes]), name="biases_out")}

        logits = self.RNN(X, DROP_PROB, weights, biases)
        prediction = tf.nn.softmax(logits, name="prediction")

        # Define loss and optimizer
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=Y),
            name="loss_op"
        )

        # loss_op = tf.reduce_mean(
        #     tf.nn.weighted_cross_entropy_with_logits(
        #         logits=logits, targets=Y, pos_weight=1),
        #     name="loss_op"
        # )

        # Define learning rate decay
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            self.init_learning_rate,
            global_step,
            int(self.features_train.shape[0] / self.batch_size),
            self.decay,
            staircase=True,
            name='ExponentialDecay'
        )
        # Define optimizer
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=learning_rate, initial_accumulator_value=0.1,
            name="optimizer_op"
        )
        train_op = optimizer.minimize(loss_op, name="train_op")

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(
            Y, 1), name="correct_prediction")
        accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32), name="accuracy")

        # define confusion matrix
        conf_matrix = tf.confusion_matrix(
            tf.argmax(Y, 1), tf.argmax(prediction, 1), self.num_classes,
            name="confusion_matrix"
        )

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # run training and testing experiment
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True

        with tf.Session(config=conf) as sess:
            # debug
            # sess=tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan",tf_debug.has_inf_or_nan)

            # Run the initializer
            sess.run(init)
            labels_train = sess.run(self.labels_train)
            f1_list = []
            print('START TIME ', time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()))
            for step in range(1, self.training_steps + 1):

                # random shuffle data for each epoch for better training
                rand_array = np.arange(self.features_train.shape[0])
                np.random.shuffle(rand_array)
                self.features_train = self.features_train[rand_array]
                labels_train = labels_train[rand_array]

                for b in range(num_batch + 1):
                    batch_x, batch_y = (
                        self.features_train[b *
                                            self.batch_size: (b + 1) * self.batch_size],
                        labels_train[b *
                                     self.batch_size: (b + 1) * self.batch_size],
                    )

                    # Run optimization op (backprop)
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y,
                                                  DROP_PROB: self.dropout,
                                                  # LEARN_RATE: learning_rate
                                                  })

                if step % self.display_step == 0 or step == 1:
                    batch_loss = []
                    batch_acc = []
                    # Calculate batch loss and accuracy
                    for b in range(num_batch):
                        batch_x, batch_y = (
                            self.features_train[b *
                                                self.batch_size: (b + 1) * self.batch_size],
                            labels_train[b *
                                         self.batch_size: (b + 1) * self.batch_size],
                        )
                        loss, acc = sess.run(
                            [loss_op, accuracy],
                            feed_dict={X: batch_x, Y: batch_y, DROP_PROB: 0.0,
                                       # LEARN_RATE: learning_rate
                                       },
                        )
                        batch_loss.append(loss)
                        batch_acc.append(acc)
                    batch_loss_av = sum(batch_loss) / float(len(batch_loss))
                    batch_acc_av = sum(batch_acc) / float(len(batch_acc))
                    print(
                        "Step "
                        + str(step)
                        + ", Minibatch Loss= "
                        + "{:.4f}".format(batch_loss_av)
                        + ", Training Accuracy= "
                        + "{:.3f}".format(batch_acc_av)
                    )

                    print("Optimization Finished!")

                    # Calculate accuracy for testing data
                    test_data = self.features_test
                    test_label = sess.run(self.labels_test)

                    n = 7
                    size = int(test_data.shape[0] / n)
                    test_acc = []
                    test_loss = []
                    test_correct = []
                    conf_mm = []
                    # F1_list = []
                    for b in range(n + 1):
                        batch_x, batch_y = (
                            test_data[b * size: (b + 1) * size],  # noqa :E203
                            test_label[b * size: (b + 1) * size],  # noqa :E203
                        )
                        loss, acc, correct, conf_m = sess.run(
                            [loss_op, accuracy, correct_pred, conf_matrix],
                            feed_dict={X: batch_x, Y: batch_y, DROP_PROB: 0.0,
                                       # LEARN_RATE: learning_rate
                                       },
                        )
                        test_acc.append(acc)
                        test_loss.append(loss)
                        test_correct.extend(correct)

                        if b == 0:
                            conf_mm = conf_m
                        else:
                            conf_mm = conf_mm + conf_m

                    t_acc = sum(test_acc) / float(len(test_acc))
                    t_loss = sum(test_loss) / float(len(test_loss))

                    print("Testing Accuracy: %f" % t_acc)
                    print("Testing Loss: %f" % t_loss)
                    print("Confusion Matrix: ")
                    print(conf_mm)

                    PRE = conf_mm[1, 1] / float(np.sum(conf_mm, axis=0)[1])
                    REC = conf_mm[1, 1] / float(np.sum(conf_mm, axis=1)[1])
                    F1 = 2 * PRE * REC / float(PRE + REC)

                    print("Testing Precision: ", PRE)
                    print("Testing Recall: ", REC)
                    print("Testing F1 score: ", F1)
                    print('\n')
                    f1_list.append(F1)

                    # if not os.path.exists(self.model_dir):
                    #     os.makedirs(self.model_dir)

                    f = open(join(self.model_dir, "accuracy_loss.txt"), "a")
                    f.write('dropout = ' + str(self.dropout) +
                            ', num_hidden = ' + str(self.num_hidden) + '\n')
                    f.write("STEP " + str(step) + "\n")
                    f.write("TRAIN ACC " + str(batch_acc_av) + "\n")
                    f.write("TRAIN LOSS " + str(batch_loss_av) + "\n")
                    f.write("TEST ACC " + str(t_acc) + "\n")
                    f.write("TEST LOSS " + str(t_loss) + "\n")
                    f.write("CONFUSION MATRIX " + "\n")
                    f.write(str(conf_mm) + "\n\n")
                    f.write('PRECISION ' + str(PRE) + '\n')
                    f.write('RECALL ' + str(REC) + '\n')
                    f.write('F1-SCORE ' + str(F1) + '\n\n')
                    f.close()

                    # save predict correct and wrong data
                    np.savez(join(self.model_dir, "predict_result_" +
                                  str(step) + ".npz"), test_correct)

                    saver = tf.train.Saver()
                    saver.save(
                        sess,
                        join(self.model_dir, "LSTM_model"),
                        global_step=step,
                        write_meta_graph=True,
                    )

            model_path = self.model_select(f1_list)
            print("finish model file re-constructing")
            print('model path : {}'.format(model_path))
            meta = glob.glob(join(model_path, '*.meta'))[0]
        return join(model_path, meta)
