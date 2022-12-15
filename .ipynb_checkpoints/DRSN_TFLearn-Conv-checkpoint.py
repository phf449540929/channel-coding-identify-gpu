#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    @File:         DRSN_TFLearn.py
    @Author:       haifeng
    @Since:        python3.9
    @Version:      V1.0
    @Date:         2021/10/28 10:40
    @Description:
-------------------------------------------------
    @Change:
        2021/10/28 10:40
-------------------------------------------------
"""

from __future__ import division, print_function, absolute_import

import datetime
import sys


import tflearn
import tensorflow.compat.v1 as tf
import pandas
import numpy

from sklearn import model_selection, preprocessing
from sklearn.metrics import classification_report
from tflearn.layers.conv import conv_2d


# Data loading
# from tflearn.datasets import cifar10

import os

from tensorflow.python.client import device_lib

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(tf.test.is_gpu_available())

tf.disable_v2_behavior()

# filter_number = int(sys.argv[1])
# filter_size = int(sys.argv[2])

filter_number = 16
filter_size = 4

is_turn = 0
turn = 0

snr = sys.argv[1]
epoch = int(sys.argv[2])
# epoch = 10
# turn = sys.argv[2]
# is_turn = 1
# file_name = "dataset-awgn-conv-" + str(snr) + "db-pre-raw.csv"
# file_name = "dataset-rayleigh-single-conv-" + str(snr) + "db-pre-raw.csv"
file_name = "dataset-rayleigh-dual-conv-" + str(snr) + "db-pre-raw.csv"
# file_name = "dataset-awgn-conv-0db-pre-raw.csv"
coding = file_name.split('-')[3]

classes = 15
shape = [32, 32, 4]
pretreatment = 3


def get_dataset(matrix):
    dataset = []
    for s in matrix:
        a = s.strip().split(' ')
        col = []
        for b in a:
            row = []
            b = float(b)
            row.append(b)
            col.append(row)
        if pretreatment == 1:
            # gfft
            col.append([0, 0])
        elif pretreatment == 2:
            # llr cal
            col.append([0])
        # elif pretreatment == 3:
        # llr raw
        # i = 96
        # for j in range(i):
        #     col.append([0])
        dataset.append(numpy.array([col]).reshape((shape[0], shape[1], shape[2])))
    return dataset


dataframe = pandas.read_csv(file_name)
X = numpy.array(get_dataset(dataframe['X']))
Y = dataframe['Y']

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(X, Y)
x_train = X
y_train = Y
encoder = preprocessing.LabelEncoder()
y_valid_class = y_valid
y_train = encoder.fit_transform(y_train)
y_valid = encoder.fit_transform(y_valid)
y_valid_num = y_valid

# X = numpy.expand_dims(X, axis=0)
# X = numpy.expand_dims(X, axis=0)
# testX = numpy.expand_dims(testX, axis=0)
# testX = numpy.expand_dims(testX, axis=

# Add noise
# X = X + numpy.random.random((50000, 32, 32, 3)) * 0.1
# testX = testX + numpy.random.random((10000, 32, 32, 3)) * 0.1

# Transform labels to one-hot format
# Y = tflearn.data_utils.to_categorical(Y, 10)
# testY = tflearn.data_utils.to_categorical(testY, 10)
y_train = tflearn.data_utils.to_categorical(y_train, classes)
y_valid = tflearn.data_utils.to_categorical(y_valid, classes)


def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2, activation='relu', batch_norm=True,
                             bias=True, weights_init='variance_scaling',
                             bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                             trainable=True, restore=True, reuse=False, scope=None,
                             name="ResidualBlock", filter_size_residual=3):
    # residual shrinkage blocks with channel-wise thresholds

    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming], reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    with vscope as scope:
        name = scope.name  # TODO

        for i in range(nb_blocks):

            identity = residual

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                residual = tflearn.batch_normalization(residual)

            residual = tflearn.activation(residual, activation)
            residual = conv_2d(residual, out_channels, filter_size_residual,
                               downsample_strides, 'same', 'linear',
                               bias, weights_init, bias_init,
                               regularizer, weight_decay, trainable,
                               restore)

            if batch_norm:
                residual = tflearn.batch_normalization(residual)
            residual = tflearn.activation(residual, activation)
            residual = conv_2d(residual, out_channels, filter_size_residual, 1, 'same',
                               'linear', bias, weights_init,
                               bias_init, regularizer, weight_decay,
                               trainable, restore)

            # get thresholds and apply thresholding
            abs_mean = tf.reduce_mean(tf.reduce_mean(tf.abs(residual), axis=2, keep_dims=True), axis=1, keep_dims=True)
            scales = tflearn.fully_connected(abs_mean, n_units=out_channels // 4, activation='linear', regularizer='L2',
                                             weight_decay=0.0001, weights_init='variance_scaling')
            scales = tflearn.batch_normalization(scales)
            scales = tflearn.activation(scales, 'relu')
            scales = tflearn.fully_connected(scales, n_units=out_channels, activation='linear', regularizer='L2',
                                             weight_decay=0.0001, weights_init='variance_scaling')
            scales = tf.expand_dims(tf.expand_dims(scales, axis=1), axis=1)
            thres = tf.multiply(abs_mean, tflearn.activations.sigmoid(scales))
            # soft thresholding
            residual = tf.multiply(tf.sign(residual), tf.maximum(tf.abs(residual) - thres, 0))

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, 1,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                if (out_channels - in_channels) % 2 == 0:
                    ch = (out_channels - in_channels) // 2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch]])
                else:
                    ch = (out_channels - in_channels) // 2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch + 1]])
                in_channels = out_channels

            residual = residual + identity

    return residual


# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()

# Build a Deep Residual Shrinkage Network with 3 blocks
'''
# Real-time data augmentation
img_aug.add_random_crop([32, 32], padding=4)

# Build a Deep Residual Shrinkage Network with 3 blocks
net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = residual_shrinkage_block(net, 1, 16)
net = residual_shrinkage_block(net, 1, 32, downsample=True)
net = residual_shrinkage_block(net, 1, 32, downsample=True)

nb_filter: `int`. The number of convolutional filters.
filter_size: `int` or `list of int`. Size of filters.

n_units: 神经元个数
'''

img_aug.add_random_crop([shape[0], shape[1]], padding=4)

net = tflearn.input_data(shape=[None, shape[0], shape[1], shape[2]],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, nb_filter=filter_number, filter_size=filter_size, regularizer='L2', weight_decay=0.0001)
net = residual_shrinkage_block(net, nb_blocks=1, out_channels=filter_number,
                               filter_size_residual=filter_size)
net = residual_shrinkage_block(net, nb_blocks=1, out_channels=filter_number * 2,
                               filter_size_residual=filter_size, downsample=True)
net = residual_shrinkage_block(net, nb_blocks=1, out_channels=filter_number * 2,
                               filter_size_residual=filter_size, downsample=True)

net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, n_units=classes, activation='softmax')
mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.1, decay_step=20000, staircase=True)
net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='model-channel-coding', max_checkpoints=10,
                    tensorboard_verbose=0, clip_gradients=0.)

model.fit(x_train, y_train, n_epoch=epoch, snapshot_epoch=False, snapshot_step=2000, show_metric=True,
          batch_size=64, shuffle=True, run_id='model-channel-coding')


def get_acc(epoch_read):
    training_acc = model.evaluate(x_train, y_train)
    validation_acc = model.evaluate(x_valid, y_valid)
    print("\n")
    print("tranining_acc = {}".format(training_acc))
    print("validation_acc = {}".format(validation_acc))

    y_pred = model.predict(x_valid)
    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0

    y_pred_num = []
    for i in range(len(y_pred)):
        for j in range(len(y_valid)):
            if all(y_pred[i] == y_valid[j]):
                y_pred_num.append(y_valid_num[j])
                break

    y_pred_class = encoder.inverse_transform(y_pred_num)
    validation_report = classification_report(y_valid_class, y_pred_class, digits=8)
    print(validation_report)

    with open("acc-{}-{}.txt".format(coding, datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')), "w",
              encoding='utf-8') as f:
#         f.write("filter_number = {}\n".format(filter_number))
#         f.write("filter_size = {}\n".format(filter_size))
        f.write("file_name = {}\n".format(file_name))
        f.write("modulate = {}\n".format('bpsk'))
        f.write("epoch = {}\n".format(epoch_read))
        if is_turn:
            f.write("turn = {}\n".format(turn))
        f.write("training_acc = {}\n".format(training_acc))
        f.write("validation_acc = {}\n".format(validation_acc))
        f.write(validation_report)


get_acc(epoch)

# n = int(epoch / 100)
# for e in range(n):
#     epoch_read = (e + 1) * 100
#     model.load("model-channel-coding-{}".format(epoch_read * 20))
#     get_acc(model, epoch_read)
