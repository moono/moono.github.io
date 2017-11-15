import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


'''
tf.layers.conv2d()'s default parameters
kernel_initializer: glorot_uniform_initializer ==> Xavier uniform initializer
'''

def squash(sj, norm_axis):
    epsilon = 1e-9
    sj_squared_norm = tf.reduce_sum(tf.square(sj), axis=norm_axis, keep_dims=True)
    scale = sj_squared_norm / (1.0 + sj_squared_norm) / tf.sqrt(sj_squared_norm + epsilon)
    vj = scale * sj
    return vj


class CapsNet(object):
    def __init__(self):
        # start building graph
        tf.reset_default_graph()

        # set class variables

        # we are handling MNIST dataset
        self.im_size = 28
        self.y_dim = 10
        self.inputs_x = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 1], name='inputs_x')
        self.inputs_y = tf.placeholder(tf.float32, [None, self.y_dim], name='inputs_y')

        # build architecture
        n_k = 9
        n_routing = 3

        # first convolution layer: returns [batch_size, 20, 20, 256]
        l1 = self.conv_layer(self.inputs_x, n_filter=256, n_k=n_k)

        # primary caps layer: returns [batch_size, 1152, 8]
        l2 = self.primary_caps_layer(l1, n_dim=8, n_channel=32, n_k=n_k)

        # digit caps layer: returns
        l3 = self.digit_caps_layer(l2, n_dim=16, n_classes=self.y_dim, n_routing=n_routing)

        return

    @staticmethod
    def conv_layer(inputs, n_filter, n_k):
        """
        :param inputs: [batch_size, 28, 28, 1]
        :param n_filter: 256
        :param n_k: 9
        :return: [batch_size, 20, 20, 256]
        """
        with tf.variable_scope('Conv_layer'):
            # [batch_size, 28, 28, 1] => [batch_size, 20, 20, 256]
            layer = tf.layers.conv2d(inputs, filters=n_filter, kernel_size=n_k, strides=1, padding='valid')
            layer = tf.nn.relu(layer)
        return layer

    @staticmethod
    def primary_caps_layer(inputs, n_dim, n_channel, n_k):
        """
        :param inputs: [batch_size, 20, 20, 256]
        :param n_dim: 8
        :param n_channel: 32
        :param n_k: 9
        :return: [batch_size, 1152, 8]
        """
        with tf.variable_scope('PrimaryCaps_layer'):
            # [batch_size, 20, 20, 256] => [batch_size, 6, 6, 256]
            layer = tf.layers.conv2d(inputs, filters=n_dim * n_channel, kernel_size=n_k, strides=2, padding='valid')
            layer = tf.nn.relu(layer)
            l_shape = layer.get_shape().as_list()

            # [batch_size, 6, 6, 256] => [batch_size, 6 * 6 * 32, 8] => [batch_size, 1152, 8]
            # there are 1152 (6 * 6 * 32) capsules (8-D)
            layer = tf.reshape(layer, shape=[-1, l_shape[1] * l_shape[2] * n_channel, n_dim])
            layer = squash(layer, norm_axis=2)
        return layer

    @staticmethod
    def digit_caps_layer(inputs, n_dim, n_classes, n_routing):
        """
        :param inputs: [batch_size, 1152, 8]
        :param n_dim: 16
        :param n_classes: 10
        :param n_routing: 3
        :return:
        """
        batch_size = tf.shape(inputs)[0]
        inputs_shape = inputs.get_shape().as_list()
        n_capsules_i = inputs_shape[1]
        n_capsules_j = n_classes
        n_dim_i = inputs_shape[2]
        n_dim_j = n_dim
        with tf.variable_scope('DigitCaps_layer'):
            stddev = 0.02

            # create transform matrix W: [1, 1152, 10, 8, 16], trainable
            w = tf.get_variable(name='W', shape=[1, n_capsules_i, n_capsules_j, n_dim_i, n_dim_j], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))

            # prepare inputs for matrix multiplication: [batch_size, 1152, 8] => [batch_size, 1152, 1, 1, 8]
            inputs_reshaped = tf.expand_dims(inputs, axis=2)
            inputs_reshaped = tf.expand_dims(inputs_reshaped, axis=2)

            # Replicate num_capsule dimension to prepare being multiplied by W
            # : [batch_size, input_num_capsule, num_capsule, 1, input_dim_vector]
            # : [batch_size, 1152, 1, 1, 8] => [batch_size, 1152, 10, 1, 8]
            inputs_reshaped_tiled = tf.tile(inputs_reshaped, multiples=[1, 1, n_capsules_j, 1, 1])

            # replicate matrix w as well: [1, 1152, 10, 8, 16] => [batch_size, 1152, 10, 8, 16]
            w_tiled = tf.tile(w, multiples=[batch_size, 1, 1, 1, 1])

            # compute u_hat
            # in last 2 dims: [1, 8] x [8, 16] => [1, 16] => [batch_size, 1152, 10, 1, 16]
            u_hat = tf.matmul(inputs_reshaped_tiled, w_tiled)


            # create cupling coefficients b_ij: [1, 1152, 10, 1, 1], not trainable
            b_ij = tf.zeros([1, n_capsules_i, n_capsules_j, 1, 1], dtype=tf.float32)


        return None


def main():
    # mnist datset loader
    mnist_dir = 'mnist'
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    net = CapsNet()
    return


if __name__ == '__main__':
    main()
