import os
import time
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
        self.m_plus = 0.9
        self.m_minus = 1.0 - self.m_plus
        self.lmbd = 0.5
        self.learning_rate = 0.0002

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

        # digit caps layer: returns [batch_size, 10, 16]
        l3 = self.digit_caps_layer(l2, n_dim=16, n_classes=self.y_dim, n_routing=n_routing)

        # masking layer: returns [batch_size, 10, 16]
        l4 = self.masking_layer(l3, self.inputs_y)

        # reconstruction layer: returns [batch_size, 784]
        l5 = self.reconstruction_layer(l4)

        # loss
        self.margin_loss, self.recon_loss, self.total_loss = self.model_loss(l3, self.inputs_y, l5, self.inputs_x)

        # optimizer
        self.train_opt = self.model_opt(self.total_loss)
        return

    def model_loss(self, digit_caps_out, true_label, recon_out, true_inputs):
        # compute length of the digit caps layer
        epsilon = 1e-9
        pred_label = tf.sqrt(tf.reduce_sum(tf.square(digit_caps_out), axis=2, keep_dims=True) + epsilon)
        pred_label = tf.squeeze(pred_label, axis=2)

        # compute margin loss
        maximum_l = tf.square(tf.maximum(0., self.m_plus - pred_label))
        maximum_r = tf.square(tf.maximum(0., pred_label - self.m_minus))
        maximum = true_label * maximum_l + self.lmbd * (1.0 - true_label) * maximum_r
        margin_loss = tf.reduce_mean(tf.reduce_sum(maximum, axis=1))

        # compute reconstruction loss
        reshaped = tf.reshape(true_inputs, shape=(-1, 784))
        reconstruction_loss = tf.reduce_sum(tf.square(recon_out - reshaped))

        # total loss
        total_loss = margin_loss + reconstruction_loss

        return margin_loss, reconstruction_loss, total_loss

    def model_opt(self, total_loss):
        t_vars = tf.trainable_variables()

        beta1 = 0.5
        train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(total_loss, var_list=t_vars)

        return train_opt

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
        :return:[batch_size, 10, 16]
        """
        # get inputs shape sizes as int
        inputs_shape = inputs.get_shape().as_list()
        n_capsules_i = inputs_shape[1]
        n_dim_i = inputs_shape[2]

        # set current capsule size and vector size
        n_capsules_j = n_classes
        n_dim_j = n_dim

        # get batch size as Tensor in order to apply tf.tile()
        batch_size = tf.shape(inputs)[0]

        with tf.variable_scope('DigitCaps_layer'):

            # u_hat = W x u
            with tf.variable_scope('transformation'):
                # create transform matrix W: [1, 1152, 10, 8, 16], trainable
                stddev = 0.02
                w = tf.get_variable(name='W', shape=[1, n_capsules_i, n_capsules_j, n_dim_i, n_dim_j], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=stddev))

                # prepare inputs and W for transform matrix multiplication
                # inputs: [batch_size, 1152, 8] => [batch_size, 1152, 1, 1, 8]
                #         => [batch_size, 1152 (n_capsules_i), 10 (n_capsules_j), 1, 8 (n_dim_i)]
                inputs_reshaped = tf.expand_dims(inputs, axis=2)
                inputs_reshaped = tf.expand_dims(inputs_reshaped, axis=2)
                inputs_reshaped_tiled = tf.tile(inputs_reshaped, multiples=[1, 1, n_capsules_j, 1, 1])

                # W: [1, 1152, 10, 8, 16] => [batch_size, 1152, 10, 8, 16]
                w_tiled = tf.tile(w, multiples=[batch_size, 1, 1, 1, 1])

                # compute u_hat
                # tf.matmul() will do matrix multiplication on last two dimension
                # last 2 dims: [1, 8] x [8, 16] => [1, 16]
                # final output u_hat: [batch_size, 1152, 10, 1, 16]
                u_hat = tf.matmul(inputs_reshaped_tiled, w_tiled)

            with tf.variable_scope('routing'):
                # create coupling coefficients b_ij: [batch_size, 1152, 10, 1, 1], not trainable
                b_ij = tf.zeros([batch_size, n_capsules_i, n_capsules_j, 1, 1], dtype=tf.float32)

                for i in range(n_routing):
                    # weighted sum: s_j = sigma_i {c_ij x u_hat}
                    # c_ij: [batch_size, 1152, 10, 1, 1]
                    c_ij = tf.nn.softmax(b_ij, dim=2)

                    # s_j: [batch_size, 1152, 10, 1, 16] => [batch_size, 1, 10, 1, 16]
                    s_j = tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=1, keep_dims=True)

                    # squash
                    # v_j: [batch_size, 1, 10, 1, 16]
                    v_j = squash(s_j, norm_axis=2)

                    # update b_ij
                    # u_hat x v_j: [batch_size, 1152, 10, 1, 16]
                    # b_ij: [batch_size, 1152, 10, 1, 1]
                    b_ij = tf.reduce_sum(tf.multiply(u_hat, v_j), axis=4, keep_dims=True)

        # reduce dimension
        v_j = tf.squeeze(v_j, axis=3)
        v_j = tf.squeeze(v_j, axis=1)
        return v_j

    @staticmethod
    def masking_layer(inputs, class_label):
        """
        :param inputs: [batch_size, 10, 16]
        :param class_label: [batch_size, 10]
        :return: [batch_size, 10, 16]
        """
        with tf.variable_scope('masking'):
            label = tf.expand_dims(class_label, axis=2)
            masked = tf.multiply(inputs, label)

        return masked

    @staticmethod
    def reconstruction_layer(inputs):
        """
        :param inputs: [batch_size, 10, 16]
        :return: [batch_size, 784]
        """
        inputs_shape = inputs.get_shape().as_list()
        with tf.variable_scope('reconstruction'):
            reshaped = tf.reshape(inputs, shape=[-1, inputs_shape[1] * inputs_shape[2]])
            fc1 = tf.layers.dense(reshaped, units=512, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=1024, activation=tf.nn.relu)
            fc3 = tf.layers.dense(fc2, units=784, activation=tf.sigmoid)

        return fc3


def train(net, epochs, batch_size, print_every=50):
    # mnist datset loader
    mnist_dir = 'mnist'
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    # val_block_size = 10
    # val_size = val_block_size * val_block_size
    steps = 0
    losses = []

    start_time = time.time()

    with tf.Session() as sess:
        # reset tensorflow variables
        sess.run(tf.global_variables_initializer())

        # start training
        for e in range(epochs):
            for ii in range(mnist.train.num_examples // batch_size):
                # get training data
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # reshape input
                batch_x = np.reshape(batch_x, (-1, 28, 28, 1))

                fd = {
                    net.inputs_x: batch_x,
                    net.inputs_y: batch_y
                }

                # Run optimizers
                _ = sess.run(net.train_opt, feed_dict=fd)

                # print losses
                if steps % print_every == 0:
                    margin_loss = net.margin_loss.eval(fd)
                    recon_loss = net.recon_loss.eval(fd)
                    total_loss = net.total_loss.eval(fd)

                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Margin Loss: {:.4f}...".format(margin_loss),
                          "Reconstruction Loss: {:.4f}...".format(recon_loss),
                          "Total Loss: {:.4f}...".format(total_loss))
                    losses.append((margin_loss, recon_loss, total_loss))

                steps += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed: {:05.03f} s'.format(elapsed_time))

    return


def main():
    epochs = 50
    batch_size = 128
    net = CapsNet()

    train(net, epochs, batch_size)
    return


if __name__ == '__main__':
    main()
