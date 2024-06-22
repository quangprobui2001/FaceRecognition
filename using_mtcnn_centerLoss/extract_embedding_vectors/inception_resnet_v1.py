# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as layers

#inception_resnet_A

def block35(net, scale=1.0, activation='relu', scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = layers.Conv2D(32, 1, activation=activation, name='Conv2d_1x1')(net)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = layers.Conv2D(32, 1, activation=activation, name='Conv2d_0a_1x1')(net)
            tower_conv1_1 = layers.Conv2D(32, 3, activation=activation, name='Conv2d_0b_3x3')(tower_conv1_0)
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = layers.Conv2D(32, 1, activation=activation, name='Conv2d_0a_1x1')(net)
            tower_conv2_1 = layers.Conv2D(32, 3, activation=activation, name='Conv2d_0b_3x3')(tower_conv2_0)
            tower_conv2_2 = layers.Conv2D(32, 3, activation=activation, name='Conv2d_0c_3x3')(tower_conv2_1)
        mixed = layers.Concatenate(axis=3)([tower_conv, tower_conv1_1, tower_conv2_2])
        up = layers.Conv2D(net.get_shape()[3], 1, activation='linear', name='Conv2d_1x1')(mixed)
        net += scale * up
        if activation:
            net = tf.nn.relu(net)
    return net

def block17(net, scale=1.0, activation='relu',scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = layers.Conv2D(128, 1, activation=activation, name='Conv2d_1x1')(net)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = layers.Conv2D(128, 1, activation=activation, name='Conv2d_0a_1x1')(net)
            tower_conv1_1 = layers.Conv2D(160, [1, 7], activation=activation, name='Conv2d_0b_1x7')(tower_conv1_0)
            tower_conv1_2 = layers.Conv2D(192, [7, 1], activation=activation, name='Conv2d_0c_7x1')(tower_conv1_1)
        mixed = layers.Concatenate(axis=3)([tower_conv, tower_conv1_2])
        up = layers.Conv2D(net.get_shape()[3], 1, activation='linear', name='Conv2d_1x1')(mixed)
        net += scale * up
        if activation:
            net = tf.nn.relu(net)
    return net

def block8(net, scale=1.0, activation='relu', scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = layers.Conv2D(192, 1, activation=activation, name='Conv2d_1x1')(net)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = layers.Conv2D(192, 1, activation=activation, name='Conv2d_0a_1x1')(net)
            tower_conv1_1 = layers.Conv2D(224, [1, 3], activation=activation, name='Conv2d_0b_1x3')(tower_conv1_0)
            tower_conv1_2 = layers.Conv2D(256, [3, 1], activation=activation, name='Conv2d_0c_3x1')(tower_conv1_1)
        mixed = layers.Concatenate(axis=3)([tower_conv, tower_conv1_2])
        up = layers.Conv2D(net.get_shape()[3], 1, activation='linear', name='Conv2d_1x1')(mixed)
        net += scale * up
        if activation:
            net = tf.nn.relu(net)
    return net

def reduction_a(net, k, l, m, n):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope('Branch_0'):
        tower_conv = layers.Conv2D(n, 3, strides=2, activation='relu', padding='valid', name='Conv2d_1a_3x3')(net)
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = layers.Conv2D(k, 1, activation='relu', name='Conv2d_0a_1x1')(net)
        tower_conv1_1 = layers.Conv2D(l, 3, activation='relu', name='Conv2d_0b_3x3')(tower_conv1_0)
        tower_conv1_2 = layers.Conv2D(m, 3, strides=2, activation='relu', padding='valid', name='Conv2d_1a_3x3')(tower_conv1_1)
    with tf.variable_scope('Branch_2'):
        tower_pool = layers.MaxPooling2D(3, strides=2, padding='valid', name='MaxPool_1a_3x3')(net)
    net = layers.Concatenate(axis=3)([tower_conv, tower_conv1_2, tower_pool])
    return net

def reduction_b(net):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope('Branch_0'):
        tower_conv = layers.Conv2D(256, 1, activation='relu', name='Conv2d_0a_1x1')(net)
        tower_conv_1 = layers.Conv2D(384, 3, strides=2, activation='relu', padding='valid', name='Conv2d_1a_3x3')(tower_conv)
    with tf.variable_scope('Branch_1'):
        tower_conv1 = layers.Conv2D(256, 1, activation='relu', name='Conv2d_0a_1x1')(net)
        tower_conv1_1 = layers.Conv2D(256, 3, strides=2, activation='relu', padding='valid', name='Conv2d_1a_3x3')(tower_conv1)
    with tf.variable_scope('Branch_2'):
        tower_conv2 = layers.Conv2D(256, 1, activation='relu', name='Conv2d_0a_1x1')(net)
        tower_conv2_1 = layers.Conv2D(256, 3, activation='relu', name='Conv2d_0b_3x3')(tower_conv2)
        tower_conv2_2 = layers.Conv2D(256, 3, strides=2, activation='relu', padding='valid', name='Conv2d_1a_3x3')(tower_conv2_1)
    with tf.variable_scope('Branch_3'):
        tower_pool = layers.MaxPooling2D(3, strides=2, padding='valid', name='MaxPool_1a_3x3')(net)
    net = layers.Concatenate(axis=3)([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool])
    return net


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      bottleneck_layer_size: the size of the bottleneck layer for R-FCN.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.
    """
    end_points = {}
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with tf.variable_scope('Stem'):
            net = layers.Conv2D(32, 3, strides=2, activation='relu', padding='valid', name='Conv2d_1a_3x3')(inputs)
            net = layers.Conv2D(32, 3, activation='relu', name='Conv2d_2a_3x3')(net)
            net = layers.Conv2D(64, 3, padding='valid', name='Conv2d_2b_3x3')(net)
            with tf.variable_scope('Branch_0'):
                tower_conv = layers.Conv2D(96, 3, strides=2, padding='valid', activation='relu', name='Conv2d_3b_3x3')(net)
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = layers.Conv2D(64, 1, activation='relu', name='Conv2d_0a_1x1')(net)
                tower_conv1_1 = layers.Conv2D(96, 3, padding='valid', activation='relu', name='Conv2d_0b_3x3')(tower_conv1_0)
                tower_conv1_2 = layers.Conv2D(96, 3, strides=2, padding='valid', activation='relu', name='Conv2d_1a_3x3')(tower_conv1_1)
            with tf.variable_scope('Branch_2'):
                tower_pool = layers.MaxPooling2D(3, strides=2, padding='valid', name='MaxPool_1a_3x3')(net)
            net = layers.Concatenate(axis=3)([tower_conv, tower_conv1_2, tower_pool])




def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with layers.arg_scope(
        [layers.conv2d, layers.fully_connected],
        weights_regularizer=layers.l2_regularizer(weight_decay),
        weights_initializer=layers.variance_scaling_initializer(),
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
                                    dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


