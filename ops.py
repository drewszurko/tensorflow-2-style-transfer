# MIT License
#
# Copyright (c) 2019 Drew Szurko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from absl import flags
from tensorflow import linalg
from tensorflow.python.keras import applications
from tensorflow.python.keras import models
from tensorflow.python.keras.applications.vgg19 import VGG19

FLAGS = flags.FLAGS


def get_vgg_layers(layer_names):
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = models.Model([vgg.input], outputs)
    return model


def calculate_gram_matrix(tensor):
    input_shape = tf.shape(tensor)
    result = linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def compute_loss(outputs, targets):
    return tf.add_n([
        tf.reduce_mean(tf.square(outputs[name] - targets[name]))
        for name in outputs.keys()
    ])


def get_high_frequencies(img):
    x = img[:, :, 1:, :] - img[:, :, :-1, :]
    y = img[:, 1:, :, :] - img[:, :-1, :, :]
    return x, y


def variation_loss(img):
    x, y = get_high_frequencies(img)
    return tf.reduce_mean(tf.square(x)) + tf.reduce_mean(tf.square(y))


def get_style_content_loss(outputs, content_targets, style_targets, content_layers,
                           style_layers, img):
    content_loss = compute_loss(outputs['content'], content_targets)
    style_loss = compute_loss(outputs['style'], style_targets)

    content_loss *= FLAGS.content_weight / len(content_layers)
    style_loss *= FLAGS.style_weight / len(style_layers)

    total_loss = style_loss + content_loss
    total_loss += FLAGS.tv_weight * variation_loss(img)

    return total_loss
