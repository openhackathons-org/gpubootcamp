# Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
# 
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""functions used to construct different architectures
"""

# Import Necessary Libraries

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D , Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def int_shape(x):
  return x.get_shape().as_list()

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat(values=[x, -x], axis=axis))

def set_nonlinearity(name):
  if name == 'concat_elu':
    return concat_elu
  elif name == 'elu':
    return tf.nn.elu
  elif name == 'concat_relu':
    return tf.nn.crelu
  elif name == 'relu':
    return tf.nn.relu
  else:
    raise('nonlinearity ' + name + ' is not supported')

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable(name, shape, initializer):
  """Helper to create a Variable.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  # getting rid of stddev for xavier ## testing this for faster convergence
  var = tf.Variable(name=name, initial_value=initializer(shape=shape))
  return var

def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):

    conv = Conv2D(num_features, kernel_size, strides=(stride, stride), padding='same')(inputs)
    
    if nonlinearity is not None:
      conv = nonlinearity(conv)
    return conv

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
        
    conv = Conv2DTranspose(num_features, kernel_size, strides=(stride, stride), padding='same')(inputs)
    
    if nonlinearity is not None:
      conv = nonlinearity(conv)

    return conv

def fc_layer(inputs, hiddens, idx, nonlinearity=None, flat = False):
    input_shape = inputs.get_shape().as_list()
    if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
        dim = input_shape[1]
        inputs_processed = inputs
    output_biased = Dense(hiddens, input_dim=dim)(inputs_processed)
    if nonlinearity is not None:
      output_biased = nonlinearity(ouput_biased)
    return output_biased
def nin(x, num_units, idx):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.keras.layers.Reshape((np.prod(s[1:-1]),s[-1]))(x)
    x = fc_layer(x, num_units, idx) # fully connected layer
    return tf.keras.layers.Reshape(tuple(s[1:-1]+[num_units]))(x)
