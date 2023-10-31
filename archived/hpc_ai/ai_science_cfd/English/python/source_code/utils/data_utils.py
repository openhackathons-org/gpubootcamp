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
 

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import skfmm
from tqdm import tqdm


def eval_input_fn(dataset):
    dataset = dataset.batch(32)
    return dataset.__iter__()

def load_test_data(number):
    filename = "data/computed_car_flow/sample_{0:d}/fluid_flow_0002.h5".format(number)

    stream_flow = h5py.File(filename)
    v = stream_flow['Velocity_0']
    v = np.array(v).reshape([1,128,256+128,3])[:,:,0:256,0:2]
    b = np.array(stream_flow["Gamma"]).reshape(1, 128, 256 + 128,1)[:,:,0:256,:]
    return (b,v)

def plot_keras_loss(history):
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def plot_boundary(ax, boundary):
    ax.imshow(np.squeeze(boundary), cmap='Greys')


def plot_flow(ax, velocity):
    velocity = np.squeeze(velocity)
    Y, X = np.mgrid[0:velocity.shape[0],0:velocity.shape[1]]
    U = velocity[:,:,0]
    V = velocity[:,:,1]
    strm = ax.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn', integration_direction="both")
    return strm


def plot_flow_data(boundary, velocity, single_plot = False, fig = None, ax = None):
    if single_plot:
        if not (ax or fig):
            fig = plt.figure(figsize = (10,4))
            ax = fig.add_subplot(111)
        plot_boundary(ax, boundary)
        strm = plot_flow(ax, velocity)
        ax.set_title('Input data + simulated flow field')
    else:
        fig = plt.figure(figsize = (16,4))
        ax = fig.add_subplot(121)
        plot_boundary(ax, boundary)
        ax.set_title('Input data X')

        ax = fig.add_subplot(122)
        strm = plot_flow(ax, velocity)
        ax.set_ylim((128,0)) # reverse the y axes to match the boundary plot
        ax.set_title('Simulated flow lines Y')
        ax.set_aspect('equal')

    fig.colorbar(strm.lines)
    return strm

def plot_test_result(x, y, y_hat):
    # Display the simulated and the predicted flow field
    
    # Display field lines
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(221)
    plot_boundary(ax, x)
    strm = plot_flow(ax, y)
    fig.colorbar(strm.lines)
    ax.set_title('Simulated flow')
                
    ax = fig.add_subplot(222)
    plot_boundary(ax, x)
    strm = plot_flow(ax, y_hat)
    fig.colorbar(strm.lines)
    ax.set_title('Flow predicted by NN')
    
    # Show magnitude of the flow
    sflow_plot = np.concatenate([y, y_hat, y-y_hat], axis=2) 
    boundary_concat = np.concatenate(3*[x], axis=2) 

    sflow_plot = np.sqrt(np.square(sflow_plot[:,:,:,0]) + np.square(sflow_plot[:,:,:,1])) # - .05 *boundary_concat[:,:,:,0]
    ax = fig.add_subplot(2,1,2)
    im = ax.imshow(np.squeeze(sflow_plot), cmap='hsv', zorder=1)
    
    # Adding car shape in black
    # We create an RGBA image, setting alpha from boundary_concat
    # This way we can plot the boundary image over the contour plot without the white pixels hiding the contour
    im2 = np.zeros(boundary_concat.shape[1:3] + (4,))
    im2[:, :, 3] = np.squeeze(boundary_concat)
    ax.imshow(im2, cmap='Greys', zorder=2)
    
    ax.set_title('Magnitude of the flow: (left) simulated, (middle) Neural Net prediction,'
                 ' (right) difference')
    fig.colorbar(im)


def calc_sdf(x):
    """ calculates thes signed distance function for a batch of input data
    Arguments:
    x -- nbatch x Heiht x Width [ x 1]
    """
    if x.ndim == 2 or x.ndim == 3: # H x W [ x 1 ] optional single channel
        sdf = skfmm.distance(np.squeeze(0.5-x))
    elif x.ndim == 4: # batched example Nbatch x H x W x 1
        sdf = np.zeros(x.shape)
        for i in range(x.shape[0]):
            sdf[i,:,:,:] = skfmm.distance(np.squeeze(0.5-x[i,:,:,:]))
    else:
        print("Error, invalid array dimension for calc_sdf", x.shape)
        
    return sdf 

def plot_sdf(x, sdf = None, plot_boundary=True):
    x = np.squeeze(x)
    if sdf is None:
        sdf = calc_sdf(x)
    else:
        sdf = np.squeeze(sdf)
    fig = plt.figure(figsize=(16,5))
    ax = plt.subplot(111)
    Y, X = np.mgrid[0:sdf.shape[0],0:sdf.shape[1]]
    cs = ax.contourf(X, Y, sdf, 50, cmap=matplotlib.cm.coolwarm, zorder=1)
    
    ax.set_ylim((128,0))
    fig.colorbar(cs, ax=ax)
    ax.set_aspect('equal')

    if plot_boundary:
        ax.contour(X,Y, sdf, [0], colors='black', zorder=2, linewidths=2)
        # We create an RGBA image, setting alpha from x
        # This way we can plot the boundary image over the contour plot without the white pixels hiding the contour

        #im2 = np.zeros(x.shape + (4,))
        #im2[:, :, 3] = x
        #ax.imshow(im2, cmap='Greys', zorder=2)


def display_flow(sample_number):
    b,v = load_test_data(sample_number)
    plot_flow_data(b,v)

def parse_flow_data(serialized_example):
    shape = (128,256)
    features = {
      'boundary':tf.io.FixedLenFeature([],tf.string),
      'sflow':tf.io.FixedLenFeature([],tf.string)
    }
    parsed_features = tf.io.parse_single_example(serialized_example, features)
    boundary = tf.io.decode_raw(parsed_features['boundary'], tf.uint8)
    sflow = tf.io.decode_raw(parsed_features['sflow'], tf.float32)
    boundary = tf.reshape(boundary, [shape[0], shape[1], 1])
    sflow = tf.reshape(sflow, [shape[0], shape[1], 2])
    boundary = tf.cast(boundary,dtype=tf.float32)
    sflow = tf.cast(sflow,dtype=tf.float32)
    return boundary, sflow

def parse_sdf_flow_data(serialized_example):
    shape = (128,256)
    features = {
      'sdf_boundary':tf.io.FixedLenFeature([],tf.string),
      'sflow':tf.io.FixedLenFeature([],tf.string)
    }
    parsed_features = tf.io.parse_single_example(serialized_example, features)
    boundary = tf.io.decode_raw(parsed_features['sdf_boundary'], tf.float32)
    sflow = tf.io.decode_raw(parsed_features['sflow'], tf.float32)
    boundary = tf.reshape(boundary, [shape[0], shape[1], 2])
    sflow = tf.reshape(sflow, [shape[0], shape[1], 2])
    boundary = tf.cast(boundary,dtype=tf.float32)
    sflow = tf.cast(sflow,dtype=tf.float32)
    return boundary, sflow


# helper function
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def create_sdf_file(name):
#     # Set up a dataset for the input data
#     dataset = tf.data.TFRecordDataset('data/'+ name + '.tfrecords')
#
#     # Transform binary data into image arrays
#     dataset = dataset.map(parse_flow_data) 
#
#     # Create an iterator for reading a batch of input and output data
#     iterator = iter(dataset)
#     boundary_t, sflow_t = next(iterator)
#
#
#     # create tf writer
#     record_filename = 'data/' + name + '_sdf.tfrecords'
#
#     writer = tf.io.TFRecordWriter(record_filename)
#
#     shape = [128, 256]
#     
#     if name == 'train':
#         num_images = 3000
#     elif name == 'test' :
#         num_images = 28
#     else:
#         print('error, number of images is not known for ', name)
#         num_images = 1000
#    
#     for i in (range(num_images)):
#             print("*",end='')
#             # read in images    
#             b, s = boundary_t, sflow_t
#  
#             # calculate signed distance function
#             sdf = np.reshape(calc_sdf(b),(shape[0], shape[1], 1))
#         
#             # keep both the boundary (channel 1) and the SDF (channel 2) as input
#             b_sdf = np.concatenate((b,sdf),axis=2)
#     
#             # process frame for saving
#             boundary = np.float32(b_sdf)
#             boundary = boundary.reshape([1,shape[0]*shape[1]*2])
#             boundary = boundary.tostring()
#             sflow = np.float32(s)
#             sflow = sflow.reshape([1,shape[0]*shape[1]*2])
#             sflow = sflow.tostring()
#
#             # create example and write it
#             example = tf.train.Example(features=tf.train.Features(feature={
#             'sdf_boundary': _bytes_feature(boundary),
#             'sflow': _bytes_feature(sflow)}))
#             writer.write(example.SerializeToString())
#             print("/")

def create_sdf_file(name):
    # Set up a dataset for the input data
    dataset = tf.data.TFRecordDataset('data/'+ name + '.tfrecords')

    # Transform binary data into image arrays
    dataset = dataset.map(parse_flow_data) 

    # Create an iterator for reading a batch of input and output data
    iterator = iter(dataset)


    # create tf writer
    record_filename = 'data/' + name + '_sdf.tfrecords'

    writer = tf.io.TFRecordWriter(record_filename)

    shape = [128, 256]
    
    if name == 'train':
        num_images = 3000
    elif name == 'test' :
        num_images = 28
    else:
        print('error, number of images is not known for ', name)
        num_images = 1000
   
    for i in tqdm(range(num_images)):
        try:
            # read in images
            boundary_t, sflow_t = next(iterator)
            b, s = boundary_t, sflow_t
 
            # calculate signed distance function
            sdf = np.reshape(calc_sdf(b),(shape[0], shape[1], 1))
        
            # keep both the boundary (channel 1) and the SDF (channel 2) as input
            b_sdf = np.concatenate((b,sdf),axis=2)
    
            # process frame for saving
            boundary = np.float32(b_sdf)
            boundary = boundary.reshape([1,shape[0]*shape[1]*2])
            boundary = boundary.tostring()
            sflow = np.float32(s)
            sflow = sflow.reshape([1,shape[0]*shape[1]*2])
            sflow = sflow.tostring()

            # create example and write it
            example = tf.train.Example(features=tf.train.Features(feature={
            'sdf_boundary': _bytes_feature(boundary),
            'sflow': _bytes_feature(sflow)}))
            writer.write(example.SerializeToString())
        except tf.errors.OutOfRangeError:
            print('Finished writing into', record_filename)
            print('Read in ', idx, ' images')
            break


