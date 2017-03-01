#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import tensorflow as tf

#Convolutional Net taken from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/


#from caffe_classes import class_names
#IMPORTING OF THREE CLASSES ACTIONS FROM CAFFE STYLE FILE
from classes_2 import class_names

#FLAGS FOR MODEL SAVE/LOAD
Load = True
Save = False


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def Preprocessing(weights,x):
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(weights["conv1"][0])
    conv1b = tf.Variable(weights["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(weights["conv2"][0])
    conv2b = tf.Variable(weights["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(weights["conv3"][0])
    conv3b = tf.Variable(weights["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(weights["conv4"][0])
    conv4b = tf.Variable(weights["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(weights["conv5"][0])
    conv5b = tf.Variable(weights["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    return maxpool5

class Q_Net():

  def __init__(self):
        self.W6 = tf.Variable(tf.random_normal([5120,4096]))
        self.B6 = tf.Variable(tf.random_normal([4096]))
        self.W7 = tf.Variable(tf.random_normal([4096,4096]))
        self.B7 = tf.Variable(tf.random_normal([4096]))
        self.W8 = tf.Variable(tf.random_normal([4096,3]))
        self.B8 = tf.Variable(tf.random_normal([3]))

  def Initialitation(self, Load):
    if Load == True:
        saver.restore(sess,'./model/my-model')

  def Preprocessing(self,weights,x):
    self.maxpool5 = Preprocessing(weights,x)

  def Train(self,Target):
     self.loss = tf.reduce_mean(tf.square(Target - self.prob))
     self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
     self.updateWeights = self.trainer.minimize(self.loss)

  def Predict(self):
    self.fc6 = tf.nn.relu_layer(tf.reshape(self.maxpool5, [-1, int(prod(self.maxpool5.get_shape()[1:]))]), self.W6, self.B6)
    self.fc7 = tf.nn.relu_layer(self.fc6, self.W7, self.B7)
    self.fc8 = tf.matmul(self.fc7, self.W8) + self.B8
    self.prob = tf.nn.softmax(self.fc8)

  def Test(self):
     pass

  def Argmax(self):
      self.argmax = tf.argmax(self.prob,1)

#FORMATION OF THE GRAPH, READING OF THE IMAGE AND THE PRETRAINED WEIGHTS

image = (imread("atari.png")[:,:,:3]).astype(float32)
image = image - mean(image)
xdim = image.shape[0:]
x = tf.placeholder(tf.float32, (None,) + xdim)
weights = load("weights_alexnet.py").item()


#Costructors of the Nets

QTarget = Q_Net()
Q2 = Q_Net()

#Definition of the session and of the variable that can be saved and restored
sess = tf.Session()
saver = tf.train.Saver({'W6': QTarget.W6, 'B6': QTarget.B6,'W7': QTarget.W7, 'B7': QTarget.B7, 'W8': QTarget.W8, 'B8': QTarget.B8})

#Loading of model
QTarget.Initialitation(Load)
Q2.Initialitation(not(Load))

#Preprocessing of Data with Constants Layer and predictions with the two nets
#QTarget the fixed one and Q2 the one to be trained

QTarget.Preprocessing(weights,x)
QTarget.Predict()
Q2.Preprocessing(weights,x)
Q2.Predict()
Q2.Train(QTarget.prob)

#After Training prediction and Argmax
Q2.Predict()
Q2.Argmax()

#INITIALITATION OF THE GRAPH

init = tf.initialize_all_variables()
sess.run(init)

output = sess.run([Q2.prob,Q2.argmax], feed_dict = {x:[image]})


#Saving the model eventually
if ( Save == True ):
    saver.save(sess, './model/my-model')

################################################################################

#Output:

for input_im_ind in range(output.shape[0]):
    inds = argsort(output)[input_im_ind,:]
    print "Image", input_im_ind
    for i in range(3):print class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]]
