#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from numpy import *
import os
from pylab import *
from skimage.color import rgb2gray
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

class Q_Net():
  
  def __init__(self):    
        input_size = 4  

        self.input = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="X")
        self.target = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions = tf.placeholder(shape=[None,2], dtype=tf.float32, name="actions")


    
        self.W_fc1 = tf.Variable(tf.truncated_normal([input_size, 32],stddev=0.01))
        self.b_fc1 = tf.Variable(tf.zeros(32))

        self.W_fc2 = tf.Variable(tf.truncated_normal([32, 64],stddev=0.01))
        self.b_fc2 = tf.Variable(tf.zeros(64))

        self.W_fc3 = tf.Variable(tf.truncated_normal([64, 2],stddev=0.01))
        self.b_fc3 = tf.Variable(tf.zeros(2))


        self.out_fc1 = tf.nn.relu(tf.matmul(self.input, self.W_fc1) + self.b_fc1)
        self.out_fc2 = tf.nn.relu(tf.matmul(self.out_fc1, self.W_fc2) + self.b_fc2)
        self.out_fc3 = tf.matmul(self.out_fc2, self.W_fc3) + self.b_fc3



        self.action_predictions = tf.reduce_sum(tf.multiply(self.out_fc3, self.actions), reduction_indices=[1])
        self.loss = tf.reduce_mean(tf.square(self.target - self.action_predictions))

        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    