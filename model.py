from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
import math
# import cv2
import tensorflow as tf
from conv import *
from data_utils import * 

def model(X,is_training):
    with tf.variable_scope("conv1"):
        out=mapping_conv(X,[3, 3, 3, 32],[1,1,1,1],padding='SAME')
        out=activation_conv(out)
    with tf.variable_scope("conv2"):
        out=mapping_conv(out,[3, 3, 32, 64],[1,1,1,1],padding='SAME')
        out=activation_conv(out)
        out=tf.nn.max_pool(out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    with tf.variable_scope("conv3"):
        out=mapping_conv(out,[3, 3, 64, 96],[1,1,1,1],padding='SAME')
        out=activation_conv(out)
    with tf.variable_scope("conv4"):
        out=mapping_conv(out,[3, 3, 96, 96],[1,1,1,1],padding='SAME')
        out=activation_conv(out)
        out=tf.nn.max_pool(out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        out=tf.reshape(out,[-1,6144])
    with tf.variable_scope("fc1"):
        out=mapping_fc(out,[6144,1024])
        out=activation_fc(out)
        out=tf.cond(is_training,lambda: tf.nn.dropout(out,0.5),lambda: out)
    with tf.variable_scope("fc2"):
        out=mapping_fc(out,[1024,10])
    y_out = out
    return y_out

def loss_accuracy(y_out,y):
    mean_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out))
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_out,1)),tf.float32))
    return mean_loss,accuracy








