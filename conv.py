from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
# import cv2
import tensorflow as tf

class Convolution(object):

    def __init__(self,X,kernel,strides,is_training,padding='SAME',activation='relu',epsilon=1e-6):
        self.X=X
        self.kernel=kernel
        self.strides=strides
        self.padding=padding
        self.is_training=is_training
        self.activation=activation
        self.channel=kernel[-1]
        self.epsilon=epsilon
        self.W1=tf.Variable(tf.truncated_normal(self.kernel, stddev=0.1),trainable=True)
        self.b1=tf.Variable(tf.constant(0.1, shape=[self.channel]),trainable=True)

    def mapping(self):
    	
    	out=tf.nn.conv2d(self.X,self.W1,strides=self.strides,padding=self.padding)+self.b1
    	self.out=out

    def bn(self):
    	with tf.variable_scope('bn'):
    		self.gamma=tf.get_variable('gamma',shape=[self.channel],trainable=True)
    		self.beta=tf.get_variable('beta',shape=[self.channel],trainable=True)
    		x_mean,x_var=tf.nn.moments(self.out,[0,1,2])
    		ema=tf.train.ExponentialMovingAverage(decay=0.99)

    		def update():
    			update_process=ema.apply([x_mean,x_var])
    			with tf.control_dependencies([update_process]):
    				return x_mean,x_var

    		mean,var=tf.cond(self.is_training,update,lambda: (ema.average(x_mean),ema.average(x_var)))
    		self.out=tf.nn.batch_normalization(self.out,mean,var,self.beta,self.gamma,self.epsilon)
    		# return self.out
    		
    def activation_map(self):
    	out=tf.nn.relu(self.out)
    	return out


class FullyConnected(object):
	def __init__(self,X,dimension,is_training,activation='relu',epsilon=1e-6):
		self.X=X
		self.dimension=dimension
		self.is_training=is_training
		self.activation=activation
		self.epsilon=epsilon
		self.W=tf.Variable(tf.truncated_normal(self.dimension,mean=0.0, stddev=0.05),trainable=True)
		self.b=tf.Variable(tf.zeros([self.dimension[-1]]),trainable=True)

	def mapping(self):
		# self.W=tf.get_variable('W1',shape=self.dimension,trainable=True)
		# self.b=tf.get_variable('b',shape=[self.dimension[-1]],trainable=True)
		self.out=tf.matmul(self.X,self.W)+self.b

	def bn(self):
		with tf.variable_scope('bn'):
			self.gamma=tf.get_variable('gamma',shape=[self.dimension[-1]],trainable=True)
			self.beta=tf.get_variable('beta',shape=[self.dimension[-1]],trainable=True)
			x_mean,x_var=tf.nn.moments(self.X,[0])
			ema=tf.train.ExponentialMovingAverage(decay=0.99)
			def update():
				update_process=ema.apply([x_mean,x_var])
				with tf.control_dependencies([update_process]):
					return x_mean,x_var
			mean,var=tf.cond(self.is_training,update,lambda: (ema.average(x_mean),ema.average(x_var)))
			self.out=tf.nn.batch_normalization(self.out,mean,var,self.beta,self.gamma,self.epsilon)

	def activation_map(self):
		out=tf.nn.relu(self.out)
		return out

def mapping_conv(X,kernel,strides,padding='SAME'):
	W=tf.get_variable('Wconv',shape=kernel,trainable=True)
	b=tf.get_variable('bconv',shape=[kernel[-1]],trainable=True)
	out=tf.nn.conv2d(X,W,strides=strides,padding=padding)+b
	return out

def activation_conv(X,activation='relu'):
	return tf.nn.relu(X)

def bn_conv(X,output_channel,is_training,epsilon=1e-6):
	gamma=tf.get_variable('gamma',shape=[output_channel],trainable=True)
	beta=tf.get_variable('beta',shape=[output_channel],trainable=True)
	x_mean,x_var=tf.nn.moments(X,[0,1,2])
	ema=tf.train.ExponentialMovingAverage(decay=0.99)
	def update():
		update_process=ema.apply([x_mean,x_var])
		with tf.control_dependencies([update_process]):
			return x_mean,x_var
	mean,var=tf.cond(is_training,update,lambda: (ema.average(x_mean),ema.average(x_var)))
	out=tf.nn.batch_normalization(X,mean,var,beta,gamma,epsilon)
	return out

def mapping_fc(X,dimension):
	Weight=tf.get_variable('Weight',shape=dimension,trainable=True)
	biases=tf.get_variable('biases',shape=[dimension[-1]],trainable=True)
	return tf.matmul(X,Weight)+biases

def activation_fc(X,activation='relu'):
	return tf.nn.relu(X)



