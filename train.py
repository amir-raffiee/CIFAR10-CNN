from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
import math
import tensorflow as tf
from data_utils import * 
from conv import *
from model import *


X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

def estimator():
	tf.reset_default_graph()
	X=tf.placeholder(tf.float32,[None,32,32,3])
	y=tf.placeholder(tf.float32,[None,10])
	is_training=tf.placeholder(tf.bool)

	y_out = model(X,is_training)

	mean_loss,accuracy=loss_accuracy(y_out,y)

	corrected_pred=tf.cast(tf.equal(tf.argmax(y_out,1),tf.argmax(y,1)),tf.float32)
	optimizer = tf.train.AdamOptimizer(1e-3)
	extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(extra_update_ops):
		train_step = optimizer.minimize(mean_loss)
	def run_model(session, Xd, yd,X_val,y_val,epochs=1, batch_size=64, print_every=100,plot_losses=False):
		train_indicies = np.arange(Xd.shape[0])
		np.random.shuffle(train_indicies)
		training_now = train_step is not None
		iter_cnt = 0
		losses=[]
		losses_val=[]
		accuracy_val=[]
		for e in range(epochs):
			correct = 0
			for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
				start_idx = (i*batch_size)%Xd.shape[0]
				idx = train_indicies[start_idx:start_idx+batch_size]
				feed_dict = {X: Xd[idx,:],y: invert(yd[idx],10),is_training: True }
				actual_batch_size = yd[idx].shape[0]
				loss, corr, _ = session.run([mean_loss,corrected_pred,train_step],feed_dict=feed_dict)
				losses.append(loss)
				correct += np.sum(corr)
				if training_now and (iter_cnt % print_every) == 0:
					print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
				iter_cnt += 1
			loss,acc=sess.run([mean_loss,accuracy],feed_dict={X:X_val,y:invert(y_val,10),is_training: False})
			losses_val.append(loss)
			accuracy_val.append(acc)
			print('validation accuracy=',accuracy_val)
	with tf.Session() as sess:
		with tf.device("/cpu:0"):
			sess.run(tf.global_variables_initializer())
			print('Training')
			run_model(sess,X_train,y_train,X_val,y_val,50,128,100,True)

estimator()











