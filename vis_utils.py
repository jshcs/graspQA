#This is the utility file for visualising intermediate neuron activations
#https://arxiv.org/pdf/1506.06579.pdf

import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf

tf.reset_default_graph()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

def plot_blocks(blocks,layer_name):
	H,W,C,K=blocks.shape
	plt.figure(figsize=(10,10))
	cols=4
	rows=round(float(K)/float(cols),0)
	for filter in range(K):
		plt.subplot(rows,cols,filter+1)
		plt.title(layer_name+","+str(filter+1))
		plt.imshow(blocks[0,:,:,i])

def select_layer(im,layer_name):
	H,W,C=im.shape
	if H!=32:
		resized_im=np.resize(im,[32,32])
	flattened_im=np.resize(resized_im,[1,1024])
	feed_dict={x:flattened_im,keep_prob:1.0}
	blocks=sess.run(layer_name,feed_dict=feed_dict)
	plot_blocks(blocks,layer_name)
