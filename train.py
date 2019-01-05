import tensorflow as tf 
from sklearn import preprocessing as sklearn_pre
import numpy as np 

from util import *

batch_size = 1000

print("Loading data... ", end="")
from data import MnistDataset
dataset = MnistDataset()
dataset.load_data()
print("done")

print("Building graph... ", end="")
from graph import *
print("done")

# ================================================================================
def training_epoch(session, dataset):
	"""
	Run one complete training epoch
	"""
	num_train = len(dataset.train_x)
	for i in range(0, num_train, batch_size):
		# Get current input and output example to feed in
		cur_x = flatten(dataset.train_x[i:i+batch_size], batch_size)
		cur_y = layerify(dataset.train_y[i])

		session.run(train, feed_dict={input_nodes: cur_x, expected_output: cur_y})

def current_accuracy(session, dataset):
	"""
	Find the current accuracy of the model, across testing examples
	"""
	num_test = len(dataset.test_x)
	num_correct = 0

	for i in range(num_test):
		cur_x = dataset.test_x[i].reshape(-1,1)
		expected = dataset.test_y[i]

		output = session.run(output_nodes, feed_dict={input_nodes: cur_x})
		if best_guess(output) == expected:
			num_correct += 1

	return num_correct / num_test

# ============================= Main code ========================================
# Allow gpu memory growth (?)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True

with tf.Session(config=config) as session:
	session.run(tf.global_variables_initializer())
	print(f"Current accuracy: {current_accuracy(session, dataset)}")

	print("Running 100 epochs")
	for i in range(100):
		training_epoch(session, dataset)
		print(f"New accuracy after {i} epochs: {current_accuracy(session, dataset)}")

	