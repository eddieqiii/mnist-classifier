import tensorflow as tf 
from sklearn import preprocessing as sklearn_pre
import numpy as np 

from util import *

print("Loading data... ", end="")
from data import MnistDataset
dataset = MnistDataset()
dataset.load_data()
print("done")

print("Building graph... ", end="")
from graph import *
print("done")

# Main code

def training_epoch(session, dataset):
	"""
	Run one complete training epoch
	"""
	num_train = len(dataset.train_x)
	for i in range(num_train):
		# Get current input and output example to feed in
		cur_x = flatten(dataset.train_x[i])
		cur_y = layerify(dataset.train_y[i])

		session.run(train, feed_dict={input_nodes: cur_x, expected_output: cur_y})

def current_accuracy(session, dataset):
	"""
	Find the current accuracy of the model, across testing examples
	"""
	num_test = len(dataset.test_x)
	num_correct = 0

	for i in range(num_test):
		cur_x = flatten(dataset.test_x[i])
		expected = dataset.test_y[i]

		output = session.run(output_nodes, feed_dict={input_nodes: cur_x})
		if best_guess(output) == expected:
			num_correct += 1

	return num_correct / num_test