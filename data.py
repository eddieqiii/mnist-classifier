import tensorflow as tf 
import numpy as np 

class MnistDataset:
	"""
	This is written as a class to avoid polluting the global namespace
	"""
	def __init__(self):
		self.mnist = tf.keras.datasets.mnist

	def load_data(self):
		(train_x, train_y), (test_x, test_y) = self.mnist.load_data()
		self.train_x = train_x / 255
		self.train_y = train_y
		self.test_x = test_x / 255
		self.test_y = test_y