import tensorflow as tf 
from sklearn import preprocessing as sklearn_pre
import numpy as np 

from util import *

from config import *

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
		cur_x = np.array([e.reshape(-1,1) for e in dataset.train_x[i:i+batch_size]]).transpose()[0]

		cur_y = layerify(dataset.train_y[i])
		for j in range(1, batch_size):
			cur_y = np.append(cur_y, layerify(dataset.train_y[i+j]), axis=1)

		session.run(train, feed_dict={input_nodes: cur_x, expected_output: cur_y})

def current_accuracy(session, dataset):
	"""
	Find the current accuracy of the model, across testing examples
	"""
	num_test = len(dataset.test_x)

	cur_x = np.array([i.reshape(-1,1) for i in dataset.test_x]).transpose()[0]

	output = session.run(output_nodes, feed_dict={input_nodes: cur_x})

	# Rotate, then reverse output nodes
	output = np.rot90(output)[::-1]

	num_correct = 0
	for i, prediction in enumerate(output):
		prediction = prediction.reshape(-1,1)
		if best_guess(prediction) == dataset.test_y[i]:
			num_correct += 1

	return num_correct / num_test

# ============================= Main code ========================================
if __name__ == "__main__":
	# Allow gpu memory growth (?)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.log_device_placement = True

	with tf.Session(config=config) as session:
		try:
			model_saver.restore(session, f"saves/{checkpoint_fname}")
			print("Loaded saved model")
		except ValueError:
			session.run(tf.global_variables_initializer())
			print("Initialised new model")

		print(f"Current accuracy: {current_accuracy(session, dataset)}")

		print(f"Running {num_epochs} epochs")
		for i in range(num_epochs):
			training_epoch(session, dataset)
			if (i+1) % 5 == 0:
				print(f"After {i+1} epochs: {current_accuracy(session, dataset)}")
		print("done")

		print(f"New accuracy after {num_epochs} epochs: {current_accuracy(session, dataset)}")

		save_dir = model_saver.save(session, f"saves/{checkpoint_fname}")
		print(f"Saved model to {save_dir}")