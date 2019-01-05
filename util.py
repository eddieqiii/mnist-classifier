import numpy as np

def flatten(arr):
	return arr.reshape(-1,1)

def layerify(num, max=10):
	"""
	Converts the numerical expected output to an output layer
	"""
	layer = np.zeros(max, dtype=np.float32)
	layer[num-1] = 1
	return flatten(layer)

def best_guess(output):
	"""
	Given the output of the network, determine the best guess.
	"""
	return output.argmax()+1