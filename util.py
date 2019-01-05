import numpy as np

def flatten(arr, length):
	return arr.reshape(-1,length)

def layerify(num, max=10):
	"""
	Converts the numerical expected output to an output layer
	"""
	layer = np.zeros(max, dtype=np.float32)
	layer[num-1] = 1
	return layer.reshape(-1,1)

def best_guess(output):
	"""
	Given the output of the network, determine the best guess.
	"""
	return output.argmax()+1