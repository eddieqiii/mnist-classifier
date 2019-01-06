from PIL import Image
import numpy as np

from util import *

from config import *

def print_arr(img_arr):
	"""
	Print an ascii representation of a 2D np array
	"""
	for line in img_arr:
		for pix in line:
			if pix > 0.1:
				print(".",end="")
			else:
				print(" ",end="")
		print()

print("Defining graph... ", end="")
from graph import *
print("done")

print(f"Opening {guess_fname}... ",end="")
image = Image.open(guess_fname)
img_list = image.getdata()
img_array = np.array(img_list).reshape(28,28) / 100 # Scale image data to between 0 and 1
print("done")

print(f"ASCII interpretation of {guess_fname}:")
print_arr(img_array)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
	# Load saved model, if it exists
	try:
		model_saver.restore(session, f"saves/{checkpoint_fname}")
		print("Loaded saved model")
	except ValueError:
		session.run(tf.global_variables_initializer())
		print("Initialised new model")

	output = session.run(output_nodes, feed_dict={input_nodes: img_array.reshape(-1,1)})
	print(f"Network output:\n{output}")

	guess = best_guess(output)
	print(f"Best guess: {guess} ({output[guess-1][0]*100}% certain)")