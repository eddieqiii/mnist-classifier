from PIL import Image
import numpy as np

from util import *

def print_arr(img_arr):
	for line in img_arr:
		for pix in line:
			if pix > 0.1:
				print(".",end="")
			else:
				print(" ",end="")
		print()

fname = "test.bmp"

print("Defining graph... ", end="")
from graph import *
print("done")

print(f"Opening {fname}... ",end="")
image = Image.open(fname)
img_list = image.getdata()
img_array = np.array(img_list).reshape(28,28) / 100
print("done")

print(f"ASCII interpretation of {fname}:")
print_arr(img_array)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
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