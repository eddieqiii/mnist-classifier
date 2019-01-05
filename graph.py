import tensorflow as tf 
import numpy as np 

# Create event file for TensorBoard
writer = tf.summary.FileWriter('logs/')

learning_rate = 0.00001

num_input = 28 ** 2 # 28x28 bitmaps
num_1 = 16
num_2 = 16
num_output = 10

with tf.device('/device:CPU:0'):
	with tf.variable_scope("input"):
		input_nodes = tf.placeholder(np.float32, shape=(num_input, None))

	with tf.variable_scope("layer1"):
		layer1_weights = tf.get_variable("weights1", 
			dtype=np.float32, 
			shape=(num_1, num_input), 
			initializer=tf.contrib.layers.xavier_initializer())
		layer1_biases = tf.get_variable("biases1", 
			dtype=np.float32, 
			shape=(num_1, 1), 
			initializer=tf.zeros_initializer())
		layer1_nodes = tf.math.sigmoid(tf.matmul(layer1_weights, input_nodes)
			+ layer1_biases)

	with tf.variable_scope("layer2"):
		layer2_weights = tf.get_variable("weights2", 
			dtype=np.float32, 
			shape=(num_2, num_1), 
			initializer=tf.contrib.layers.xavier_initializer())
		layer2_biases = tf.get_variable("biases2", 
			dtype=np.float32, 
			shape=(num_2, 1), 
			initializer=tf.zeros_initializer())
		layer2_nodes = tf.math.sigmoid(tf.matmul(layer2_weights, layer1_nodes)
			+ layer2_biases)

	with tf.variable_scope("output"):
		output_weights = tf.get_variable("weightsout", 
			dtype=np.float32, 
			shape=(num_output, num_2), 
			initializer=tf.contrib.layers.xavier_initializer())
		output_biases = tf.get_variable("biasesout", 
			dtype=np.float32, 
			shape=(num_output, 1), 
			initializer=tf.zeros_initializer())
		output_nodes = tf.math.sigmoid(tf.matmul(output_weights, layer2_nodes)
			+ output_biases)

	with tf.variable_scope("cost"):
		expected_output = tf.placeholder(np.float32, shape=(num_output, input_nodes.shape[1]))
		cost = tf.square(output_nodes - expected_output)

	with tf.variable_scope("train"):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train = optimizer.minimize(cost)

model_saver = tf.train.Saver()

# Write evemt file
writer.add_graph(tf.get_default_graph())
writer.flush()