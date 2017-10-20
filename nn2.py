#
#		Brian Santoso
#		10/18/17
#

import numpy as np

np.random.seed(5465456)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def mean_squared_error(output, target):
	difference = output - target
	sum_of_squared_differences = np.dot(difference, difference)
	return 0.5 * sum_of_squared_differences / len(target)

class NeuralNet:

	def __init__(self, sizes, randomize = True):

		self.sizes = sizes
		self.num_of_layers = len(sizes)

		# array of 2d numpy arrays M x N where 
		# M is number of nodes in current layer
		# N is number of nodes in previous layer
		if randomize:
			self.weights = [np.random.rand(sizes[x], sizes[x - 1]) for x in range(1, self.num_of_layers)]		
			self.biases = [np.random.rand(sizes[x]) for x in range(1, self.num_of_layers)]
		else:
			self.weights = [np.ones(shape=(sizes[x], sizes[x - 1])) for x in range(1, self.num_of_layers)]
			self.biases = [np.zeros(sizes[x]) for x in range(1, self.num_of_layers)]

	def feed_forward(self, input):

		# input is a numpy array
		outputs = [input]

		for l in range(1, self.num_of_layers):
			# weights of nodes of current layer 
			# (weights array is 1 smaller than outputs since it does not include inputs, hence the l - 1)
			# weights of outputs of previous layer
			x = np.dot(self.weights[l - 1], outputs[l - 1]) + self.biases[l - 1]
			outputs.append(sigmoid(x))

		return outputs

	def back_propagation(self, outputs, target, rate):

		output = outputs[self.num_of_layers - 1]
		deltas = [output * (1 - output) * (output - target)]

		for l in range(self.num_of_layers - 2, 0, -1):
			# delta_j = output_j * (1 - output_j) * sum(delta_k * weight_jk)
			o_j = outputs[l]
			delta = o_j * (1 - o_j) * np.dot(self.weights[l].transpose(), deltas[0])
			deltas.insert(0, delta)

		partial_derivatives = []
		for l in range(1, self.num_of_layers):
			partial_derivatives.append(deltas[l - 1][:,None] * outputs[l - 1])

		for l in range(1, self.num_of_layers):
			self.weights[l - 1] = self.weights[l - 1] - rate * partial_derivatives[l - 1]
			self.biases[l - 1] = self.biases[l - 1] - rate * deltas[l - 1]

		return

	def train(self, training_data, rate, epoch, min_error = 1e-6, printinfo = False):

		d = 0
		error = 1e99

		while d < epoch and epoch > min_error:

			i = d % len(training_data)
			inp = training_data[i][0]
			target = training_data[i][1]
			outputs = self.feed_forward(inp)

			error = mean_squared_error(outputs[-1:], target)

			self.back_propagation(outputs, target, rate)

			d += 1
		
			if printinfo and d % 10 == 0:
				print('epoch: ', d, ' error: ', error)

		return

	def info(self):
		print('sizes: {0} \nnum_of_layers: {1} \nweights: {2} \nbiases: {3} \n'.format(self.sizes, self.num_of_layers, self.weights, self.biases))


training_data = [ 	[np.array([0, 0]), np.array([0])],
					[np.array([1, 0]), np.array([1])],
					[np.array([0, 1]), np.array([1])],
					[np.array([1, 1]), np.array([0])]	]

nn = NeuralNet((2, 2, 1))
nn.train(training_data, 10, 2000, 1e-4)

print('tests:')
o = nn.feed_forward(np.array([0, 0]))
print('output: ', o[-1:])
o = nn.feed_forward(np.array([1, 0]))
print('output: ', o[-1:])
o = nn.feed_forward(np.array([0, 1]))
print('output: ', o[-1:])
o = nn.feed_forward(np.array([1, 1]))
print('output: ', o[-1:])

nn.info()

# a possible XOR Solution
# nn.weights = [np.array([[20, 20], [-20, -20]]), np.array([[20, 20]])]
# nn.biases = [np.array([-10, 30]), np.array([-30])]