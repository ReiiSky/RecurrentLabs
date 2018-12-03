import math
import csv
import random 
import time
import mnist_loader


#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy

class Network(object):

	def __init__(self, sizes):	
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = Sigmoid(numpy.dot(w, a)+b)
		return a

	def Train(self, trainingData, epoch, batchSize, learningRate, testingData):

		testDataSize = len(testingData)
		trainDataSize = len(trainingData)
		for i in range(epoch):

			random.shuffle(trainingData)
			
			nabla_b = [numpy.zeros(b.shape) for b in self.biases]
			nabla_w = [numpy.zeros(w.shape) for w in self.weights]

			for j in range(0, trainDataSize, batchSize):

				nabla_b = [numpy.zeros(b.shape) for b in self.biases]
				nabla_w = [numpy.zeros(w.shape) for w in self.weights]
				miniBatchSize = 0
				for x, y in trainingData[j:j+batchSize]:
					delta_nabla_b, delta_nabla_w = self.Backpropagation(x, y)

					nabla_b = [(delta_nabla_b_row + nabla_b_row) for delta_nabla_b_row, nabla_b_row in zip(delta_nabla_b, nabla_b)]
					nabla_w = [(delta_nabla_w_row + nabla_w_row) for delta_nabla_w_row, nabla_w_row in zip(delta_nabla_w, nabla_w)]
					#nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
					#nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
					miniBatchSize = miniBatchSize + 1
				#print("Mini Batch Size: " + str(miniBatchSize))
				self.weights = [w_layer-(learningRate/miniBatchSize)*nw_layer for w_layer, nw_layer in zip(self.weights, nabla_w)]
				self.biases = [b_layer -(learningRate/miniBatchSize)*nb_layer for b_layer, nb_layer in zip(self.biases, nabla_b)]

				#self.weights = [w-(learningRate/miniBatchSize)*nw for w, nw in zip(self.weights, nabla_w)]
				#self.biases = [b-(learningRate/miniBatchSize)*nb for b, nb in zip(self.biases, nabla_b)]				
				#[print((learningRate/miniBatchSize)*nb_layer) for b_layer, nb_layer in zip(self.biases, nabla_b)]

			if testingData:
				print ("Epoch {0}: {1} / {2}".format(i, self.evaluate(testingData), testDataSize))
			else:
				print ("Epoch {0} complete".format(j))


	def update_mini_batch(self, mini_batch, eta):
		nabla_b = [numpy.zeros(b.shape) for b in self.biases]
		nabla_w = [numpy.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
		#[print((eta/len(mini_batch))*nb) for b, nb in zip(self.biases, nabla_b)]

	def Backpropagation(self, x, y):
		#Change in Biases
		nabla_b = [numpy.zeros(b.shape) for b in self.biases]

		#Change in Weights
		nabla_w = [numpy.zeros(w.shape) for w in self.weights]

		activations = []
		activations.append(x)
		zs = []

		for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
			#print(weights)
			#print(activations[index])
			#print(bias)
			individualWeightedSum = numpy.dot(weights, activations[index]) + bias
			#print(individualWeightedSum.shape)
			zs.append(individualWeightedSum)
			activation = Sigmoid(individualWeightedSum)
			activations.append(activation)

		#print("ZS: " + str(zs))
		#print("Last Activations:" + str(activations[-1]))
		#print(activations)
		sigmodPrime = SigmoidPrime(zs[-1])
		#print(trainingY)
		#print(activations[-1])
		delta = self.CostDerivative(activations[-1], y) * sigmodPrime 
		#print("Sigmod: " + str(sigmodPrime))
		#print("Cost Derivative: " + str(self.CostDerivative(activations[-1], trainingY)))

		nabla_b[-1] = delta
		#print("Delta: " + str(delta))
		nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
		#print("Delta: " + str(nabla_w[-1]))

		for l in range(2, len(self.weights)):
			z = zs[-l]
			sp = SigmodPrime(z)
			delta = numpy.dot(self.weights[-l+1], delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = numpy.dot(delta, activations[-l].transpose())

		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		test_results = [(numpy.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def CostDerivative(self, output_activations, y):
		return (output_activations-y)

#### Miscellaneous functions
def Sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+numpy.exp(-z))

def SigmoidPrime(z):
    """Derivative of the sigmoid function."""
    return Sigmoid(z)*(1-Sigmoid(z))


if __name__ == '__main__':

	trainingData, validationData, testingData = mnist_loader.load_data_wrapper()
	trainingData = list(trainingData)
	testingData = list(testingData)

	network = Network([784,200, 10])
	
	network.Train(trainingData, 10000, 1000, 1.5, testingData)

