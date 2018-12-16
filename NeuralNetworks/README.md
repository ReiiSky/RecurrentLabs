# Building a Neural Networks From Scratch

## An implementation of a Neural Network using Numpy.

Neural Networks have become one of the most prevalent machine learning models because of its recent accomplishments in computer vision, speech recognition, and natural language processing. Our next tutorial, I will be breaking down how neural network works as well as how to train them by classifying handwritten numbers. I highly recommend that you use some third-party library such as Numpy for the matrix multiplication which we will be using in this tutorial. Numpy is much faster and gracefully at handling matrices than trying to accomplish this using python's lists.

<strong>The Data</strong>

We will be using the MNIST dataset which is a famous collection of handwritten digits with 50,000 in our training set, 10,000 in the validation, and 10,000 in our test set.&nbsp;Michael Nielsen has built a nice&nbsp;<a href="https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py">wrapper class</a> in python that we will be using.&nbsp;Each dataset is a list of tuples where the first item is the input and second is the handwritten digit value (i.e. 5,6,7).

<strong>Neural Networks</strong>

A Neural Network is a type of architecture&nbsp;inspired by the biological neural networks. These systems learn through labeled examples, adjusting the values connecting the nodes together to reduce the number of the error given a set of inputs and outputs. This all done without specifically telling how the network to interpret the data. An example of this would in image recognition, the system would try to learn to identify images of dogs by being labeled "dog" and "not a dog".

A Neural Network is made up of a collection of units called artificial neurons. A neuron has inputs and typically a single output value that feeds into next the rows. The first row takes in the input from the training data and each data value is tied to a set of connections with their own individual values called weights. These incoming weights that are connected to a neuron are multiplied by the inputs values plus a bias term. This bias term helps make the network flexible in terms of fitting the data and acts like a single neuron output a constant value.

<strong style="font-size: 1.125rem;">Activation Functions</strong>

An activation functional takes the summation above as input and the output becomes the output of the neuron. In our case, we will be using the Sigmond function as the activation function in the neural network. There are different types of activation functions that we can use with each their own pros and cons. We will not be getting into those details this blog.

<b>Backpropagation</b>

Given some input, how do we teach the neural network to recognize a certain digit? We can do this using a technique called backpropagation. Backpropagation allows us to penalize the network when it guesses wrong and have the correction flow from the end to the start. The next question is how do we calculate the how much penalize the network for an incorrect guess? This is established by the lost function that we can set. A&nbsp;loss function represents the calculated "loss" between an expected outcome and the actual outcome. In our case, our network will be using The quadratic loss function we will be using in our case.

<strong>Stochastic Gradient Descent</strong>

Stochastic Gradient Descent (SGD) is an iterative process for stochastic approximation of the gradient descent optimization. Its goal is to find the local maximum or minimum step by step. It is akin to&nbsp;a ball rolling down a hill, we want to find the bottom of the hill because that is where the minimum error (i.e likely where the network has the most correct guesses).

There are some problems with gradient descent, one of the main ones is called the valley problem; where there can multiple local minimus, but the one the network finds is not necessarily the global one. Plus there could be multiple local minimums. The other problem is the vanishing gradient problem. This where the changes in weights that propagates back become so small that barely change the weights. These issues with neural networks will be discussed in another blog post.

<strong>Training</strong>

So how do we use Backpropagation to train the network? We can do this by iterating over each training sample multiple times. We also are reshuffling after iteration to help make the network generalize better. We will break the whole data set into a batch that we train. Breaking the dataset into batches results in the system converging faster to a local minimum, and can be used to save on memory. An epoch is one forward and backward&nbsp;pass over all the data values in the training data. Something to note about the number of epoch configured, the higher number of epochs, the greater risk overfitting on the training data. When training our neural network, our goal is to learn a generalized model of the data so that we can successfully label data that the model hasn't seen before.

<strong>Results</strong>

Given that we can play with the number of layers, learning rate, number of nodes per layer, batch size, and epoch. Here are some of the parameters that I used and its results:
<ul>
 	<li>Number of Epochs: 1000</li>
 	<li>Batch Size: 1000</li>
 	<li>Learning Rate: 3</li>
 	<li>Input Layer: 784 nodes</li>
 	<li>Output: 10 nodes</li>
</ul>

## 9209 out of 1000 correct results

This is nowhere near the top results on the MNIST data set, but it demonstrates how Neural Networks work and learn.

<strong>Conclusion</strong>

Neural Networks are a powerful type of learning architecture that allows it learn complex representations of data without being explicitly told what to look for. Please check out the further reading for more about them and the underlying calculus. To see the full code, please check out my <a href="https://github.com/kuds/NeuralNetworks">GitHub</a>.

<strong>Further Reading &amp; Other Tutorials</strong>
<ul>
 	<li><a href="http://neuralnetworksanddeeplearning.com/">[Michael Nielsen] Neural Networks and Deep Learning</a></li>
 	<li><a href="https://www.tensorflow.org/get_started/mnist/beginners">[Tensorflow] MNIST For ML Beginners</a></li>
 	<li><a href="http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/">[Standford] Multi-Layer Neural Network</a></li>
 	<li><a href="https://www.youtube.com/watch?v=h3l4qz76JhQ">[Siraj Raval] Build a Neural Net in 4 Minutes</a></li>
 	<li><a href="http://karpathy.github.io/neuralnets/">[Andrej Karpathy] Hacker's guide to Neural Networks</a></li>
 	<li><a href="https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0">[The Theory Of Everything] Understanding Activation Functions in Neural Networks</a></li>
 	<li><a href="https://en.wikipedia.org/wiki/Backpropagation">[Wikipedia] Backpropagation</a></li>
</ul>
