import numpy
from enum import Enum


#Number of filters KK,
#their spatial extent FF,
#the stride SS,
#the amount of zero padding PP.


#Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
#Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
#Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
#Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
#Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)
#Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).

height = 28
width = 28
numberOfChannels = 1

inputArray = height * width * numberOfChannels

numberOfClassification = 10


#Convolutional Layer -> Pooling Layer -> Convolutional Layer -> Pooling Layer -> Dense Layer -> Dense layers

class ActivationFunction(Enum):
  Sigmod = 1

class InputLayer:

  def __init__(self, inputLayer):
    if (len(inputLayer.shape) == 2):
      inputLayer = numpy.expand_dims(inputLayer, axis = 0)

    self.Data = inputLayer
    self.Shape = Shape(inputLayer.shape)

class Shape:
  def __init__(self, shape):

    self.Depth = None
    self.Rows = None
    self.Columns = None

    if(len(shape) == 3):
      self.Depth = shape[0]
      self.Rows = shape[1]
      self.Columns = shape[2]
    elif (len(shape) == 2):
      self.Depth = 1
      self.Rows = shape[0]
      self.Columns = shape[1]
    else:
      raise Exception("The Shape of the Input Layer is not 2 or 3. Actually number of dimensions {0}.".format(len(shape)))


class FullyConnectedLayer:
  def __init__(self, numberOfNeurons):
    self.Weights = None
    self.NumberOfNeurons = numberOfNeurons
    self.Bias = numpy.zeros(self.NumberOfNeurons)
    

  def FeedForward(self, inputLayer):
    if(self.Weights == None):
      self.Weights = numpy.random.rand(self.NumberOfNeurons, inputLayer.Shape.Depth, inputLayer.Shape.Rows, inputLayer.Shape.Columns)

    volumeOutput = numpy.zeros(self.NumberOfNeurons)
    for n in range(self.NumberOfNeurons):
      weightedSum = numpy.dot(self.Weights[n], inputLayer.Data) + self.Bias[n]
      volumeOutput[n] = Activation(weightedSum)

    return volumeOutput

  def BackwardsProgration(self, inputLayer, expectedOutput):
    zs = []
    for n in range(self.NumberOfNeurons):
      weightedSum = numpy.dot(self.Weights[n], inputLayer.Data) + self.Bias[n]
      zs.append(weightedSum)

    sigmoidPrime = SigmoidPrime(zs)
    delta = (volumeOutput - expectedOutput) * sigmoidPrime
    self.Bias = delta
    self.Weights = numpy.dot(delta, inputLayer.Data)


class PoolingLayer:
  def __init__(self, spatialExtent, stride):
    self.SpatialExtent = spatialExtent
    self.Stride = stride

  def FeedForward(self, inputLayer):

      #Validate that the Kernal's rows, Padding, Stride, and Filter's rows align
      #if((inputLayer.Shape.Rows + (self.Padding * 2) - self.KernalSize) % self.Stride != 0):
      #  raise Exception("The Input Layer's row {0} and Padding of {1} does not match the Kernal's rows {2} and Stride {3}.".format(inputLayer.Shape.Rows, self.Padding, self.KernalSize, self.Stride))

      #Validate that the Kernal's columns, Padding, Stride, and Filter's columns align
      #if((inputLayer.Shape.Columns + (self.Padding * 2) - self.KernalSize) % self.Stride != 0):
      #  raise Exception("The Input Layer's columns {0} and Padding of {1} does not match the Kernal's rows {2} and Stride {3}.".format(inputLayer.Shape.Columns, self.Padding, self.KernalSize, self.Stride))

      #W2=(W1−F)/S+1
      #H2=(H1−F)/S+1
      #D2=D1

      #volumeSize = (self.)

      print(inputLayer.Shape.Columns)

      volumeWidth = int(((inputLayer.Shape.Columns - self.SpatialExtent) / self.Stride) + 1)
      volumeHeight = int(((inputLayer.Shape.Rows - self.SpatialExtent) / self.Stride) + 1)
      volumenDepth = inputLayer.Shape.Depth

      volumeOutput = numpy.zeros([volumenDepth, volumeWidth, volumeHeight])
      print("Volume Output Size: " + str(volumeOutput.shape))

      print("Input Layer - Depth: {0} | Rows: {1} | Columns: {2}".format(inputLayer.Shape.Depth, inputLayer.Shape.Rows, inputLayer.Shape.Columns))

      for d in range(inputLayer.Shape.Depth):
        i = 0
        while (((i * self.Stride) + self.SpatialExtent) <= inputLayer.Shape.Rows):
          j = 0
          while (((j * self.Stride) + self.SpatialExtent) <= inputLayer.Shape.Columns):
            #print("D: {0} | I: {1} | J: {2}".format(d,i,j))
            x = (i * self.Stride)
            y = (j * self.Stride)
            volumeOutput[d, i, j] = numpy.amax(inputLayer.Data[d, x:x+self.SpatialExtent, y:y+self.SpatialExtent])
            j += 1
          i += 1
      return (volumeOutput)

  def BackwardsProgration(self, inputLayer, expectedOutput):
    volumeOutput = self.FeedForward(inputLayer)
    delta = volumeOutput - expectedOutput
    self.Bias = delta


class ConvolutionalLayer:
  def __init__(self, activation, depth, filters, kernalSize, padding, stride):
    self.Activation = activation
    self.Stride = stride
    self.Depth = depth
    self.Filters = filters
    self.KernalSize = kernalSize
    self.activation = activation
    self.Padding = padding

    self.__RandomizeWeights()

    self.Bias = numpy.zeros(self.Filters)


  def __RandomizeWeights(self):
    self.Weights = numpy.random.rand(self.Filters, self.KernalSize, self.KernalSize)

  def FeedForward(self, inputLayer):

    #Validate that the Kernal's rows, Padding, Stride, and Filter's rows align
    if((inputLayer.Shape.Rows + (self.Padding * 2) - self.KernalSize) % self.Stride != 0):
      raise Exception("The Input Layer's row {0} and Padding of {1} does not match the Kernal's rows {2} and Stride {3}.".format(inputLayer.Shape.Rows, self.Padding, self.KernalSize, self.Stride))

    #Validate that the Kernal's columns, Padding, Stride, and Filter's columns align
    if((inputLayer.Shape.Columns + (self.Padding * 2) - self.KernalSize) % self.Stride != 0):
      raise Exception("The Input Layer's columns {0} and Padding of {1} does not match the Kernal's rows {2} and Stride {3}.".format(inputLayer.Shape.Columns, self.Padding, self.KernalSize, self.Stride))

    #(W−F+2P)/S+1
    #W - Input Size
    #F - Filter Size
    #P - Zero Padding
    #S - Stride 

    #volumeSize = (self.)

    volumeWidth = int(((inputLayer.Shape.Columns + (self.Padding * 2) - self.KernalSize) / self.Stride) + 1)
    volumeHeight = int(((inputLayer.Shape.Rows + (self.Padding * 2) - self.KernalSize) / self.Stride) + 1)
    volumenDepth = self.Filters

    volumeOutput = numpy.zeros([volumenDepth, volumeWidth, volumeHeight])
    print("Volume Output Size: " + str(volumeOutput.shape))

    if(self.Padding > 0):
      inputLayer = AddPadding(inputLayer, self.Padding)

    print("After Padding")
    print(inputLayer.Data)


    print("Input Layer - Depth: {0} | Rows: {1} | Columns: {2}".format(inputLayer.Shape.Depth, inputLayer.Shape.Rows, inputLayer.Shape.Columns))
    for f in range(self.Filters):
      for d in range(inputLayer.Shape.Depth):
        i = 0
        while (((i * self.Stride) + self.KernalSize) < inputLayer.Shape.Rows + (self.Padding * 2)):
          j = 0
          while (((j * self.Stride) + self.KernalSize) < inputLayer.Shape.Columns + (self.Padding * 2)):
            print("F: {0} | D: {1} | I: {2} | J: {3}".format(f,d,i,j))
            #print(inputLayer.Data[d, (i*self.Stride) : (i*self.Stride) +self.KernalSize, (j * self.Stride) : (j * self.Stride) + self.KernalSize])
            #print(self.Weights[f])
            filterTotal = numpy.sum(numpy.dot(inputLayer.Data[d, (i*self.Stride) : (i*self.Stride) +self.KernalSize, (j * self.Stride) : (j * self.Stride) + self.KernalSize], self.Weights[f])) + self.Bias[f]
            #print(filterTotal)
            volumeOutput[f, i, j] = filterTotal
            j += 1
          i += 1

    volumeOutput = numpy.maximum(volumeOutput, 0, volumeOutput)      
    return (volumeOutput)

  def BackwardsProgration(self, inputLayer, delta):
    print("Method not complete yet")


#### Miscellaneous Functions
def AddPadding(inputLayer, padding):  

  if(len(inputLayer.Data.shape) != 3):
    raise Exception("The Shape of the Input Layer is not 3. Actually number of dimensions {0}.".format(len(inputLayer.Data.shape)))

  print(inputLayer.Data.shape)
  inputLayer.Data = numpy.insert(inputLayer.Data, 0, 0, axis = 2)

  inputLayer.Shape.Columns += 1

  inputLayer.Data = numpy.insert(inputLayer.Data, inputLayer.Shape.Columns, 0, axis = 2)

  inputLayer.Shape.Columns += 1

  inputLayer.Data = numpy.insert(inputLayer.Data, 0, 0, axis = 1)

  inputLayer.Shape.Rows += 1

  inputLayer.Data = numpy.insert(inputLayer.Data, inputLayer.Shape.Rows, 0, axis = 1)

  inputLayer.Shape.Rows += 1

  return inputLayer
 
def Sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+numpy.exp(-z))

def SigmoidPrime(z):
    """Derivative of the sigmoid function."""
    return Sigmoid(z)*(1-Sigmoid(z))

if __name__ == "__main__":

  dummyData = numpy.array([[0, 1, 0, 0, 2], [0, 0, 1, 2, 1], [1, 1, 0, 0 ,0], [2, 2, 0, 1, 0], [0, 1, 1, 0, 1]])
  inputLayer = InputLayer(dummyData)

  print("Dummy Data:")
  print(dummyData)


  convLayer = ConvolutionalLayer(ActivationFunction.Sigmod, 1, 5, 3, 1, 2)
  print("Convolution Layer 1")
  print("Number of Filters: " + str(convLayer.Filters))
  print("Depth: " + str(convLayer.Depth))
  print("Stride: " + str(convLayer.Stride))
  print("Kernal Size: " + str(convLayer.KernalSize))
  print("Padding: " + str(convLayer.Padding))
  print("Activation Function: " + str(convLayer.Activation))

  print()

  print("Weights: ")
  print(convLayer.Weights)

  print()

  convLayer.FeedForward(inputLayer)

  print()

  inputLayer = InputLayer(dummyData)
  dummyData = numpy.array([[0, 1, 0, 0, 2], [0, 0, 1, 2, 1], [1, 1, 0, 0 ,0], [2, 2, 0, 1, 0], [0, 1, 1, 0, 1]])

  poolLayer = PoolingLayer(3, 2)

  print("Pooling Layer 1")
  print("Stride: " + str(poolLayer.Stride))
  print("Spatial Extent: " + str(poolLayer.SpatialExtent))

  print(dummyData)

  print(poolLayer.FeedForward(inputLayer))


'''

conv1 = ConvolutionalLayer(filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

  
# Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=10)

predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}

if mode == tf.estimator.ModeKeys.PREDICT:
  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

# Calculate Loss (for both TRAIN and EVAL modes)
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)

# Configure the Training Op (for TRAIN mode)
if mode == tf.estimator.ModeKeys.TRAIN:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

# Add evaluation metrics (for EVAL mode)
eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
'''