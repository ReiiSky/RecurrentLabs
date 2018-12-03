import matplotlib.pyplot
import numpy
import math

def GradientDescent(aCurrent, bCurrent, xDataSet, yDataSet, learningRate):
	aGradient = 0.0
	bGradient = 0.0
	N = float(zip(xDataSet))
	for x,y in zip(xDataSet, yDataSet):
		aGradient += -(2/N) * (y - ((bCurrent*x) + aCurrent))
		bGradient += -(2/N) * x * (y - ((bCurrent * x) + aCurrent))
	aNew = aCurrent - (learningRate * aGradient)
	bNew = bCurrent - (learningRate * bGradient)
	return [aNew, bNew]

def MSE(xDataSet, yDataSet, aCurrent, bCurrent):
	mse = 0.0
	for x,y in zip(xDataSet, yDataSet):
		mse += math.pow((y - (aCurrent + (bCurrent * x))), 2)

	return mse

scale = 8.0
start = 0.0
end = 50.0
increment = .5
epoch = 20

xDataSet = numpy.arange(start,end,increment)
yDataSet = numpy.array([(i+numpy.random.normal(scale=scale)+5) for i in xDataSet])

xAverage = numpy.average(xDataSet)
yAverage = numpy.average(yDataSet)

numerator = 0.0
denominator = 0.0

for x,y in zip(xDataSet, yDataSet):
	numerator += (x - xAverage)*(y - yAverage)
	denominator += math.pow((x - xAverage), 2)

b = (numerator/denominator)
a = yAverage - (xAverage * b)

leastSquares = 0.0
leastSquaresGD = 0.0

y1 = (b * start) + a
y2 = (b * end) + a

#matplotlib.pyplot.clf()
#matplotlib.pyplot.scatter(xDataSet,yDataSet)
#matplotlib.pyplot.plot([start, end],[y1, y2], color='r')
#matplotlib.pyplot.show()
#matplotlib.pyplot.savefig("lse.png")

startingA = numpy.random.normal();
startingB = numpy.random.normal();

print("A: " + str(startingA))
print("B: " + str(startingB))

#matplotlib.pyplot.clf()
#matplotlib.pyplot.plot([start, end],[(startingA + (start * startingB)), (startingA + (end * startingB))], color='g')
#matplotlib.pyplot.scatter(xDataSet,yDataSet)
#matplotlib.pyplot.show()
#matplotlib.pyplot.savefig("sgd" + str(0) + ".png")

for i in range(epoch):
	startingA, startingB = stepGradient(startingA, startingB, xDataSet, yDataSet, .0001)
	#print("A:" + str(startingA))
	#print("B:" + str(startingB))
	#matplotlib.pyplot.clf()
	#matplotlib.pyplot.plot([start, end],[(startingA + (start * startingB)), (startingA + (end * startingB))], color='g')
	#matplotlib.pyplot.scatter(xDataSet,yDataSet)
	#matplotlib.pyplot.show()
	#matplotlib.pyplot.savefig("sgd" + str(i+1) + ".png")

for x,y in zip(xDataSet, yDataSet):
	leastSquares += math.pow((y - (a + (b * x))), 2)
	leastSquaresGD += math.pow((y - (startingA + (startingB * x))), 2)

print("Least Square: " + str(leastSquares))
print("Least Square Gradient Descent: " + str(leastSquaresGD))
