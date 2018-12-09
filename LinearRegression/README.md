# Linear Regression

Linear regression is one of the simplest statistical models, and in this tutorial, we will go over two ways to find a line that represents our data. We will be using a randomly generated dataset that somewhat resembles a line. This can be done by randomly selecting from a Gaussian distribution plus the current x value. This data is not meant to represent anything in particular, it is just an easy to generate data that has some variability in it.

<strong>Line of Best Fit</strong>
The line of best fit is the line with the smallest error given its cost function. Remember a cost function maps an expected and the actual value of a “cost” or error. </span>So our goal is to determine a line that best represents our data, but how do we measure that. A common way of doing this using a method call Mean Square Error. This is just the summation of the quadratic loss divided by the number data points.

Where y' is the actual output and y is the estimated output.

<strong>Gradient Descent</strong>
We are going to bring back an old friend from our previous tutorial, gradient descent. By using this iterative process, we can approximate the line of best fit. Because remember, gradient descent allows us to minimize a function, and in our case, that will be our Mean Square Error function. We want to minimize it, and we can do that by taking the partial derivative with respect to a and b. We can also replace y with xb + a.

Then by iterating over the data points, calculating the results and summing them together, we can determine the new a and b. Below is a graph how line changes after epoch.

<strong>Least Square Method</strong>
So now that we have gone over the iterative process, let's discuss how we can solve this problem outright. We can do this by using a method call the Least Square.

This will allow us to find the line of best fit where the mean square error is at its smallest.

<strong>Results</strong>
<p class="p1"><span class="s1">Least Square: 4918.9</span></p>
<p class="p1"><span class="s1">Least Square Gradient Descent: 5690.9 [Epoch: 20 | Learning Rate: .001]</span></p>
Even though Least Square scores significantly better, Gradient Descent can usually reach similar performance with enough epochs (iterations over the data).

<strong>Conclusion</strong>
Linear Regression is an easy model to implement and allows us to get a better understanding of the data. One of the issues with Linear Regression is that it is not a very powerful modeling tool; it just tries to draw a straight line through the data. However, in general, simpler models tend to work better, so don't forget to try them out. To see the full code, please check out my GitHub.


<strong>Further Reading &amp; Other Tutorials</strong>
<ul>
 	<li class="p1"><a href="http://blog.hackerearth.com/gradient-descent-algorithm-linear-regression"><span class="s1">[Rashmi Jain] </span>Gradient descent algorithm for linear regression</a></li>
 	<li><a href="https://machinelearningmastery.com/linear-regression-tutorial-using-gradient-descent-for-machine-learning/">[Jason Brownlee] Linear Regression Tutorial Using Gradient Descent for Machine Learning</a></li>
 	<li><a href="https://en.wikipedia.org/wiki/Linear_regression">[Wikipedia] Linear Regression</a></li>
 	<li><a href="https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/">[Matt Nedrich] An Introduction to Gradient Descent and Linear Regression</a></li>
 	<li><a href="https://en.wikipedia.org/wiki/Least_squares">[Wikipedia] Least Squares</a></li>
</ul>

