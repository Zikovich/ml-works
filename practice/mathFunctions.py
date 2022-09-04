import numpy as np

# Define the activation function.
sigma = np.tanh
# Let's use a random initial weight and bias.
W = np.array([[-0.94529712, -0.2667356 , -0.91219181], [ 2.05529992, 1.21797092, 0.22914497]])
b = np.array([ 0.61273249, 1.6422662 ])
# define our feed forward function
def a1 (a0) :
    # Notice the next line is almost the same as previously,
    # except we are using matrix multiplication rather than scalar multiplication
    # hence the '@' operator, and not the '*' operator.
    z = W @ a0 + b
    # Everything else is the same though,
    return sigma(z)
# Next, if a training example is,
x = np.array([0.7, 0.6, 0.2])
y = np.array([0.9, 0.6])
# Then the cost function is,
d = a1(x) - y
# Vector difference between observed and expected activation
C = d @ d
# Absolute value squared of the difference.


# Newton-rapshon    
from scipy import optimize

def f (x) :
  return x**6/6 - 3*x**4 - 2*x**3/3 + 27*x**2/2 + 18*x - 30
  
x0 = 3.1
optimize.newton(f, x0)

### implementation of Netwon-raphson
def f (x) :
  return x**6/6 - 3*x**4 - 2*x**3/3 + 27*x**2/2 + 18*x - 30

def d_f (x) :
  return x**5 - 12 * x **3 - 2*x ** 2 + 27*x + 18 # Complete this line with the derivative you have calculated.

x = 3.1

d = {"x" : [x], "f(x)": [f(x)]}
for i in range(0, 20):
  x = x - f(x) / d_f(x)
  d["x"].append(x)
  d["f(x)"].append(f(x))

pd.DataFrame(d, columns=['x', 'f(x)'])

