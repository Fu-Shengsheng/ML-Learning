if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
from dezero import Variable
from dezero.utils import plot_dot_graph

def my_sin(x, treshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y += t
        if abs(t.data) < treshold:
            break
    
    return y

x = Variable(np.array(np.pi/4))
y = my_sin(x)
y.backward()

print(y)
print(x.grad)

x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=True, to_file='my_sin.png')