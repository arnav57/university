import numpy as np
from architecture import *

def ReLU(x):
    if x < 0:
        return 0
    else:
        return x

inputs = np.array([0.1,0.2])

print('LAYER TEST')
l1 = inputlayer(2, ReLU)
l1.set_input(inputs)
l1.__summary__()
