import sys
sys.path.append('/home/ats432/projects/Matsuzaki_Lab/scratch_NLP')
from common.np import *  # import numpy as np
W = np.arange(21).reshape(7, 3)
print(W)
print(W[2])
print(W[5])

idx = np.array([1, 0, 3, 0])
print(W[idx])