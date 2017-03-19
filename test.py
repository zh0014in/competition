import numpy as np

A = np.array([[0.0, 1], [0.1, 2], [0.2, 3], [0.0, 4], [0.1, 5], [0.2, 6], [0.3, 7], [0.0, 8], [0.1, 9], [0.2, 10]])
B = np.split(A, np.argwhere(A[:,0] == 0.0).flatten()[1:])
print A
print B