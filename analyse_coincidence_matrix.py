# THIS TAKES 2 LONG

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean

matrix = downscale_local_mean(np.load("coincidence_matrix.npz")['arr_0'], (128, 128))
#matrix = np.random.random([10000, 10000])
#print(matrix)

def index_in_square(i, x, y, dx, dy, mat_size):
	ind = np.unravel_index(i, mat_size)
	return (ind[0] >= x) & (ind[0] < x+dx) &(ind[1] >= y) & (ind[1] < y+dy)

# indices = [[], []]
#
# for i in range(matrix.shape[0]):
#     for j in range(i, matrix.shape[1]):
#         if index_in_square(i, 0, 0, 2, 2, (100, 100)) and index_in_square(j, 0, 0, 2, 2, (100, 100)):
#             indices[0].append(i)
#             indices[1].append(j)
# indices = np.indices([9, 9])
#
# print(index_in_square(indices[0], 0, 0, 2, 2, (100, 100)) & index_in_square(indices[1], 0, 0, 2, 2, (100, 100)))
# print(indices[0][index_in_square(indices[0], 0, 0, 2, 2, (100, 100)) & index_in_square(indices[1], 0, 0, 2, 2, (100, 100))])
# print(indices[1][index_in_square(indices[0], 0, 0, 2, 2, (100, 100)) & index_in_square(indices[1], 0, 0, 2, 2, (100, 100))])
# print(matrix[(indices[0][index_in_square(indices[0], 0, 0, 2, 2, (100, 100)) & index_in_square(indices[1], 0, 0, 2, 2, (100, 100))], indices[1][index_in_square(indices[0], 0, 0, 2, 2, (100, 100)) & index_in_square(indices[1], 0, 0, 2, 2, (100, 100))])])

#print(index_in_square(range(0,9), 0, 0, 2, 2, (100, 100)))
#print(matrix[index_in_square(range(0,9), 0, 0, 2, 2, (100, 100)), index_in_square(range(0,9), 0, 0, 2, 2, (100, 100))])


plt.imshow(matrix)
plt.show()
