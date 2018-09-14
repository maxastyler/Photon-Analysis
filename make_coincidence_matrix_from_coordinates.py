import numpy as np
coords = np.load("coincidence_coordinates.npz")['arr_0']
coincidence_matrix = np.zeros([320*240, 320*240], dtype=np.uint8)
for frame in coords:
    coincidence_matrix[
        tuple(frame[:, 1].reshape(2, frame.shape[0]//2))
    ] += 1
#np.savez_compressed('./coincidence_matrix.npz', coincidence_matrix)
