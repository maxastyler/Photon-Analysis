import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from time import time
import scipy.io
from pathlib import Path
from functools import reduce

def import_dark_counts_to_byteflipped_mask(dark_counts_name: str,
                                           threshold: int):
    mat_arr = scipy.io.loadmat(dark_counts_name)['c']
    return np.packbits(np.flip(mat_arr.reshape([8, 320*240//8], order='F'),
                               axis=0).reshape([320*240], order='F')
                       < threshold)


def load_counts_bin(counts_bin_name: str):
    return np.fromfile(counts_bin_name, dtype=np.uint8)

def process_counts_with_byteflipped_mask(counts_file_name: str,
                                         byteflipped_mask):
    counts_mat = np.fromfile(counts_file_name, dtype=np.uint8)[3:]
    counts_mat &= np.tile(byteflipped_mask, 10000)
    counts_mat = np.unpackbits(counts_mat, axis=0)
    return counts_mat.reshape(10000, 320*240, order='C')

def find_bins_in_folder(folder_name: str, file_prefix: str = "spc_data"):
    return list(map(str, Path(folder_name).glob("{}*.bin".format(file_prefix))))

def find_two_point_coincidences(masked_frames):
    # Takes a (n_frames, n_pixels) matrix and returns the byteflipped frames with 2 counts as a (n_two_counts, n_pixels) matrix
    processed = masked_frames[masked_frames.sum(axis=1) == 2, :]
    return np.flip(processed.reshape(processed.shape[0], 76800//8, 8),
                        axis=2).reshape(processed.shape[0], 240*320)


if __name__ == '__main__':
    coords = []
    # file_names = find_bins_in_folder("../2018_9_6_17_19_2")[0:10]
    file_names = ["./spc_data1.bin"]
    number_of_files = len(file_names)
    last_time = time()
    mask = import_dark_counts_to_byteflipped_mask('./p.mat', 1000)
    frame_size = reduce(lambda x, y: x*y, mask.shape, 1)*8
    for iteration, file_name in enumerate(file_names):
        processed = process_counts_with_byteflipped_mask(file_name, mask)
        processed = find_two_point_coincidences(processed)
        coords.append(np.argwhere(processed != 0))
        new_time = time()
        print("Done {}/{}\tIteration time: {:.3f}".format(iteration+1,
                                                          number_of_files,
                                                          new_time-last_time))
        last_time = new_time
    print(coords)
    coincidence_matrix = np.zeros([frame_size, frame_size], dtype=np.uint8)
    for frame in coords:
        coincidence_matrix[
            tuple(frame[:, 1].reshape(2, frame.shape[0]//2))
        ] += 1
    np.savez_compressed('/tmp/test_mat.npz', coincidence_matrix)
