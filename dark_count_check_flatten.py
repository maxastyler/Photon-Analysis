import numpy as np
import matplotlib.pyplot as plt
from time import time
import scipy.io
import os
from pathlib import Path

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
    counts_mat &= np.repeat(byteflipped_mask, 10000)
    print(counts_mat.shape, counts_mat.dtype)
    counts_mat = np.unpackbits(counts_mat, axis=0)
    return counts_mat.reshape(10000, 320*240, order='C')
    # return (np.unpackbits(np.fromfile(counts_file_name, dtype=np.uint8)[3:] & np.repeat(
    #     byteflipped_mask, 10000
    # ))).reshape(10000, 320*240, order='C')

def find_bins_in_folder(folder_name: str, file_prefix: str = "spc_data"):
    return list(map(str, Path(folder_name).glob("{}*.bin".format(file_prefix))))


if __name__ == '__main__':
    start = time()
    mask = import_dark_counts_to_byteflipped_mask(
        '../2018_9_6_17_19_2/p.mat', 1)
    print(find_bins_in_folder("../2018_9_6_17_19_2")[0])
    processed = process_counts_with_byteflipped_mask(
        find_bins_in_folder("../2018_9_6_17_19_2")[0], mask)
    print(processed.shape)
    processed = processed[processed.sum(axis=1) > 1, :]
    processed = np.flip(processed.reshape(processed.shape[0], 76800//8, 8),
                        axis=2).reshape(processed.shape[0], 240, 320)
    print(np.argwhere(processed != 0))
    print(time()-start)
