import numpy as np
import matplotlib.pyplot as plt
import glob
from time import time
from numpy import fft
import scipy.io

def import_dark_counts_to_byteflipped_mask(dark_counts_name: str,
                                           threshold: int):
    mat_arr = scipy.io.loadmat(dark_counts_name)['c']
    return np.packbits(np.flip(mat_arr.reshape([8, 320*240//8], order='F'), axis=0).reshape([320*240], order='F') < threshold)

def load_counts_bin(counts_bin_name: str):
    return np.fromfile(counts_bin_name, dtype=np.uint8)

def process_counts_with_byteflipped_mask(counts_file_name: str,
                                         byteflipped_mask):
    counts = np.fromfile(counts_file_name, dtype=np.uint8)[3:]
    return (np.unpackbits(counts & np.repeat(
        byteflipped_mask, 10000
    ))).reshape(10000, 320*240, order='C')


if __name__ == '__main__':
    start = time()
    mask = import_dark_counts_to_byteflipped_mask('./bin1.mat', 1)
    mat = scipy.io.loadmat('./bin1.mat')['c']
    processed = process_counts_with_byteflipped_mask("./spc_dark21.bin", mask)
    print(processed.shape)
    processed = processed[processed.sum(axis=1)==2, :]
    processed = np.flip(processed.reshape(processed.shape[0], 76800//8, 8), axis=2).reshape(processed.shape[0], 240, 320)
    print(np.argwhere(processed!=0))
    print(time()-start)
