import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
    counts_mat &= np.tile(byteflipped_mask, 10000)
    counts_mat = np.unpackbits(counts_mat, axis=0)
    return counts_mat.reshape(10000, 320*240, order='C')

def find_bins_in_folder(folder_name: str, file_prefix: str = "spc_data"):
    return list(map(str, Path(folder_name).glob("{}*.bin".format(file_prefix))))


if __name__ == '__main__':
    coords = []
    # file_names = find_bins_in_folder("../2018_9_6_17_19_2")[0:10]
    file_names = ["./spc_data1.bin"]
    number_of_files = len(file_names)
    full_image = np.zeros(320*240)
    last_time = time()
    mask = import_dark_counts_to_byteflipped_mask(
        './p.mat', 1000)
    for iteration, file_name in enumerate(file_names):
        processed = process_counts_with_byteflipped_mask(file_name, mask)
        full_image += processed.sum(axis=0)
        processed = processed[processed.sum(axis=1) > 1, :]
        processed = np.flip(processed.reshape(processed.shape[0], 76800//8, 8),
                            axis=2).reshape(processed.shape[0], 240, 320)
        coords.append(np.argwhere(processed != 0))
        new_time = time()
        print("Done {}/{}\tIteration time: {:.3f}".format(iteration+1, number_of_files, new_time-last_time))
        last_time = new_time
    print(coords)
    # plt.imshow(full_image.reshape([320, 240], order='F'))
    # plt.imshow(mask, order='F')
    mat_arr = scipy.io.loadmat("p.mat")['c']
    full_image = full_image.reshape([320, 240], order='F')
    # plt.subplot(121)
    # plt.imshow(mat_arr, norm=colors.LogNorm(vmin=mat_arr.min(), vmax=mat_arr.max()))
    # plt.imshow(mat_arr)
    plt.subplot(121)
    plt.imshow(np.unpackbits(mask).reshape([320, 240], order='F'))
    # plt.colorbar()
    # mat_arr = np.flip(mat_arr.reshape([8, 320*240//8], order='F'), axis=0).reshape([320,240], order='F')
    plt.subplot(122)
    plt.imshow(full_image)#, norm=colors.LogNorm(vmin=full_image.min()+np.finfo(float).eps, vmax=mat_arr.max()))
    plt.colorbar()
    plt.show()
