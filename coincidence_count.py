# First version of this file has no more complication optimisations
# These include:
#
# reversing the endianness of the mask bits
#
# having the mask bits packed into bytes, before
# performing & operation with data

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from time import time
from pathlib import Path
from functools import reduce, partial
from multiprocessing import Pool

def load_mask(mask_location: str, threshold: int, crop=None):
    # Loads a mask in matlab format from the file at mask_location
    # Remove pixels that have a dark count >= threshold
    # If the mask need to be cropped to a smaller frame,
    # use crop = [(row_min, col_min), (n_rows, n_cols)]
    if crop is None:
        mat_arr = scipy.io.loadmat(mask_location)['c']
    else:
        mat_arr = scipy.io.loadmat(mask_location)['c'][
            crop[0][0]:crop[0][0]+crop[1][0],
            crop[0][1]:crop[0][1]+crop[1][1]]
    return (mat_arr.flatten('F') < threshold)


def load_counts_bin(counts_location: str, frame_dim, crop=None):
    # Loads a binary file stored from the camera.
    # Removes the first 3 bytes.
    # Crops the frame if a crop is given
    # Returns a (n_frames, n_pixels)-dimensioned matrix
    wrong_bits = np.unpackbits(np.fromfile(counts_location,
                                           dtype=np.uint8)[3:])
    bits = wrong_bits.reshape((-1, 8))[:, ::-1]
    if crop is not None:
        return bits.reshape((-1, frame_dim[0], frame_dim[1]))[
            :,
            crop[0][0]:crop[0][0]+crop[1][0],
            crop[0][1]:crop[0][1]+crop[1][1]
        ].reshape((-1, crop[1][0]*crop[1][1]))
    else:
        return bits.reshape((-1, frame_dim[0]*frame_dim[1]))


def find_bins_in_folder(folder_name: str, file_prefix: str = "spc_data"):
    return list(map(str, Path(folder_name).glob(
        "{}*.bin".format(file_prefix)
    )))

def process_file(mask, crop, file_name):
    counts = load_counts_bin(file_name, (320, 240), crop)
    masked = mask & counts
    filtered = masked[masked.sum(axis=1) == 2, :]
    return np.argwhere(filtered != 0)

def process_file_timed(mask, crop, number_of_files, iter_file_tup):
    iteration = iter_file_tup[0]
    file_name = iter_file_tup[1]
    start_time = time()
    coords = process_file(mask, crop, file_name)
    print("Done {}/{}\tIteration time: {:.3f}".format(iteration+1,
                                                      number_of_files,
                                                      time()-start_time))
    return coords

if __name__ == '__main__':
    coords = []
    crop = ((170, 60), (104, 104))
    files = find_bins_in_folder("../2018_9_6_17_19_2")
    number_of_files = len(files)
    mask = load_mask("../2018_9_6_17_19_2/p.mat", 1000, crop)
    last_time = time()
    counts_image = np.zeros((crop[1]))
    with Pool(6) as p:
        processor = partial(process_file_timed, mask, crop, number_of_files)
        p.map(processor, enumerate(files[:]))
#    for iteration, file_name in enumerate(files[:2]):
#        counts = load_counts_bin(file_name, (320, 240), crop)
#        masked = mask & counts
#        filtered = masked[masked.sum(axis=1) == 2, :]
#        coords.append(np.argwhere(filtered != 0))
#        counts_image += masked.sum(axis=0).reshape(counts_image.shape, order='F')
#        new_time = time()
#        print("Done {}/{}\tIteration time: {:.3f}".format(iteration+1,
#                                                          number_of_files,
#                                                          new_time-last_time))
#        last_time = new_time
#    print(coords)
#    plt.imshow(counts_image)
#    plt.show()
