# First version of this file has no more complication optimisations
# These include:
#
# reversing the endianness of the mask bits
#
# having the mask bits packed into bytes, before
# performing & operation with data

import numpy as np
import scipy.io

from time import time
from pathlib import Path
from functools import reduce


def load_mask(mask_location: str, threshold: int, crop=None):
    # Loads a mask in matlab format from the file at mask_location
    # Remove pixels that have a dark count >= threshold
    # If the mask need to be cropped to a smaller frame,
    # use crop = [(row_min, col_min), (n_rows, n_cols)]
    if crop is None:
        mat_arr = scipy.io.loadmat(mask_location)['c'].flatten('F')
    else:
        mat_arr = scipy.io.loadmat(mask_location)['c'][
            crop[0][0]:crop[0][0]+crop[1][0],
            crop[0][1]:crop[0][1]+crop[1][1]].flatten('F')
    return (mat_arr < threshold)

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

if __name__ == '__main__':
    crop = ((20, 20), (30, 30))
    mask = load_mask("./p.mat", 1000, crop)
    coords = []
    counts = load_counts_bin("./spc_data1.bin", (320, 240), crop)
    masked = mask & counts
    filtered = masked[masked.sum(axis=1) == 2, :]
    coords.append(np.argwhere(filtered != 0))
    print(coords)
