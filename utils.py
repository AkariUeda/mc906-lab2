import numpy as np

def rms(a, b):
        """ Return the Root Mean Square between two RGB images"""
        pixel_count = a.shape[0] * a.shape[1]
        diffs = np.abs((a - b).flatten())
        values, idxs = np.histogram(diffs, bins=range(257))

        sum_of_squares = sum(value*(idx**2) for idx, value in zip(idxs, values))
        return np.sqrt(sum_of_squares / pixel_count)