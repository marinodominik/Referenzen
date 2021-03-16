import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    return np.load(path)


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    #x[start:stop:step]

    r, g, b = np.zeros(bayerdata.shape), np.zeros(bayerdata.shape), np.zeros(bayerdata.shape)
    row, column = bayerdata.shape

    r[0:row:2, 1:column:2] = bayerdata[0:row:2, 1:column:2]

    g[0:row:2, 0:column:2] = bayerdata[0:row:2, 0:column:2]
    g[1:row:2, 1:column:2] = bayerdata[1:row:2, 1:column:2]

    b[1:row:2, 0:column:2] = bayerdata[1:row:2, 0:row:2]

    return r, g, b



def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    row, column = r.shape
    image = np.zeros((row, column, 3))

    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b

    return np.asarray(image)


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    image = assembleimage(r, g, b)

    # interpolation filter for green value
    g_k = np.asarray([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])     #kernel aus script linarer kernel 0.25 * ([1, 2, 1], [2, 4, 2], [1, 2, 1])

    # Interpolation filter for red and blue values
    rb_k = np.asarray([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])

    image[:, :, 0] = convolve(image[:, :, 0], rb_k, mode="reflect")   #vllt mirror als mode
    image[:, :, 1] = convolve(image[:, :, 1], g_k, mode="reflect")
    image[:, :, 2] = convolve(image[:, :, 2], rb_k, mode="reflect")

    return np.asarray(image)