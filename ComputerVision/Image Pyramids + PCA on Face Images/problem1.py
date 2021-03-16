from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def loadimg(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """
    return np.asarray(Image.open(path)) / 255.0


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """
    m, n = fsize[0], fsize[1]

    def calc_gaus_kernel(sigma, size):
        X = np.linspace(-np.floor(size / 2), np.floor(size / 2), size)
        g = np.asarray([np.exp(- (x ** 2) / (2 * sigma ** 2)) for x in X]) / (np.sqrt(2 * np.pi) * sigma)
        return (g / np.sum(g)).reshape((1, size))

    gk = calc_gaus_kernel(sigma, m)
    gauss = np.outer(gk, gk.reshape((m, 1)))

    return gauss



def binomial2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """
    m, n = fsize[0], fsize[1]

    bk = np.asarray([binom(n - 1, k) for k in range(0, n)])
    binomial = np.outer(bk, bk.reshape((n, 1)))

    return binomial / np.sum(binomial)



def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """
    row, column = img.shape

    img_filterd = convolve(img, f, mode='mirror')
    img_filterd = img_filterd[0:row:2, 0:column:2]

    return img_filterd




def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """
    m, n = img.shape
    zoom = np.zeros((2 * m, 2 * n))
    zoom[0:2*m:2, 0:2*n:2] = img

    return 4 * convolve(zoom, f, mode='mirror')



def gaussianpyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    gpyramid = []
    gpyramid.append(img)

    for i in range(1, nlevel):
        zoom = downsample2(img, f)
        gpyramid.append(zoom)
        img = zoom

    return np.asarray(gpyramid)


def laplacianpyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    What is the difference between the top (coarsest/gröbste) level of Gaussian and Laplacian pyramids?
    Answer: Nothing, the top level for the laplacian and the gaussian pyramid is the same image

    (ausgehend vom kleinsten bild)
    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    length = len(gpyramid) - 1

    lapPyramid = []
    lapPyramid.append(gpyramid[length])

    for i in range(length - 1, -1, -1):
        image = gpyramid[i] - upsample2(gpyramid[i + 1], f)
        lapPyramid.append(image)

    return np.asarray(lapPyramid)[::-1]


def reconstructimage(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """
    length = len(lpyramid) - 1

    lapPyramid = []
    img = lpyramid[length]
    lapPyramid.append(img)

    for i in range(length - 1, -1, -1):
        img = lpyramid[i] + upsample2(img, f)
        lapPyramid.append(img)

    return lapPyramid[length]



def amplifyhighfreq(lpyramid, l0_factor=1.4, l1_factor=1.6):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    factors do a more blurred version of the image, but increasing makes it low contrast

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid

    """
    images = deepcopy(lpyramid)

    images[0] *= l0_factor
    images[1] *= l1_factor

    return images


def createcompositeimage(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """
    def normalize_img(img):
        min_value, max_value = min(img.flatten()), max(img.flatten())
        return (img - min_value) / (max_value - min_value)

    row, column = pyramid[0].shape
    images = pyramid[0]

    for idx in range(1, len(pyramid)):
        img = pyramid[idx]
        img = normalize_img(img)
        m, n = img.shape
        z = np.zeros((row, n))
        z[0:m, 0:n] = img
        images = np.hstack((images, z))

    return images