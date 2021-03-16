from functools import partial
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.ndimage import convolve

####################
# Provided functions
####################


conv2d = partial(convolve, mode="mirror")


def gauss2d(fsize, sigma):
    """ Create a 2D Gaussian filter

    Args:
        fsize: (w, h) dimensions of the filter
        sigma: wIth of the Gaussian filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def downsample(img, fsize=(5, 5), sigma=1.4):
    """
    Downsampling an image by a factor of 2

    Args:
        img: image as (h, w) np.array
        fsize and sigma: parameters for Gaussian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as (h/2, w/2) np.array
    """
    g_k = gauss2d(fsize, sigma)
    img = conv2d(img, g_k)
    return img[::2, ::2]


def gaussian_pyramid(img, nlevels=3, fsize=(5, 5), sigma=1.4):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        fsize: gaussian kernel size
        sigma: sigma of gaussian kernel

    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    pyramid = [img]
    for i in range(0, nlevels - 1):
        pyramid.append(downsample(pyramid[i], fsize, sigma))

    return pyramid


def resize(arr, shape):
    """ Resize an image to target shape

    Args:
        arr: image as (h, w) np.array
        shape: target size (h', w') as tuple

    Returns:
        resized image as (h', w') np.array
    """
    return np.array(Image.fromarray(arr).resize(shape[::-1]))


######################
# Basic Lucas-Kanade #
######################


def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives

    Args:
        im1: first image as (h, w) np.array
        im2: second image as (h, w) np.array

    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
                    as (h, w) np.array
    """

    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    Ix = conv2d(im1, fx)
    Iy = conv2d(im1, fy)
    It = im2 - im1
    return Ix, Iy, It


def compute_motion(Ix, Iy, It, patch_size=15):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t each as (h, w) np.array
        patch_size: specifies the side of the square region R in Eq. (1)
    Returns:
        u: optical flow in x direction as (h, w) np.array
        v: optical flow in y direction as (h, w) np.array
    """


    patch = np.ones((patch_size, patch_size))


    det = conv2d(Ix ** 2,  patch) * conv2d(Iy ** 2,  patch) - conv2d((Ix * Iy),  patch) ** 2
    det[det == 0] = np.inf
    adj11 = conv2d(Iy ** 2,  patch)
    adj12 = -conv2d((Ix * Iy),  patch)
    adj22 = conv2d(Ix ** 2, patch)

    u = (adj11 * -conv2d(Ix * It, patch) + adj12 * -conv2d(Iy * It, patch)) / det
    v = (adj12 * -conv2d(Ix * It, patch) + adj22 * -conv2d(Iy * It, patch)) / det
    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        im: input image as (h, w) np.array
        u, v: optical flow in x and y direction each as (h, w) np.array

    Returns:
        im_warp: warped image as (h, w) np.array
    """

    imx,imy = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    points = np.concatenate((imx.reshape(im.shape[0]*im.shape[1], 1), imy.reshape(im.shape[0]*im.shape[1], 1)), axis=1)
    imx_ = imx + v
    imy_ = imy + u
    a = interpolate.griddata(points, im.flatten(), (imx_, imy_), method='linear', fill_value=0)
    return a
    #
    # You code here
    #


def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade.
    Args:
        im1, im2: Images as (h, w) np.array

    Returns:
        Cost as float scalar
    """

    return np.sum((im2 - im1) ** 2)/(im1.shape[0]*im1.shape[1])


###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(pyramid1, pyramid2, n_iter=10):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.

    Args:
        pyramid1, pyramid2: Gaussian pyramids corresponding to
                            im1 and im2, in fine to coarse order
        n_iter: number of refinement iterations

    Returns:
        u: OF in x direction as np.array
        v: OF in y direction as np.array
    """

    # img1 = pyramid1[-1]
    # img2 = pyramid2[-1]
    #
    # for i in range(len(pyramid1) - 1, 0, -1):
    #     for j in range(n_iter):
    #         Ix, Iy, It = compute_derivatives(img1, img2)
    #         u, v = compute_motion(Ix, Iy, It)
    #         img1 = warp(img1, u, v)
    #
    #
    #     img1 = resize(img1, pyramid1[i-1].shape)
    #     u = resize(u, pyramid1[i-1].shape)
    #     v = resize(v, pyramid1[i-1].shape)
    #     img1 = warp(img1, u, v)
    #     img2 = resize(img2, pyramid1[i-1].shape)
    #
    # return u,v
    #
    # You code here
    #

    img1 = pyramid1[-1]
    img2 = pyramid2[-1]
    Ix, Iy, It = compute_derivatives(img1, img2)
    u, v = compute_motion(Ix, Iy, It)
    for i in range(len(pyramid1) - 1, 0, -1):
        for j in range(n_iter):
            img1 = warp(img1, u, v)
            Ix, Iy, It = compute_derivatives(img1, img2)
            u_, v_ = compute_motion(Ix, Iy, It)

            u += u_
            v += v_



        img1 = pyramid1[i-1]
        u = resize(u, pyramid1[i-1].shape)
        v = resize(v, pyramid1[i-1].shape)
        u *= 2
        v *= 2
        img2 = pyramid2[i-1]

    for j in range(n_iter):
        img1 = warp(img1, u, v)
        Ix, Iy, It = compute_derivatives(img1, img2)
        u_, v_ = compute_motion(Ix, Iy, It)

        u += u_
        v += v_


    return u,v



    #
    # You code here
    #

    # img1 = pyramid1[-1]
    # img2 = pyramid2[-1]
    # Ix, Iy, It = compute_derivatives(img1, img2)
    # u, v = compute_motion(Ix, Iy, It)
    #
    #
    # for i in range(len(pyramid1) - 1, 0, -1):
    #     u = resize(u, pyramid1[i - 1].shape)
    #     v = resize(v, pyramid1[i - 1].shape)
    #     u *= 2
    #     v *= 2
    #     img1 = pyramid1[i - 1]
    #     img2 = pyramid2[i - 1]
    #     for j in range(n_iter):
    #
    #         img1 = warp(img1, u, v)
    #         Ix, Iy, It = compute_derivatives(img1, img2)
    #         u_, v_ = compute_motion(Ix, Iy, It)
    #
    #         u += u_
    #         v += v_
    #
    #         #u *= 2
    #         #v *= 2
    #
    #
    #
    #
    #
    # for j in range(n_iter):
    #     img1 = warp(img1, u, v)
    #     Ix, Iy, It = compute_derivatives(img1, img2)
    #     u_, v_ = compute_motion(Ix, Iy, It)
    #
    #     u += u_
    #     v += v_
    #
    #     #u *= 2
    #     #v *= 2
    #
    # return u,v
