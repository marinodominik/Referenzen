import math
import numpy as np
from scipy import ndimage


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """
  m, n = fsize[0], fsize[1]
  X = np.linspace(-np.floor(m/2), np.floor(m/2), m)

  gauss = np.asarray([np.exp(- (x ** 2) / (2 * sigma ** 2)) for x in X]) / (np.sqrt(2 * np.pi) * sigma)

  return np.asarray(gauss) / np.sum(gauss)


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """
  h = 1
 
  dx = np.asarray([-1, 0, 1]) / (2 * h)
  gx = gauss2d(0.9, (3, 1))

  fx = np.outer(gx, dx)
  fy = np.outer(dx.reshape(1, 3), gx.reshape(3, 1))

  return fx, fy



def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """
  Ix = ndimage.convolve(I, fx)
  Iy = ndimage.convolve(I, fy)

  return Ix, Iy


def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

  edges = np.sqrt(Ix ** 2 + Iy ** 2)
  edges[edges < thr] = 0

  return edges


def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """
  row, column = edges.shape
  edges2 = np.copy(edges)
  theta_matrix = np.rad2deg(np.arctan(Iy / Ix))

  for i in range(1, row - 1):
    for j in range(1, column - 1):
      theta = theta_matrix[i, j]

      # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]
      if((-90 <= theta <= -67.5) or (67.5 < theta <= 90)):
        value = max(edges[i - 1, j], edges[i + 1, j])
        if(edges[i, j] < value):
          edges2[i, j] = 0

      # handle left-to-right edges: theta in (-22.5, 22.5]
      if (-22.5 < theta <= 22.5):
        value = max(edges[i, j - 1], edges[i, j + 1])
        if (edges[i, j] < value):
          edges2[i, j] = 0

      # handle bottomleft-to-topright edges: theta in (22.5, 67.5]
      if(22.5 < theta <= 67.5):
        value = max(edges[i - 1, j - 1], edges[i + 1, j + 1])
        if (edges[i, j] < value):
          edges2[i, j] = 0

      # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]
      if(-67.5 <= theta <= -22.5):
        value = max(edges[i - 1, j], edges[i + 1, j])
        if (edges[i, j] < value):
          edges2[i, j] = 0

  return edges2