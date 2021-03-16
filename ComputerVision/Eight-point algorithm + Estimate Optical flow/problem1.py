import numpy as np
import matplotlib.pyplot as plt



def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2,2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """
    U, D, V = np.linalg.svd(A, full_matrices=True)
    D[2] = 0
    A_hat = U @ (np.diag(D) @ V)

    return A_hat



def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """
    corr_points = len(p1)
    x = p1[:, 0]
    y = p1[:, 1]
    x_ = p2[:, 0]
    y_ = p2[:, 1]

    A = np.zeros((corr_points, 9))
    for i in range(corr_points):
        A[i] = [x[i] * x_[i], y[i] * x_[i], x_[i], x[i] * y_[i], y[i] * y_[i], y_[i], x[i], y[i], 1]

    _, _, V = np.linalg.svd(A)
    f = V[-1, :].reshape((3, 3))
    F = enforce_rank2(f)
    return F




def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """
    U1, T1 = condition_points(p1)
    U2, T2 = condition_points(p2)
    F_ = compute_fundamental(U1, U2)
    F = T2.T @ (F_ @ T1)

    return F




def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """
    n, _ = p1.shape

    hom_points = np.hstack((p1, np.ones((len(p1), 1))))
    l = F @ hom_points.T

    x = l[0]
    y = l[1]
    z = l[2]

    #left border
    X1 = np.zeros(len(p1))
    Y1 = - (z + x * X1) / y

    #right border
    X2 = X1 + img.shape[1] - 1
    Y2 = - (z + x * X2) / y

    return X1, X2, Y1, Y2



def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """
    residual = np.sum(p1 @ (F @ p2.T), 0)
    max_residual = np.max(residual)
    avg_residual = np.mean(residual)
    return max_residual, avg_residual


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """
    u, _, V = np.linalg.svd(F)
    e1 = V[-1].reshape((3, 1))

    _, _, V = np.linalg.svd(F.T)
    e2 = V[-1].reshape((3, 1))

    return e1[:2] / e1[2], e2[:2] / e2[2]
