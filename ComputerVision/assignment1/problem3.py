import numpy as np
import matplotlib.pyplot as plt


# Plot 2D points
def displaypoints2d(points):
    plt.figure(0)
    plt.plot(points[0, :], points[1, :], '.b')
    plt.xlabel('Screen X')
    plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0, :], points[1, :], points[2, :], 'b')
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")


def cart2hom(points):
    """ Transforms from cartesian to homogeneous coordinates.

  Args:
    points: a np array of points in cartesian coordinates

  Returns:
    points_hom: a np array of points in homogeneous coordinates
  """

    return np.append(points, np.ones((1, points.shape[1])), axis=0)


def hom2cart(points):
    """ Transforms from homogeneous to cartesian coordinates.

  Args:
    points: a np array of points in homogenous coordinates

  Returns:
    points_hom: a np array of points in cartesian coordinates
  """

    return (points/points[len(points)-1])[:len(points)-1]


def gettranslation(v):
    """ Returns translation matrix T in homogeneous coordinates for translation by v.

  Args:
    v: 3d translation vector

  Returns:
    T: translation matrix in homogeneous coordinates
  """

    T = np.zeros((v.shape[0] + 1, v.shape[0] + 1))
    np.fill_diagonal(T, 1)

    T[:v.shape[0], v.shape[0]] = v

    return T


def getxrotation(d):
    """ Returns rotation matrix Rx in homogeneous coordinates for a rotation of d degrees around the x axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rx: rotation matrix
  """
    rads = np.radians(d)
    T = np.zeros((4, 4))
    np.fill_diagonal(T, 1)

    T[1:3, 1:3] = np.array([[np.cos(rads), -np.sin(rads)], [np.sin(rads), np.cos(rads)]])

    return T


def getyrotation(d):
    """ Returns rotation matrix Ry in homogeneous coordinates for a rotation of d degrees around the y axis.

  Args:
    d: degrees of the rotation

  Returns:
    Ry: rotation matrix
  """
    rads = np.radians(d)
    T = np.zeros((4, 4))
    np.fill_diagonal(T, 1)

    T[[0, 0, 2, 2], [0, 2, 0, 2]] = np.array([np.cos(rads), np.sin(rads), -np.sin(rads), np.cos(rads)])

    return T

def getzrotation(d):
    """ Returns rotation matrix Rz in homogeneous coordinates for a rotation of d degrees around the z axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rz: rotation matrix
  """
    rads = np.radians(d)
    T = np.zeros((4, 4))
    np.fill_diagonal(T, 1)

    T[0:2, 0:2] = np.array([[np.cos(rads), -np.sin(rads)], [np.sin(rads), np.cos(rads)]])

    return T

def getcentralprojection(principal, focal):
    """ Returns the (3 x 4) matrix L that projects homogeneous camera coordinates on homogeneous
  image coordinates depending on the principal point and focal length.
  
  Args:
    principal: the principal point, 2d vector
    focal: focal length

  Returns:
    L: central projection matrix
  """

    return np.array([[focal, 0, principal[0], 0],
            [0, focal, principal[1], 0],
            [0, 0, 1, 0]])



def getfullprojection(T, Rx, Ry, Rz, L):
    """ Returns full projection matrix P and full extrinsic transformation matrix M.

  Args:
    T: translation matrix
    Rx: rotation matrix for rotation around the x-axis
    Ry: rotation matrix for rotation around the y-axis
    Rz: rotation matrix for rotation around the z-axis
    L: central projection matrix

  Returns:
    P: projection matrix
    M: matrix that summarizes extrinsic transformations
  """

    M = Rz @ Rx @ Ry @ T
    P = L @ M

    return P, M

def projectpoints(P, X):
    """ Apply full projection matrix P to 3D points X in cartesian coordinates.

  Args:
    P: projection matrix
    X: 3d points in cartesian coordinates

  Returns:
    x: 2d points in cartesian coordinates
  """

    return hom2cart(P @ cart2hom(X))


def loadpoints():
    """ Load 2D points from obj2d.npy.

  Returns:
    x: np array of points loaded from obj2d.npy
  """

    x = np.load('./data/obj2d.npy')
    return x


def loadz():
    """ Load z-coordinates from zs.npy.

  Returns:
    z: np array containing the z-coordinates
  """

    z = np.load('./data/zs.npy')
    return z


def invertprojection(L, P2d, z):
    """
  Invert just the projection L of cartesian image coordinates P2d with z-coordinates z.

  Args:
    L: central projection matrix
    P2d: 2d image coordinates of the projected points
    z: z-components of the homogeneous image coordinates

  Returns:
    P3d: 3d cartesian camera coordinates of the points
  """

    return np.linalg.inv(L[:,0:3]) @ cart2hom(P2d) * z


def inverttransformation(M, P3d):
    """ Invert just the model transformation in homogeneous coordinates
  for the 3D points P3d in cartesian coordinates.

  Args:
    M: matrix summarizing the extrinsic transformations
    P3d: 3d points in cartesian coordinates

  Returns:
    X: 3d points after the extrinsic transformations have been reverted
  """

    return np.linalg.inv(M) @ cart2hom(P3d)


def p3multiplecoice():
    '''
  Change the order of the transformations (translation and rotation).
  Check if they are commutative. Make a comment in your code.
  Return 0, 1 or 2:
  0: The transformations do not commute.
  1: Only rotations commute with each other.
  2: All transformations commute.
  '''

    return 0
