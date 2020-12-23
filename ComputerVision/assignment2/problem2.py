import numpy as np
import os
from PIL import Image


def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images.
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        imgs: (N, H, W) numpy array
    """
    imgs = np.ndarray(shape=(1,96,84))
    for root, dirs, files in os.walk(path):
        for name in files:
            im = Image.open(os.path.join(root, name))
            arr = np.asarray(im)
            imgs = np.concatenate((imgs, arr[None, ...]))

    return imgs[1:,:,:]

def vectorize_images(imgs):
    """Turns an  array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images into
    an (N, M) array where M=H*W is the image dimension.
    
    Args:
        imgs: (N, H, W) numpy array
    
    Returns:
        x: (N, M) numpy array
    """
    
    return imgs.reshape((imgs.shape[0],-1))


def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an numpy array with N M-dimensional features
    
    Returns:
        mean_face: (M,) numpy array representing the mean face
        u: (M, M) numpy array, bases with D principal components
        cumul_var: (N, ) numpy array, corresponding cumulative variance
    """
    mean_face = np.mean(X, axis=0)
    Xnew = X - mean_face
    C = 1/X.shape[0] * Xnew @ Xnew.T

    u,s,v = np.linalg.svd(Xnew.T)
    eigenvalues = (s**2)/X.shape[0]
    cumul_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    return mean_face, u, cumul_var

def basis(u, cumul_var, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    Answer:
    
    Args:
        u: (M, M) numpy array containing principal components.
        For example, i'th vector is u[:, i]
        cumul_var: (N, ) numpy array, variance along the principal components.
    
    Returns:
        v: (M, D) numpy array, contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """

    i = np.argmax(cumul_var > p)
    return u[:,:i]

def compute_coefficients(face_image, mean_face, u):
    """Computes the coefficients of the face image with respect to
    the principal components u after projection.
    
    Args:
        face_image: (M, ) numpy array (M=h*w) of the face image a vector
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        a: (D, ) numpy array, containing the coefficients
    """

    return u.T @ (face_image - mean_face)


def reconstruct_image(a, mean_face, u):
    """Reconstructs the face image with respect to
    the first D principal components u.
    
    Args:
        a: (D, ) numpy array containings the image coefficients w.r.t
        the principal components u
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (M, ) numpy array, projected vector of face_image on 
        principal components
    """
    sum = 0
    for i in range(0, a.shape[0]):
        sum += a[i] * u[:,i]
    return sum + mean_face

def compute_similarity(Y, x, u, mean_face):
    """Compute the similarity of an image x to the images in Y
    based on the cosine similarity.

    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) image we would like to retrieve
        u: (M, D) bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector

    Returns:
        sim: (N, ) numpy array containing the cosine similarity values
    """

    sim = []
    a_x = compute_coefficients(x, mean_face, u)

    for img in Y:
        a_y = compute_coefficients(img, mean_face, u)
        cosine = (a_x.T @ a_y) / (np.absolute(a_x) @ np.absolute(a_y))
        sim.append(cosine)

    return np.asarray(sim)


def search(Y, x, u, mean_face, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) numpy array, image we would like to retrieve
        u: (M, D) numpy arrray, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M) numpy array containing the top_n most similar images
        sorted by similarity
    """

    similarity = compute_similarity(Y,x,u,mean_face)
    orderedIndi = similarity.argsort()
    ordered = Y[orderedIndi[::-1]]
    return ordered[:top_n,:]


def interpolate(x1, x2, u, mean_face, n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (M, ) numpy array, the first image
        x2: (M, ) numpy array, the second image
        u: (M, D) numpy array, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        n: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate n equally-spaced points on a line
    
    Returns:
        Y: (n, M) numpy arrray, interpolated results.
        The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """
    coe1 = compute_coefficients(x1,mean_face,u)
    coe2 = compute_coefficients(x2,mean_face,u)

    interpol = np.linspace(coe1[0], coe2[0], n)

    for i, item in enumerate(coe1):
        print()
        interpol = np.dstack((interpol, np.linspace(item,coe2[i],n)))

    interpol = interpol[0,:,1:]
    Y = np.ndarray(shape=(1,x1.shape[0]))
    for i in range(0, n):
        Y = np.concatenate((Y, reconstruct_image(interpol[i],mean_face,u)[None,...]))

    return Y[1:,:]