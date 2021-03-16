import numpy as np
import matplotlib.pyplot as plt


def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """

    plt.figure()
    plt.imshow(img)
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    return np.load(path)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    return np.flip(img, axis=1)


def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Horizontally stacked subplots")
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()
