"""
Methods to characterize image Hurst textures.
"""

import numpy as np
from skimage.filter import rank

def hurst_exponent(image):
    """Calculate the Hurst exponent.

    Hurst exponent [1] relates to the local fractal dimension of the texture.

    Parameters
    ----------
    image : array_like of uint8
        Integer typed input image. The image will be cast to uint8, so
        the maximum value must be less than 256.


    Returns
    -------
    H : image of the Hurst exponent for each pixel.

    References
    ----------

    .. [1] Wikipedia, http://en.wikipedia.org/wiki/Co-occurrence_matrix


    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = greycomatrix(image, [1], [0, np.pi/2], levels=4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1]
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)

    """

    H = image

    return H

import matplotlib.pyplot as plt
from skimage import data

def rings(w,n):
    # build n concentric rings inside a 2*w x 2*w image
    x = np.arange(-w, w+1, 1)
    y = np.arange(-w, w+1, 1)
    xv, yv = np.meshgrid(x, y)
    z = np.sqrt(xv**2+yv**2)
    r = np.linspace(0,w,n,endpoint=True)
    delta = r[1]
    print r
    elem = []
    for r1 in r[1:]:
        print r1
        elem.append(np.logical_and(z>=r1-delta,z<=r1))
    return elem

r = rings(10,4)
d = data.camera()

for e in r:
    f = rank.mean(d,e)
    plt.imshow(f,interpolation='nearest')
    plt.show()