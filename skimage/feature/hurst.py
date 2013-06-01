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
from skimage.morphology import disk
from skimage import data

def rings(w,n):
    # build n concentric rings inside a 2*w x 2*w image
    x = np.arange(-w, w+1, 1)
    y = np.arange(-w, w+1, 1)
    xv, yv = np.meshgrid(x, y)
    z = np.sqrt(xv**2+yv**2)
    r = np.linspace(0,w,n,endpoint=True)[1:]
    delta = r[0]
    elem = []
    for r1 in r:
        print r1
        elem.append((np.logical_and(z>=r1-delta,z<=r1)).astype(np.uint8))
    return (r,elem)

def filter(ima):
    x,elem = rings(30,7)
    plt.figure()
    plt.imshow(np.sum(elem*x[:,np.newaxis,np.newaxis],0),interpolation='nearest')
    x = np.log(x)
    n = x.shape[0]
    Y = np.ones((ima.shape[0],ima.shape[1],n))
    for i,e in enumerate(elem):
        f = rank.percentile_gradient(d,e,p0=.1,p1=.9)
        # f = rank.gradient(d,e)
        Y[:,:,i] = np.log(f)

    # line fitting cf. http://www.johndcook.com/blog/2008/10/20/comparing-two-ways-to-fit-a-line-to-data/
    n = x.shape[0]
    sx = x.sum()
    sy = Y.sum(axis=2)

    stt = np.zeros(ima.shape)
    sts = np.zeros(ima.shape)

    for i in range(n):
        print i
        t = x[i] - sx/n
        stt += t*t
        sts += t*Y[:,:,i]

    slope = sts/stt
    intercept = (sy - sx*slope)/n

    plt.figure()
    plt.imshow(slope)
    plt.colorbar()
    for i in range(Y.shape[2]):
        ima = Y[:,:,i]
        plt.figure()
        plt.imshow(ima,interpolation='nearest')
    plt.figure()
    for i in range(10):
        plt.plot(x,Y[10*i,10*i,:])
        plt.text(x[0],Y[10*i,10*i,0],'%.2f'%slope[10*i,10*i])
    return slope

d = data.camera()
#d = plt.imread('ims_broadatz.jpg')
f = filter(d)
plt.figure()
plt.imshow(d,cmap=plt.cm.gray)
plt.show()
