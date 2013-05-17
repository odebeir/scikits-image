#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for computing seam lines in an image

To be done...

:author:

:license: modified BSD

References
----------
.. [1] http://en.wikipedia.org/wiki/Seam_carving
"""

import numpy as np
from scipy import ndimage
from scipy import stats
from skimage.color import rgb2grey
from skimage.util import img_as_float

def compute_scores(img):
    """computes seams of an image following axis 0 from 0 to img.shape[0]
        returns scores array and path array

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    cost : ndarray
        cost matrix
    direction : ndarray
        direction to the previous pixel (contains -1,0,1 for each pixel)
    """

    m,n = img.shape

    img = img

    d = np.dstack((img[:,1:-1],img[:,0:-2],img[:,2:]))
    direction = np.argmax(d,axis=2)
    direction = np.hstack((np.zeros((m,1)),direction,np.zeros((m,1))))
    scores = np.zeros_like(img,dtype=int)
    scores[0,:] = img[0,:]
    for i in range(1,m):
        id0 = direction[i-1,:]==0
        id1 = direction[i-1,:]==1
        id2 = direction[i-1,:]==2

        scores[i,id0] = scores[i-1,id0] + img[i,id0]
        scores[i,id1] = scores[i-1,np.roll(id1,-1)] + img[i,id1]
        scores[i,id2] = scores[i-1,np.roll(id2,+1)] + img[i,id2]

    return (scores,direction)

def backtrack(scores,best_id):
    """backtrack from each pixel of the bottom line
        returns path array

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    path_X : ndarray
        best path for each of the bottom pixel up to the top
    path_y : array
        corresponding y coordinate (same for each path, i.e. arange(0,np.image.shape[0]) )
    """

    m,n = scores.shape
    path_X = np.zeros((m,n))
    path_X[-1,:] = np.arange(n)
    path_y = np.arange(0,path_X.shape[0])
    for i in range(m-2,-1,-1):
        id0 = best_id[i+1,np.ix_(path_X[i+1,:])]==0
        id1 = best_id[i+1,np.ix_(path_X[i+1,:])]==1
        id2 = best_id[i+1,np.ix_(path_X[i+1,:])]==2

        path_X[i,id0.flatten()] = path_X[i+1,id0.flatten()]
        path_X[i,id1.flatten()] = path_X[i+1,id1.flatten()]-1
        path_X[i,id2.flatten()] = path_X[i+1,id2.flatten()]+1

    return (path_X,path_y)

def remove_seam(im,path_x):
    r = np.zeros_like(im)[:,:-1]
    n = im.shape[1]
    for row_in,row_out,x in zip(im,r,path_x):
        row_out[0:x] = row_in[0:x]
        row_out[x:] = row_in[x+1:]
    return r

def test_seams():
    im = np.zeros((500, 500)).astype(float)
    im[10:25, 10:25] = 1.

    scores,direction = compute_scores(im)
    path = backtrack(scores,direction)
    ax1 = plt.subplot(1,2,1)
    plt.imshow(im,interpolation='nearest')
    ax2 = plt.subplot(1,2,2)
    plt.imshow(scores,interpolation='nearest')

    #compute best path
    idx = np.argsort(scores[-1,:])
    ax1.plot(path[:,idx[-1]],np.arange(0,path.shape[0]),'k')
    ax2.plot(path[:,idx[-1]],np.arange(0,path.shape[0]),'w')
    print path
    plt.show()


def test_fascicule():

    from skimage.filter.rank import median
    from skimage.morphology import disk

    im = plt.imread('fascicule.jpg')[:,:,0]
    im = median(im,disk(3))

    (scores,direction) = compute_scores(im)
    path_X,path_y = backtrack(scores,direction)

    ax1 = plt.subplot(1,2,1)
    plt.imshow(im,interpolation='nearest')
    ax2 = plt.subplot(1,2,2)
    plt.imshow(scores,interpolation='nearest')

    #draw the best path
    idx = np.argsort(scores[-1,:])
    x = path_X[:,idx[-1]]
    y = path_y
    ax1.plot(x,y,'k')
    ax2.plot(x,y,'w')

    plt.show()

def test_resize():
    from skimage.filter.rank import entropy,gradient
    from skimage.morphology import disk
    from skimage.color import rgb2gray
    from skimage.util import img_as_ubyte

    im = img_as_ubyte(rgb2gray(plt.imread('800px-Broadway_tower_edit.jpg')))[-1::-1,:]

    gr = gradient(im,disk(3))
    orig = im.copy()

    for iter in range(10):

        (scores,direction) = compute_scores(-gr)
        path_X,path_y = backtrack(scores,direction)
        idx = np.argsort(scores[-1,:])
        x = path_X[:,idx[-1]]
        im = remove_seam(im,x)
        gr = remove_seam(gr,x)
        print iter

    ax1 = plt.subplot(1,2,1)
    plt.imshow(orig,interpolation='nearest')
    ax2 = plt.subplot(1,2,2)
    plt.imshow(im,interpolation='nearest')

    plt.show()




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test_fascicule()
    # test_seams()
    test_resize()