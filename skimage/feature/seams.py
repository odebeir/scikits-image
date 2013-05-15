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
    imy : ndarray
        cost matrix

    """

    m,n = img.shape

    d = np.dstack((img[:,1:-1],img[:,0:-2],img[:,2:]))
    best_id = np.argmax(d,axis=2)
    best_id = np.hstack((np.zeros((m,1)),best_id,np.zeros((m,1))))
    scores = np.zeros_like(img,dtype=int)
    scores[0,:] = img[0,:]
    for i in range(1,m):
        id0 = best_id[i-1,:]==0
        id1 = best_id[i-1,:]==1
        id2 = best_id[i-1,:]==2

        scores[i,id0] = scores[i-1,id0] + img[i,id0]
        scores[i,id1] = scores[i-1,np.roll(id1,-1)] + img[i,id1]
        scores[i,id2] = scores[i-1,np.roll(id2,+1)] + img[i,id2]


    #back track one single seam
    path = np.zeros((m,n))
    path[-1,:] = np.arange(n)
    for i in range(m-2,-1,-1):
        id0 = best_id[i+1,np.ix_(path[i+1,:])]==0
        id1 = best_id[i+1,np.ix_(path[i+1,:])]==1
        id2 = best_id[i+1,np.ix_(path[i+1,:])]==2

        path[i,id0.flatten()] = path[i+1,id0.flatten()]
        path[i,id1.flatten()] = path[i+1,id1.flatten()]-1
        path[i,id2.flatten()] = path[i+1,id2.flatten()]+1

    return (scores,path)

def test_seams():
    im = np.zeros((500, 500)).astype(float)
    im[10:25, 10:25] = 1.

    scores,path = compute_scores(im)
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.imshow(im,interpolation='nearest')
    plt.subplot(1,2,2)
    plt.imshow(scores,interpolation='nearest')
    plt.colorbar()
    plt.show()
    print path

def test_fascicule():
    import matplotlib.pyplot as plt

    im = plt.imread('fascicule.jpg')[:,:,0].T

    scores,path = compute_scores(im)

    ax1 = plt.subplot(1,2,1)
    plt.imshow(im,interpolation='nearest')
    ax2 = plt.subplot(1,2,2)
    plt.imshow(path,interpolation='nearest')
    plt.colorbar()

    print path

    #compute best path
    x = np.argmax(scores[-1,:])
    print x
    p = [(x,path.shape[0]-1)]
    for y in range(path.shape[0]-1,1,-1):
        x = path[y,x]-1
        p.append((x,y))
    p = np.asarray(p)
    ax2.plot(p[:,0],p[:,1])
    plt.show()

if __name__ == '__main__':

    test_fascicule()