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
    path : ndarray
        best path for each of the bottom pixel up to the top

    """
    m,n = scores.shape
    path = np.zeros((m,n))
    path[-1,:] = np.arange(n)
    for i in range(m-2,-1,-1):
        id0 = best_id[i+1,np.ix_(path[i+1,:])]==0
        id1 = best_id[i+1,np.ix_(path[i+1,:])]==1
        id2 = best_id[i+1,np.ix_(path[i+1,:])]==2

        path[i,id0.flatten()] = path[i+1,id0.flatten()]
        path[i,id1.flatten()] = path[i+1,id1.flatten()]-1
        path[i,id2.flatten()] = path[i+1,id2.flatten()]+1

    return path

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

    im = im.T
    # im = im.T[-1::-1,:]

    (scores,direction) = compute_scores(im)
    path = backtrack(scores,direction)

    ax1 = plt.subplot(1,2,1)
    plt.imshow(im,interpolation='nearest')
    ax2 = plt.subplot(1,2,2)
    plt.imshow(scores,interpolation='nearest')

    #compute best path
    idx = np.argsort(scores[-1,:])
    ax1.plot(path[:,idx[-1]],np.arange(0,path.shape[0]),'k')
    ax2.plot(path[:,idx[-1]],np.arange(0,path.shape[0]),'w')


    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_fascicule()
    # test_seams()