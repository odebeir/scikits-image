import numpy as np
from numpy.testing import assert_array_equal

from skimage import data
from skimage import img_as_float

from skimage.feature import compute_seams


def test_seams():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.

    results = seams(im)

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
