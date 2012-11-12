import numpy as np
from numpy.testing import run_module_suite, assert_array_equal, assert_raises
import unittest

from skimage import data
from skimage.morphology import cmorph, disk
from skimage.filter import rank
from skimage.filter.rank import _crank8, _crank16, _crank16_percentiles


def test_random_sizes():
    # make sure the size is not a problem

    niter = 10
    elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    for m, n in np.random.random_integers(1, 100, size=(10, 2)):
        mask = np.ones((m, n), dtype=np.uint8)

        image8 = np.ones((m, n), dtype=np.uint8)
        out8 = np.empty_like(image8)
        _crank8.mean(image=image8, selem=elem, mask=mask, out=out8,
            shift_x=0, shift_y=0)
        assert_array_equal(image8.shape, out8.shape)
        _crank8.mean(image=image8, selem=elem, mask=mask, out=out8,
            shift_x=+1, shift_y=+1)
        assert_array_equal(image8.shape, out8.shape)

        image16 = np.ones((m, n), dtype=np.uint16)
        out16 = np.empty_like(image8, dtype=np.uint16)
        _crank16.mean(image=image16, selem=elem, mask=mask, out=out16,
            shift_x=0, shift_y=0)
        assert_array_equal(image16.shape, out16.shape)
        _crank16.mean(image=image16, selem=elem, mask=mask, out=out16,
            shift_x=+1, shift_y=+1)
        assert_array_equal(image16.shape, out16.shape)

        _crank16_percentiles.mean(image=image16, mask=mask, out=out16,
            selem=elem, shift_x=0, shift_y=0, p0=.1, p1=.9)
        assert_array_equal(image16.shape, out16.shape)
        _crank16_percentiles.mean(image=image16, mask=mask, out=out16,
            selem=elem, shift_x=+1, shift_y=+1, p0=.1, p1=.9)
        assert_array_equal(image16.shape, out16.shape)


def test_compare_with_cmorph_dilate():
    # compare the result of maximum filter with dilate

    image = (np.random.random((100, 100)) * 256).astype(np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    for r in range(1, 20, 1):
        elem = np.ones((r, r), dtype=np.uint8)
        _crank8.maximum(image=image, selem=elem, out=out, mask=mask)
        cm = cmorph.dilate(image=image, selem=elem)
        assert_array_equal(out, cm)


def test_compare_with_cmorph_erode():
    # compare the result of maximum filter with erode

    image = (np.random.random((100, 100)) * 256).astype(np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    for r in range(1, 20, 1):
        elem = np.ones((r, r), dtype=np.uint8)
        _crank8.minimum(image=image, selem=elem, out=out, mask=mask)
        cm = cmorph.erode(image=image, selem=elem)
        assert_array_equal(out, cm)


def test_bitdepth():
    # test the different bit depth for rank16

    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty((100, 100), dtype=np.uint16)
    mask = np.ones((100, 100), dtype=np.uint8)

    for i in range(5):
        image = np.ones((100, 100),dtype=np.uint16) * 255 * 2**i
        r = _crank16_percentiles.mean(image=image, selem=elem, mask=mask,
            out=out, shift_x=0, shift_y=0, p0=.1, p1=.9, bitdepth=8 + i)


def test_population():
    # check the number of valid pixels in the neighborhood

    image = np.zeros((5, 5), dtype=np.uint8)
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    _crank8.pop(image=image, selem=elem, out=out, mask=mask)
    r = np.array([[4, 6, 6, 6, 4],
                  [6, 9, 9, 9, 6],
                  [6, 9, 9, 9, 6],
                  [6, 9, 9, 9, 6],
                  [4, 6, 6, 6, 4]])
    assert_array_equal(r, out)


def test_structuring_element8():
    # check the output for a custom structuring element

    r = np.array([[  0,   0,   0,   0,   0,   0],
                  [  0,   0,   0,   0,   0,   0],
                  [  0,   0, 255,   0,   0,   0],
                  [  0,   0, 255, 255, 255,   0],
                  [  0,   0,   0, 255, 255,   0],
                  [  0,   0,   0,   0,   0,   0]])

    # 8bit
    image = np.zeros((6, 6), dtype=np.uint8)
    image[2, 2] = 255
    elem = np.asarray([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    _crank8.maximum(image=image, selem=elem, out=out, mask=mask,
        shift_x=1, shift_y=1)
    assert_array_equal(r, out)

    # 16bit
    image = np.zeros((6, 6), dtype=np.uint16)
    image[2, 2] = 255
    out = np.empty_like(image)

    _crank16.maximum(image=image, selem=elem, out=out, mask=mask,
        shift_x=1, shift_y=1)
    assert_array_equal(r, out)


def test_fail_on_bitdepth():
    # should fail because data bitdepth is too high for the function

    image = np.ones((100, 100), dtype=np.uint16) * 255
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)
    assert_raises(AssertionError, _crank16_percentiles.mean, image=image,
        selem=elem, out=out, mask=mask, shift_x=0, shift_y=0, bitdepth=4)


def test_inplace_output():
    # rank filters are not supposed to filter inplace

    selem = disk(20)
    image = (np.random.random((500,500))*256).astype(np.uint8)
    out = image
    assert_raises(NotImplementedError, rank.mean, image, selem, out=out)


def test_compare_autolevels():
    # compare autolevel and percentile autolevel with p0=0.0 and p1=1.0
    # should returns the same arrays

    image = data.camera()

    selem = disk(20)
    loc_autolevel = rank.autolevel(image, selem=selem)
    loc_perc_autolevel = rank.percentile_autolevel(image, selem=selem,
        p0=.0, p1=1.)

    assert_array_equal(loc_autolevel, loc_perc_autolevel)


def test_compare_autolevels_16bit():
    # compare autolevel(16bit) and percentile autolevel(16bit) with p0=0.0 and
    # p1=1.0 should returns the same arrays

    image = data.camera().astype(np.uint16) * 4

    selem = disk(20)
    loc_autolevel = rank.autolevel(image, selem=selem)
    loc_perc_autolevel = rank.percentile_autolevel(image, selem=selem,
        p0=.0, p1=1.)

    assert_array_equal(loc_autolevel, loc_perc_autolevel)


def test_compare_8bit_vs_16bit():
    # filters applied on 8bit image ore 16bit image (having only real 8bit of
    # dynamic) should be identical

    image8 = data.camera()
    image16 = image8.astype(np.uint16)
    assert_array_equal(image8, image16)

    methods = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'maximum',
               'mean', 'meansubstraction', 'median', 'minimum', 'modal',
               'morph_contr_enh', 'pop', 'threshold',  'tophat']

    for method in methods:
        func = getattr(rank, method)
        f8 = func(image8, disk(3))
        f16 = func(image16, disk(3))
        assert_array_equal(f8, f16)


def test_trivial_selem8():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint8)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[0, 0, 0], [0, 1, 0],[0, 0, 0]], dtype=np.uint8)
    _crank8.mean(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank8.minimum(image=image, selem=elem, out=out, mask=mask,
                    shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank8.maximum(image=image, selem=elem, out=out, mask=mask,
                    shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_trivial_selem16():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[0, 0, 0], [0, 1, 0],[0, 0, 0]], dtype=np.uint8)
    _crank16.mean(image=image, selem=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank16.minimum(image=image, selem=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank16.maximum(image=image, selem=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_smallest_selem8():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint8)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[1]], dtype=np.uint8)
    _crank8.mean(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank8.minimum(image=image, selem=elem, out=out, mask=mask,
                    shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank8.maximum(image=image, selem=elem, out=out, mask=mask,
                    shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_smallest_selem16():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[1]], dtype=np.uint8)
    _crank16.mean(image=image, selem=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank16.minimum(image=image, selem=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    _crank16.maximum(image=image, selem=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
    assert_array_equal(image, out)


if __name__ == "__main__":
    run_module_suite()