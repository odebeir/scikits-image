cimport numpy as np

cdef np.uint16_t kernel_mean( Py_ssize_t * histo, float pop, np.uint16_t g, Py_ssize_t bitdepth,
        Py_ssize_t maxbin, Py_ssize_t midbin, float p0, float p1, Py_ssize_t s0, Py_ssize_t s1)

cdef np.uint16_t kernel_pop( Py_ssize_t * histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin,
        Py_ssize_t midbin, float p0, float p1, Py_ssize_t s0, Py_ssize_t s1)