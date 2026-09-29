from ..interfaces.scanner.scanner cimport Scanner
cimport numpy as cnp

cdef class Scanner2D(Scanner):
    cdef public long[:, :] data_view
    cdef public long x, y
    cdef public long shape_x, shape_y

    cpdef get(self, tuple xy=*)
