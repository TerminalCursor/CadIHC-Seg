from ..scanner2d cimport Scanner2D

cdef class TreeSubworker:
    cdef public object coords
    cdef public long background
    cdef public long _width
    cdef public Scanner2D scanner
    cdef public long[:, :] new_data_view
    cdef public object new_data

    cpdef get(self, long self_value, set points)
