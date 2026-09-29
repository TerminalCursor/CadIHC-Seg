import numpy as np
cimport numpy as cnp

cdef class Scanner:
    cdef public tuple head
    cdef public object data
    cdef public list workers

    cpdef assign(self, worker)
    cpdef get(self, tuple xy=*)
    cpdef scan(self, bint progress=*)
