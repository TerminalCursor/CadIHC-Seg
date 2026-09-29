from ...interfaces.scanner.worker cimport Worker
from ..scanner2d cimport Scanner2D
from .tree_subworker cimport TreeSubworker

cdef class TreeWorker(Worker):
    cdef public list subworkers
    cdef public list new_data
    cdef public Scanner2D scanner
    cdef public long background
    cdef public object coords
    cdef public long _width

    cpdef get(self, scanner)
