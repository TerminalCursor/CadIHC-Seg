from ..scanner2d cimport Scanner2D
import numpy as np

cdef class TreeSubworker:
    def __init__(self, scanner, coords, background=0, width=2):
        self.scanner = scanner
        self.coords = coords
        self.background = int(background)
        self._width = width
        self.new_data = np.zeros_like(scanner.data, dtype=np.int64)
        self.new_data_view = self.new_data

    cpdef get(self, long self_value, set points):
        raise NotImplementedError("Implement me!")
