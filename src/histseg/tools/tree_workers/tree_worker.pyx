from ...interfaces.scanner.worker cimport Worker
from ..scanner2d cimport Scanner2D
from .tree_subworker cimport TreeSubworker
import numpy as np

cdef class TreeWorker(Worker):
    def __init__(self, scanner, background=0, width=2):
        self.scanner = scanner
        self.background = int(background)
        self.coords = np.array(np.where(scanner.data - self.background + 1)).T
        self._width = width
        self.subworkers = []
        self.new_data = []

    def new_worker(self, subworker_class):
        cdef TreeSubworker subworker = subworker_class(
                scanner=self.scanner,
                coords=self.coords,
                background=self.background,
                width=self._width,
            )
        self.subworkers.append(subworker)
        self.new_data.append(subworker.new_data)

    cpdef get(self, scanner_obj):
        cdef list return_values = []
        cdef set points = set()
        cdef list vs = [v for v in range(-self._width, self._width+1)]
        cdef long dx, dy
        cdef object point
        cdef Scanner2D scanner = <Scanner2D>scanner_obj
        cdef long hx = scanner.x
        cdef long hy = scanner.y
        
        for dx in vs:
            for dy in vs:
                if dx == 0 and dy == 0:
                    continue
                point = scanner.get((hx + dx, hy + dy))
                if point is None:
                    continue
                points.add(int(point))
                
        cdef long index
        cdef TreeSubworker subworker
        cdef object self_point = scanner.get(None)
        cdef long self_value = 0
        if self_point is not None:
            self_value = int(self_point)
        
        for index, subworker_obj in enumerate(self.subworkers):
            subworker = <TreeSubworker>subworker_obj
            return_values.append(subworker.get(self_value, points))
            self.new_data[index] = subworker.new_data
            
        return return_values
