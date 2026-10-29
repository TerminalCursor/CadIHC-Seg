from ..interfaces.scanner.scanner cimport Scanner
import numpy as np

cdef class Scanner2D(Scanner):
    def __init__(self, source_data):
        super().__init__(source_data)
        self.data = np.asarray(source_data, dtype=np.int64)
        self.data_view = self.data
        assert len(self.data.shape) == 2
        self.shape_x = self.data.shape[0]
        self.shape_y = self.data.shape[1]
        self.x = 0
        self.y = 0
        self.head = (self.x, self.y)

    def __next__(self):
        cdef long x = self.x
        cdef long y = self.y
        
        for worker in self.workers:
            worker.get(self)
        
        x += 1
        if x >= self.shape_x:
            x = 0
            y += 1
        if y >= self.shape_y:
            y = 0
            self.x = x
            self.y = y
            self.head = (x, y)
            raise StopIteration
            
        self.x = x
        self.y = y
        self.head = (x, y)
        return self

    cpdef get(self, tuple xy = None):
        cdef long x, y
        if xy is None:
            return self.data_view[self.x, self.y]
        x = xy[0]
        y = xy[1]
        if x < 0 or y < 0 or x >= self.shape_x or y >= self.shape_y:
            return None
        return self.data_view[x, y]
