from ..tree_subworker cimport TreeSubworker

cdef class FullOutlineSubworker(TreeSubworker):
    cpdef get(self, long self_value, set points):
        if self_value == self.background:
            return None
            
        cdef set without_self = set(points)
        without_self.discard(self_value)
        
        cdef long sx = self.scanner.x
        cdef long sy = self.scanner.y
        
        if len(without_self) > 0:
            self.new_data_view[sx, sy] = 1
        elif (sx - self._width + 1 <= 0) or (sy - self._width + 1 <= 0):
            self.new_data_view[sx, sy] = 1
        elif (sx + self._width >= self.scanner.shape_x) or (sy + self._width >= self.scanner.shape_y):
            self.new_data_view[sx, sy] = 1
            
        return points
