from ..tree_subworker cimport TreeSubworker

cdef class InteriorOutlineSubworker(TreeSubworker):
    cpdef get(self, long self_value, set points):
        if self_value == self.background:
            return None
        
        cdef set unique_values = set(points)
        unique_values.discard(0)
        unique_values.discard(self_value)
        
        if len(unique_values) > 0:
            self.new_data_view[self.scanner.x, self.scanner.y] = 1
        return unique_values
