cdef class Worker:
    cpdef get(self, scanner):
        raise NotImplementedError("Implement me!")
