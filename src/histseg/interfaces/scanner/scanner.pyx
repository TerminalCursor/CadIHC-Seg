cimport numpy as cnp

cdef class Scanner:
    """# General Sequence Scanner Class"""

    def __init__(self, data):
        self.data = data
        self.workers = []
        self.head = (0,) * len(data.shape)

    def __iter__(self):
        return self

    cpdef assign(self, worker):
        self.workers.append(worker)
        return self

    def __next__(self):
        raise NotImplementedError("Implement Me!")

    cpdef get(self, tuple xy=None):
        raise NotImplementedError("Implement Me!")

    cpdef scan(self, bint progress=True):
        from tqdm.auto import tqdm
        from numpy import prod
        cdef long _max = int(prod(self.data.shape))
        if progress:
            pbar = tqdm(iter(self), total=_max)
        else:
            pbar = iter(self)
        for _ in pbar:
            pass
        return self
