from numpy import ndarray, asarray, zeros, dtype
from .cython.dtypes import Float, Float2
from .cython.ppic2_wrapper import cppaguard2xl, cppnaguard2l
from .cython.ppic2_wrapper import cppcguard2xl, cppncguard2l


class Field(ndarray):

    def __new__(cls, grid, comm, **kwds):

        # I don't know why PPIC2 uses two guard cells in the x-direction
        # instead of one. Whatever the reason though, let's not change this for
        # now.
        nxv = grid.nx + 2
        obj = super().__new__(cls, shape=(grid.nypmx, nxv), **kwds)

        # Store grid
        obj.grid = grid

        # Store MPI communicator
        obj.comm = comm

        # MPI communication
        obj.above = (comm.rank + 1) % comm.size
        obj.below = (comm.rank - 1) % comm.size

        # Scratch array needed for PPIC2's "add_guards" routine
        obj.scr = zeros((2, grid.nx + 2), Float)

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.grid = getattr(obj, "grid", None)
        self.comm = getattr(obj, "comm", None)
        self.above = getattr(obj, "above", None)
        self.below = getattr(obj, "below", None)
        self.scr = getattr(obj, "scr", None)

    def send_up(self, sendbuf):
        return self.comm.sendrecv(
                sendbuf, dest=self.above, source=self.below)

    def send_down(self, sendbuf):
        return self.comm.sendrecv(
                sendbuf, dest=self.below, source=self.above)

    def trim(self):
        return asarray(self[:-1, :-2])

    def copy_guards(self):

        # Copy data to guard cells from corresponding active cells
        self[:-1, -2] = self[:-1, 0]
        self[-1, :-2] = self.send_down(self[0, :-2])
        self[-1, -2] = self.send_down(self[0, 0])

    def add_guards(self):

        # Add data from guard cells to corresponding active cells
        self[:-1, 0] += self[:-1, -2]
        self[0, :-2] += self.send_up(self[-1, :-2])
        self[0, 0] += self.send_up(self[-1, -2])

        # Erase guard cells (TH: Not sure why PPIC2 does this, but it's OK)
        self[:-1, -2] = 0.0
        self[-1, :-2] = 0.0
        self[-1, -2] = 0.0

    def copy_guards2(self):

        # Copy data to guard cells from corresponding active cells
        self[:, -2] = self[:, 0]
        self[-1, :] = self.send_down(self[0, :])

    def add_guards2(self):

        # Add data from guard cells to corresponding active cells
        self[:, 0] += self[:, -2]
        self[0, :] += self.send_up(self[-1, :])

        # Erase guard cells (TH: Not sure why PPIC2 does this, but it's OK)
        self[:, -2] = 0.0
        self[-1, :] = 0.0

    def add_guards_ppic2(self):

        # This routine *only* works for scalar fields
        assert self.dtype == dtype(Float)

        cppaguard2xl(self, self.grid)
        cppnaguard2l(self, self.scr, self.grid)

    def copy_guards_ppic2(self):

        # This routine *only* works for vector fields
        assert self.dtype == dtype(Float2)

        cppncguard2l(self, self.grid)
        cppcguard2xl(self, self.grid)
