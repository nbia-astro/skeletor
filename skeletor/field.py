from numpy import ndarray, asarray, zeros
from .cython.dtypes import Float
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
        above = (comm.rank + 1) % comm.size
        below = (comm.rank - 1) % comm.size
        up = {'dest': above, 'source': below}
        down = {'dest': below, 'source': above}
        obj.send_up = lambda sendbuf: comm.sendrecv(sendbuf, **up)
        obj.send_down = lambda sendbuf: comm.sendrecv(sendbuf, **down)

        # Scratch array needed for PPIC2's "add_guards" routine
        obj.scr = zeros((2, grid.nx + 2), Float)

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.grid = getattr(obj, "grid", None)
        self.comm = getattr(obj, "comm", None)
        self.scr = getattr(obj, "scr", None)
        self.send_up = getattr(obj, "send_up", None)
        self.send_down = getattr(obj, "send_down", None)

    def trim(self):
        return asarray(self[:-1, :-2])

    def copy_guards(self):

        self[:-1, -2] = self[:-1, 0]
        self[-1, :-2] = self.send_down(self[0, :-2])
        self[-1, -2] = self.send_down(self[0, 0])

    def add_guards(self):

        self[:-1, 0] += self[:-1, -2]
        self[0, :-2] += self.send_up(self[-1, :-2])
        self[0, 0] += self.send_up(self[-1, -2])

    def copy_guards2(self):

        self[:, -2] = self[:, 0]
        self[-1, :] = self.send_down(self[0, :])

    def add_guards2(self):

        self[:, 0] += self[:, -2]
        self[0, :] += self.send_up(self[-1, :])

    def add_guards_ppic2(self):

        cppaguard2xl(self, self.grid)
        cppnaguard2l(self, self.scr, self.grid)

    def copy_guards_ppic2(self):

        from .field import Field
        from .cython.dtypes import Float2

        field = Field(self.grid, self.comm, dtype=Float2)
        field["x"] = self

        cppncguard2l(field, self.grid)
        cppcguard2xl(field, self.grid)

        self[...] = field["x"]
