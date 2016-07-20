from numpy import ndarray, asarray
from mpi4py.MPI import COMM_WORLD as comm
from ppic2_wrapper import cppaguard2xl, cppnaguard2l
from ppic2_wrapper import cppcguard2xl, cppncguard2l


class Field(ndarray):

    def __new__(cls, grid, **kwds):

        # I don't know why PPIC2 uses two guard cells in the x-direction
        # instead of one. Whatever the reason though, let's not change this for
        # now.
        shape = grid.nyp + 1, grid.nx + 2
        obj = super().__new__(cls, shape, **kwds)

        obj.grid = grid

        return obj

    def __array_finalize__(self, obj):

        above = (comm.rank + 1) % comm.size
        below = (comm.rank - 1) % comm.size

        self.up = {'dest': above, 'source': below}
        self.down = {'dest': below, 'source': above}

        if obj is None:
            return

        self.grid = obj.grid

    def trim(self):
        return asarray(self[:-1, :-2])

    def sendrecv(self, sendbuf, **kwds):
        return comm.sendrecv(sendbuf, **kwds)

    def copy_guards(self):

        self[:-1, -2] = self[:-1, 0]
        self[-1, :-2] = self.sendrecv(self[0, :-2], **self.down)
        self[-1, -2] = self.sendrecv(self[0, 0], **self.down)

    def add_guards(self):

        self[:-1, 0] += self[:-1, -2]
        self[0, :-2] += self.sendrecv(self[-1, :-2], **self.up)
        self[0, 0] += self.sendrecv(self[-1, -2], **self.up)

    def copy_guards2(self):

        self[:, -2] = self[:, 0]
        self[-1, :] = self.sendrecv(self[0, :], **self.down)

    def add_guards2(self):

        self[:, 0] += self[:, -2]
        self[0, :] += self.sendrecv(self[-1, :], **self.up)

    def add_guards_ppic2(self):

        cppaguard2xl(self, self.grid.nyp, self.grid.nx)
        cppnaguard2l(self, self.grid.nyp, self.grid.nx)

    def copy_guards_ppic2(self):

        from fields import Field
        from dtypes import Float

        field = Field(self.grid, dtype=[("x", Float), ("y", Float)])
        field["x"] = self

        cppncguard2l(field, self.grid.nyp, self.grid.nx)
        cppcguard2xl(field, self.grid.nyp, self.grid.nx)

        self[...] = field["x"]


class GlobalField(Field):

    def __new__(cls, grid, **kwds):

        shape = grid.ny + 1, grid.nx + 2
        obj = super(Field, cls).__new__(cls, shape, **kwds)

        obj.grid = grid

        return obj

    def sendrecv(self, sendbuf, **kwds):
        return sendbuf
