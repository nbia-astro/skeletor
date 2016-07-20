from numpy import ndarray, asarray
from mpi4py.MPI import COMM_WORLD as comm
from ppic2_wrapper import cppaguard2xl, cppnaguard2l
from dtypes import Float
import numpy


class Field(ndarray):

    def __new__(cls, grid, **kwds):

        shape = grid.nyp + 1, grid.nx + 1
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
        return asarray(self[:-1, :-1])

    def sendrecv(self, sendbuf, **kwds):
        return comm.sendrecv(sendbuf, **kwds)

    def copy_guards(self):

        self[:-1, -1] = self[:-1, 0]
        self[-1, :-1] = self.sendrecv(self[0, :-1], **self.up)
        self[-1, -1] = self.sendrecv(self[0, 0], **self.up)

    def add_guards(self):

        self[:-1, 0] += self[:-1, -1]
        self[0, :-1] += self.sendrecv(self[-1, :-1], **self.up)
        self[0, 0] += self.sendrecv(self[-1, -1], **self.up)

    def copy_guards2(self):

        self[:, -1] = self[:, 0]
        self[-1, :] = self.sendrecv(self[0, :], **self.down)

    def add_guards2(self):

        self[:, 0] += self[:, -1]
        self[0, :] += self.sendrecv(self[-1, :], **self.up)

    def add_guards_ppic2(self):

        grid = self.grid
        field = numpy.zeros((grid.nypmx, grid.nx + 2), Float)
        field[:grid.nyp+1, :grid.nx+1] = self
        cppaguard2xl(field, grid.nyp, grid.nx)
        cppnaguard2l(field, grid.nyp, grid.nx)
        self[...] = field[:grid.nyp+1, :grid.nx+1]


class GlobalField(Field):

    def __new__(cls, grid, **kwds):

        shape = grid.ny + 1, grid.nx + 1
        obj = super(Field, cls).__new__(cls, shape, **kwds)

        obj.grid = grid

        return obj

    def sendrecv(self, sendbuf, **kwds):
        return sendbuf
