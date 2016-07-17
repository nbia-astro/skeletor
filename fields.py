from numpy import ndarray
from mpi4py.MPI import COMM_WORLD as comm


class Field(ndarray):

    def __new__(cls, grid, **kwds):

        shape = grid.nyp + 2*grid.nghost, grid.nx + 2*grid.nghost
        obj = super().__new__(cls, shape, **kwds)

        return obj

    def __array_finalize__(self, obj):

        above = (comm.rank + 1) % comm.size
        below = (comm.rank - 1) % comm.size

        self.up = {'dest': above, 'source': below}
        self.down = {'dest': below, 'source': above}

    def sendrecv(sendbuf, **kwds):
        return comm.sendrecv(sendbuf, **kwds)

    def apply_bc(self):

        self[1:-1, -1] = self[1:-1, 1]
        self[1:-1, 0] = self[1:-1, -2]

        self[-1, 1:-1] = self.sendrecv(self[1, 1:-1], **self.down)
        self[0, 1:-1] = self.sendrecv(self[-2, 1:-1], **self.up)

        self[-1, -1] = self.sendrecv(self[1, 1], **self.down)
        self[-1, 0] = self.sendrecv(self[1, -2], **self.down)
        self[0, -1] = self.sendrecv(self[-2, 1], **self.up)
        self[0, 0] = self.sendrecv(self[-2, -2], **self.up)

    def apply_bc2(self):

        self[-1, 1:-1] = self.sendrecv(self[1, 1:-1], **self.down)
        self[0, 1:-1] = self.sendrecv(self[-2, 1:-1], **self.up)

        self[:, -1] = self[:, 1]
        self[:, 0] = self[:, -2]


class SourceField(Field):

    def add_guards(self):

        self[1:-1, 1] += self[1:-1, -1]
        self[1:-1, -2] += self[1:-1, 0]

        self[1, 1:-1] += comm.sendrecv(self[-1, 1:-1], **self.up)
        self[-2, 1:-1] += comm.sendrecv(self[0, 1:-1], **self.down)

        self[1, 1] += comm.sendrecv(self[-1, -1], **self.up)
        self[1, -2] += comm.sendrecv(self[-1, 0], **self.up)
        self[-2, 1] += comm.sendrecv(self[0, -1], **self.down)
        self[-2, -2] += comm.sendrecv(self[0, 0], **self.down)

    def add_guards2(self):

        self[:, 1] += self[:, -1]
        self[:, -2] += self[:, 0]

        self[1, 1:-1] += comm.sendrecv(self[-1, 1:-1], **self.up)
        self[-2, 1:-1] += comm.sendrecv(self[0, 1:-1], **self.down)


class GlobalField(Field):

    def __new__(cls, grid, **kwds):

        shape = grid.ny + 2*grid.nghost, grid.nx + 2*grid.nghost
        obj = super(Field, cls).__new__(cls, shape, **kwds)

        return obj

    def sendrecv(sendbuf, **kwds):
        return sendbuf


class GlobalSourceField(SourceField):

    def __new__(cls, grid, **kwds):

        shape = grid.ny + 2*grid.nghost, grid.nx + 2*grid.nghost
        obj = super(Field, cls).__new__(cls, shape, **kwds)

        return obj

    def sendrecv(sendbuf, **kwds):
        return sendbuf
