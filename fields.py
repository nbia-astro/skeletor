from numpy import ndarray, asarray, zeros
from dtypes import Float
from mpi4py import MPI
from ppic2_wrapper import cppaguard2xl, cppnaguard2l
from ppic2_wrapper import cppcguard2xl, cppncguard2l


class Field(ndarray):

    def __new__(cls, grid, comm=MPI.COMM_WORLD, **kwds):

        # I don't know why PPIC2 uses two guard cells in the x-direction
        # instead of one. Whatever the reason though, let's not change this for
        # now.
        shape = grid.nyp + 1, grid.nx + 2
        obj = super().__new__(cls, shape, **kwds)

        # Store grid
        obj.grid = grid

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

        self.grid = obj.grid
        self.scr = obj.scr
        self.send_up = obj.send_up
        self.send_down = obj.send_down

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

        cppaguard2xl(self, self.grid.nyp, self.grid.nx)
        cppnaguard2l(self, self.scr, self.grid.nyp, self.grid.nx)

    def copy_guards_ppic2(self):

        from fields import Field
        from dtypes import Float

        field = Field(self.grid, dtype=[("x", Float), ("y", Float)])
        field["x"] = self

        cppncguard2l(field, self.grid.nyp, self.grid.nx)
        cppcguard2xl(field, self.grid.nyp, self.grid.nx)

        self[...] = field["x"]


class GlobalField(Field):

    def __new__(cls, grid, comm=MPI.COMM_WORLD, **kwds):

        shape = grid.ny + 1, grid.nx + 2
        obj = super(Field, cls).__new__(cls, shape, **kwds)

        obj.grid = grid
        obj.scr = zeros((2, grid.nx + 2), Float)
        obj.send_up = lambda sendbuf: sendbuf
        obj.send_down = lambda sendbuf: sendbuf

        return obj
