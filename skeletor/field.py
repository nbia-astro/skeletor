from numpy import ndarray, asarray, zeros, dtype
from .cython.dtypes import Float, Float2
from .cython.ppic2_wrapper import cppaguard2xl, cppnaguard2l
from .cython.ppic2_wrapper import cppcguard2xl, cppncguard2l


class Field(ndarray):

    def __new__(cls, grid, comm, **kwds):

        # Field size set accoring to number of guard layes
        obj = super().__new__(cls, shape=(grid.nypmx, grid.nxpmx), **kwds)

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
        return asarray(self[self.grid.lby:self.grid.uby,
                            self.grid.lbx:self.grid.ubx])

    def copy_guards(self):
        lbx = self.grid.lbx
        lby = self.grid.lby
        nubx = self.grid.nubx
        nuby = self.grid.nuby
        ubx = self.grid.ubx
        uby = self.grid.uby

        # lower active cells to upper guard layers
        self[uby:, lbx:ubx] = self.send_down(self[lby:lby+nuby, lbx:ubx])
        # upper active cells to lower guard layers
        self[:lby, lbx:ubx] = self.send_up(self[uby-lby:uby, lbx:ubx])
        # lower active cells to upper guard layers
        self[:, ubx:] = self[:, lbx:lbx+nubx]
        # upper active cells to lower guard layers
        self[:, :lbx] = self[:, ubx-lbx:ubx]


    def add_guards(self):
        lbx = self.grid.lbx
        lby = self.grid.lby
        nubx = self.grid.nubx
        nuby = self.grid.nuby
        ubx = self.grid.ubx
        uby = self.grid.uby

        # Add data from guard cells to corresponding active cells
        self[:, lbx:lbx+nubx] += self[:, ubx:]
        self[:, ubx-lbx:ubx] += self[:, :lbx]

        self[lby:lby+nuby, :] += self.send_up(self[uby:, :])
        self[uby-lby:uby, :] += self.send_down(self[:lby, :])

        # Erase guard cells
        self[:lby, :] = 0.0
        self[uby:, :] = 0.0
        self[:, ubx:] = 0.0
        self[:, :lbx] = 0.0


    def add_guards_ppic2(self):

        # This routine *only* works for scalar fields
        assert self.dtype == dtype(Float)

        cppaguard2xl(self, self.grid)
        cppnaguard2l(self, self.scr, self.grid)

    def copy_guards_ppic2(self):

        # This routine *only* works for scalar fields
        assert self.dtype == dtype(Float2)

        cppncguard2l(self, self.grid)
        cppcguard2xl(self, self.grid)
