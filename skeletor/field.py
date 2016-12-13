from numpy import ndarray, asarray, zeros, dtype
from .cython.dtypes import Float, Float2
from .cython.ppic2_wrapper import cppaguard2xl, cppnaguard2l
from .cython.ppic2_wrapper import cppcguard2xl, cppncguard2l


class Field(ndarray):

    def __new__(cls, grid, **kwds):

        # Field size set accoring to number of guard layes
        obj = super().__new__(cls, shape=(grid.nypmx, grid.nxpmx), **kwds)

        # Store grid
        obj.grid = grid

        # MPI communication
        obj.above = (grid.comm.rank + 1) % grid.comm.size
        obj.below = (grid.comm.rank - 1) % grid.comm.size

        # Scratch array needed for PPIC2's "add_guards" routine
        obj.scr = zeros((2, grid.nx + 2), Float)

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.grid = getattr(obj, "grid", None)
        self.above = getattr(obj, "above", None)
        self.below = getattr(obj, "below", None)
        self.scr = getattr(obj, "scr", None)

    def send_up(self, sendbuf):
        return self.grid.comm.sendrecv(
                sendbuf, dest=self.above, source=self.below)

    def send_down(self, sendbuf):
        return self.grid.comm.sendrecv(
                sendbuf, dest=self.below, source=self.above)

    def trim(self):
        return asarray(self[self.grid.lby:self.grid.uby,
                            self.grid.lbx:self.grid.ubx])

    @property
    def active(self):
        return asarray(self[self.grid.lby:self.grid.uby,
                            self.grid.lbx:self.grid.ubx])

    @active.setter
    def active(self, rhs):
        self[self.grid.lby:self.grid.uby,
             self.grid.lbx:self.grid.ubx] = rhs

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

        # PPIC2 setup
        if nubx == 2 and nuby == 1 and lbx == 0 and lby == 0:
            # Set the extra guard layer in x to zero
            # TODO: Get rid of this extra guard layer in PPIC2
            # That is, make PPIC2's FFT work with the extended grid
            self[:, -1] = 0.0


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

        # This routine only works for ppic grid layout
        assert(self.grid.nubx == 2 and self.grid.nuby == 1 and
               self.grid.lbx == 0 and self.grid.lby == 0)

        cppaguard2xl(self, self.grid)
        cppnaguard2l(self, self.scr, self.grid)

    def copy_guards_ppic2(self):

        # This routine *only* works for vector fields
        assert self.dtype == dtype(Float2)

        # This routine only works for ppic grid layout
        assert(self.grid.nubx == 2 and self.grid.nuby == 1 and
               self.grid.lbx == 0 and self.grid.lby == 0)

        cppncguard2l(self, self.grid)
        cppcguard2xl(self, self.grid)


class ShearField(Field):

    def __new__(cls, grid, comm, **kwds):

        from numpy.fft import rfftfreq
        from numpy import outer, pi

        obj = super().__new__(cls, grid, comm, **kwds)

        # Grid spacing
        # TODO: this should be a property of the Grid class
        dx = obj.grid.Lx/obj.grid.nx

        # Wave numbers for real-to-complex transforms
        obj.kx = 2*pi*rfftfreq(obj.grid.nx)/dx

        # Outer product of y and kx
        obj.y_kx = outer(obj.grid.y, obj.kx)

        return obj

    def _translate_boundary(self, trans, iy):

        "Translation using FFTs"
        from numpy.fft import rfft, irfft
        from numpy import exp

        # Number of active grid points in x
        nx = self.grid.nx

        # Translate in real space by phase shifting in spectral space
        self[iy, :nx] = irfft(exp(-1j*self.kx*trans)*rfft(self[iy, :nx]))

        # Set boundary condition
        self[iy, -2] = self[iy, 0]

    def add_guards(self, St):

        # Add data from guard cells to corresponding active cells in x
        self[:, 0] += self[:, -2]

        # Translate the y-ghostzones
        if self.comm.rank == self.comm.size - 1:
            trans = self.grid.Ly*St
            self._translate_boundary(trans, self.grid.nyp)

        # Add data from guard cells to corresponding active cells in y
        self[0, :] += self.send_up(self[-1, :])

        # Erase guard cells
        self[:, -2:] = 0.0
        self[-1, :] = 0.0

    def copy_guards(self, St):

        # Copy data to guard cells from corresponding active cells
        self[:-1, -2] = self[:-1, 0]
        self[-1, :-2] = self.send_down(self[0, :-2])
        self[-1, -2] = self.send_down(self[0, 0])

        # Translate the y-ghostzones
        if self.comm.rank == self.comm.size - 1:
            trans = -self.grid.Ly*St
            self._translate_boundary(trans, -1)

    def translate(self, St):
        """Translation using numpy's fft."""
        from numpy.fft import rfft, irfft
        from numpy import exp

        # Fourier transform along x
        fx_hat = rfft(self[:-1, :-2], axis=1)

        # Translate along x by an amount -S*t*y
        fx_hat *= exp(1j*St*self.y_kx)

        # Inverse Fourier transform along x
        self[:-1, :-2] = irfft(fx_hat, axis=1)

        # Set boundary condition?

    def copy_guards2(self):
        raise 'copy_guards2 not implemented for shearing fields'

    def add_guards2(self):
        raise 'add_guards2 not implemented for shearing fields'

    def add_guards_ppic2(self):
        raise 'add_guards_ppic2 not implemented for shearing fields'

    def copy_guards_ppic2(self):
        raise 'copy_guards_ppic2 not implemented for shearing fields'
