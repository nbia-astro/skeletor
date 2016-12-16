from numpy import ndarray, asarray, zeros, dtype
from .cython.dtypes import Float, Float2
from .cython.ppic2_wrapper import cppaguard2xl, cppnaguard2l
from .cython.ppic2_wrapper import cppcguard2xl, cppncguard2l


class Field(ndarray):

    def __new__(cls, grid, time=0.0, **kwds):

        # Field size set accoring to number of guard layes
        obj = super().__new__(cls, shape=(grid.nypmx, grid.nxpmx), **kwds)

        # Store grid
        obj.grid = grid

        # MPI communication
        obj.above = (grid.comm.rank + 1) % grid.comm.size
        obj.below = (grid.comm.rank - 1) % grid.comm.size

        # Scratch array needed for PPIC2's "add_guards" routine
        obj.scr = zeros((2, grid.nx + 2), Float)

        # Boolean indicating whether boundaries are set
        obj.boundaries_set = False

        # Time of the field
        obj.time = time

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.grid = getattr(obj, "grid", None)
        self.above = getattr(obj, "above", None)
        self.below = getattr(obj, "below", None)
        self.scr = getattr(obj, "scr", None)
        self.boundaries_set = getattr(obj, "boundaries_set", None)
        self.time = getattr(obj, "time", None)

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

        msg = 'Boundaries are already set!'
        assert(not self.boundaries_set), msg
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

        self.boundaries_set = True

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
    # TODO: ShearField really needs to have a ShearManifold passed, it
    # will fail if a standard Grid or Manifold is passed. Change variable name
    # to manifold or shearmanifold?

    def __new__(cls, grid, time=0.0, **kwds):

        from numpy.fft import rfftfreq
        from numpy import outer, pi

        obj = super().__new__(cls, grid, time=time, **kwds)

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

        lbx = self.grid.lbx
        nubx = self.grid.nubx
        ubx = self.grid.ubx

        # Translate in real space by phase shifting in spectral space
        phase = -1j*self.kx*trans
        if self.dtype == dtype(Float):
            self[iy, self.grid.lbx:self.grid.ubx] = \
                irfft(exp(phase)*rfft(self[iy, self.grid.lbx:self.grid.ubx]))
        elif self.dtype == dtype(Float2):
            for dim in ('x', 'y'):
                self[iy, self.grid.lbx:self.grid.ubx][dim] = \
                    irfft(exp(phase) *
                          rfft(self[iy, self.grid.lbx:self.grid.ubx][dim]))

        # lower active cells to upper guard layers
        self[iy, ubx:] = self[iy, lbx:lbx+nubx]
        # upper active cells to lower guard layers
        self[iy, :lbx] = self[iy, ubx-lbx:ubx]

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

        # Translate the y-ghostzones
        if self.grid.comm.rank == self.grid.comm.size - 1:
            trans = self.grid.Ly*self.grid.S*self.time
            for iy in range(uby, uby + nuby):
                self._translate_boundary(trans, iy)
        if self.grid.comm.rank == 0:
            trans = -self.grid.Ly*self.grid.S*self.time
            for iy in range(0, lby):
                self._translate_boundary(trans, iy)

        # Add data from guard cells to corresponding active cells in y
        self[lby:lby+nuby, :] += self.send_up(self[uby:, :])
        self[uby-lby:uby, :] += self.send_down(self[:lby, :])

        # Erase guard cells
        self[:lby, :] = 0.0
        self[uby:, :] = 0.0
        self[:, ubx:] = 0.0
        self[:, :lbx] = 0.0

    def copy_guards(self):

        msg = 'Boundaries are already set!'
        assert not self.boundaries_set, msg

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

        # Translate the y-ghostzones
        if self.grid.comm.rank == self.grid.comm.size - 1:
            trans = -self.grid.Ly*self.grid.S*self.time
            for iy in range(uby, uby + nuby):
                self._translate_boundary(trans, iy)
        if self.grid.comm.rank == 0:
            trans = +self.grid.Ly*self.grid.S*self.time
            for iy in range(0, lby):
                self._translate_boundary(trans, iy)

        self.boundaries_set = True

    def translate(self, time):
        """Translation using numpy's fft."""
        from numpy.fft import rfft, irfft
        from numpy import exp

        # Fourier transform along x
        fx_hat = rfft(self.trim(), axis=1)

        # Translate along x by an amount -S*t*y
        fx_hat *= exp(1j*self.grid.S*time*self.y_kx)

        # Inverse Fourier transform along x
        self.active = irfft(fx_hat, axis=1)

        # Set boundary condition?
        self.boundaries_set = False

    def add_guards_ppic2(self):
        raise 'add_guards_ppic2 not implemented for shearing fields'

    def copy_guards_ppic2(self):
        raise 'copy_guards_ppic2 not implemented for shearing fields'
