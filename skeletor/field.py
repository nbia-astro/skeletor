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

        # This routine *only* works for scalar fields
        assert self.dtype == dtype(Float2)

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

    def _translate_fft(self, trans, iy):

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
            self._translate_fft(trans, self.grid.nyp)

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
            self._translate_fft(trans, -1)

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
