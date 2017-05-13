import numpy as np


class Field(np.ndarray):

    def __new__(cls, grid, time=0.0, **kwds):

        # Field size set accoring to number of guard layes
        obj = super().__new__(cls, shape=(grid.myp, grid.mx), **kwds)

        # Store grid
        obj.grid = grid

        # MPI communication
        obj.above = (grid.comm.rank + 1) % grid.comm.size
        obj.below = (grid.comm.rank - 1) % grid.comm.size

        # Boolean indicating whether boundaries are set
        obj.boundaries_set = False

        # Time of the field
        obj.time = time

        # Is there shear?
        obj.shear = hasattr(grid, 'S')

        if obj.shear:

            # Wave numbers for real-to-complex transforms
            obj.kx = 2*np.pi*np.fft.rfftfreq(grid.nx)/grid.dx

            # Outer product of y and kx
            obj.y_kx = np.outer(grid.y, obj.kx)

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.grid = getattr(obj, "grid", None)
        self.above = getattr(obj, "above", None)
        self.below = getattr(obj, "below", None)
        self.boundaries_set = getattr(obj, "boundaries_set", None)
        self.time = getattr(obj, "time", None)
        self.shear = getattr(obj, "shear", None)
        if self.shear:
            self.kx = getattr(obj, "kx", None)
            self.y_kx = getattr(obj, "y_kx", None)

    def send_up(self, sendbuf):
        return self.grid.comm.sendrecv(
                sendbuf, dest=self.above, source=self.below)

    def send_dn(self, sendbuf):
        return self.grid.comm.sendrecv(
                sendbuf, dest=self.below, source=self.above)

    @property
    def active(self):
        return np.asarray(self[self.grid.lby:self.grid.uby,
                               self.grid.lbx:self.grid.ubx])

    @active.setter
    def active(self, rhs):
        self[self.grid.lby:self.grid.uby,
             self.grid.lbx:self.grid.ubx] = rhs

    def trim(self):
        return self.active.squeeze()

    def copy_guards_x(self):
        """
        Periodic boundary condition in x.
        Note: this acts on _all_ grid points along y (active plus guards).
        """
        g = self.grid

        for ix in range(g.lbx - 1, -1, -1):
            self[:, ix] = self[:, ix + g.nx]
        for ix in range(g.ubx, g.ubx + g.lbx):
            self[:, ix] = self[:, ix - g.nx]

    def copy_guards_y(self):
        """
        Periodic boundary condition in y.
        Note: this only acts on active grid points along x.
        """
        g = self.grid

        for iy in range(g.lby - 1, -1, -1):
            self[iy, g.lbx:g.ubx] = self[iy + g.nyp, g.lbx:g.ubx]
        for iy in range(g.uby, g.uby + g.lby):
            self[iy, g.lbx:g.ubx] = self[iy - g.nyp, g.lbx:g.ubx]

        self[g.uby:, g.lbx:g.ubx] = self.send_dn(self[g.uby:, g.lbx:g.ubx])
        self[:g.lby, g.lbx:g.ubx] = self.send_up(self[:g.lby, g.lbx:g.ubx])

    def copy_guards(self):
        "Copy data to guard cells from corresponding active cells."

        # Make sure boundaries aren't already set
        # TODO: turn this into a warning. Other than communication overhead,
        # calling this multiple times doesn't really cause any harm.
        assert not self.boundaries_set, 'Boundaries are already set!'

        # y-boundaries
        self.copy_guards_y()
        # x-boundaries
        self.copy_guards_x()

        if self.shear:
            # Short hand
            g = self.grid
            # Translate the y-ghostzones
            if g.comm.rank == g.comm.size - 1:
                trans = -g.Ly*g.S*self.time
                for iy in range(g.uby, g.uby + g.lby):
                    self._translate_boundary(trans, iy)
            if g.comm.rank == 0:
                trans = +g.Ly*g.S*self.time
                for iy in range(0, g.lby):
                    self._translate_boundary(trans, iy)

        self.boundaries_set = True

    def _translate_boundary(self, trans, iy):
        "Translation using FFTs"

        # Short hand
        g = self.grid

        # Translate in real space by phase shifting in spectral space
        fac = np.exp(-1j*self.kx*trans)
        if self.dtype.names is None:
            self[iy, g.lbx:g.ubx] = np.fft.irfft(
                    fac*np.fft.rfft(self[iy, g.lbx:g.ubx]))
        else:
            for dim in self.dtype.names:
                self[iy, g.lbx:g.ubx][dim] = np.fft.irfft(
                        fac*np.fft.rfft(self[iy, g.lbx:g.ubx][dim]))

        # Update x-boundaries
        self[iy, g.ubx:] = self[iy, g.lbx:g.lbx+g.lbx]
        self[iy, :g.lbx] = self[iy, g.ubx-g.lbx:g.ubx]

    def translate(self, time):
        """Translation using numpy's fft."""

        if not self.shear:
            return

        # Fourier transform along x
        fx_hat = np.fft.rfft(self.trim(), axis=1)

        # Translate along x by an amount -S*t*y
        fx_hat *= np.exp(1j*self.grid.S*time*self.y_kx)

        # Inverse Fourier transform along x
        self.active = np.fft.irfft(fx_hat, axis=1)

        # Set boundary condition?
        self.boundaries_set = False

    def translate_vector(self, time):
        """Translation using numpy's fft."""
        # TODO: Combine this with `translate()`

        if not self.shear:
            return

        for dim in ('x', 'y', 'z'):
            # Fourier transform along x
            fx_hat = np.fft.rfft(self.trim()[dim], axis=1)

            # Translate along x by an amount -S*t*y
            fx_hat *= np.exp(1j*self.grid.S*time*self.y_kx)

            # Inverse Fourier transform along x
            self.active[dim] = np.fft.irfft(fx_hat, axis=1)

        # Set boundary condition?
        self.boundaries_set = False
