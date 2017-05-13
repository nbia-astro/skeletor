from .cython.types import Float
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

        # Scratch array needed for PPIC2's "add_guards" routine
        obj.scr = np.zeros((2, grid.nx + 2), Float)

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

    def __iadd__(self, other):
        """Overloads the += operator. This is convenient for doing arithmetic
        with structured Numpy arrays (in particular the sources array)."""

        # If this is not a structured array, just call the corresponding method
        # from ndarray. Note that we can't replace the call to __iadd__ with
        #   self += other
        # as that would lead to infinite recursion
        if self.dtype.names is None:
            return super().__iadd__(other)

        # If this _is_ a structured array, loop over all names
        for dim in self.dtype.names:
            self[dim] += other[dim]
        return self

    def send_up(self, sendbuf):
        return self.grid.comm.sendrecv(
                sendbuf, dest=self.above, source=self.below)

    def send_down(self, sendbuf):
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

    def copy_guards(self):
        "Copy data to guard cells from corresponding active cells."

        # Make sure boundaries aren't already set
        # TODO: turn this into a warning. Other than communication overhead,
        # calling this multiple times doesn't really cause any harm.
        assert not self.boundaries_set, 'Boundaries are already set!'

        # Short hand
        g = self.grid

        # y-boundaries
        for iy in range(g.lby - 1, -1, -1):
            self[iy, g.lbx:g.ubx] = self[iy + g.nyp, g.lbx:g.ubx]
        for iy in range(g.uby, g.uby + g.lby):
            self[iy, g.lbx:g.ubx] = self[iy - g.nyp, g.lbx:g.ubx]
        self[g.uby:, g.lbx:g.ubx] = self.send_down(self[g.uby:, g.lbx:g.ubx])
        self[:g.lby, g.lbx:g.ubx] = self.send_up(self[:g.lby, g.lbx:g.ubx])

        # x-boundaries
        for ix in range(g.lbx - 1, -1, -1):
            self[:, ix] = self[:, ix + g.nx]
        for ix in range(g.ubx, g.ubx + g.lbx):
            self[:, ix] = self[:, ix - g.nx]

        self.boundaries_set = True

    def add_guards_old(self):
        "Add data from guard cells to corresponding active cells."

        # TODO: We might wanna do some kind of check analogous to the one being
        # done in `copy_guards()`. And here we should probably throw an error
        # if it fails because calling this method multiple times actually does
        # cause harm.

        # Short hand
        g = self.grid

        if self.dtype.names is None:
            # x-boundaries
            self[:, g.lbx:g.lbx+g.lbx] += self[:, g.ubx:]
            self[:, g.ubx-g.lbx:g.ubx] += self[:, :g.lbx]
            # y-boundaries
            self[g.lby:g.lby+g.lby, :] += self.send_up(self[g.uby:, :])
            self[g.uby-g.lby:g.uby, :] += self.send_down(self[:g.lby, :])
        else:
            for dim in self.dtype.names:
                # x-boundaries
                self[:, g.lbx:g.lbx+g.lbx][dim] += self[:, g.ubx:][dim]
                self[:, g.ubx-g.lbx:g.ubx][dim] += self[:, :g.lbx][dim]
                # y-boundaries
                # TODO: reduce number of communications. Separate
                # communications for each vector field component can't be good
                # for performance.
                self[g.lby:g.lby+g.lby, :][dim] += \
                    self.send_up(self[g.uby:, :][dim])
                self[g.uby-g.lby:g.uby, :][dim] += \
                    self.send_down(self[:g.lby, :][dim])

        # Erase guard cells
        # TODO: I suggest we get rid of this. The guard layes will be
        # overwritten anyway by `copy_guards()`.
        self[:g.lby, :] = 0.0
        self[g.uby:, :] = 0.0
        self[:, g.ubx:] = 0.0
        self[:, :g.lbx] = 0.0

    def add_guards(self):
        "Add data from guard cells to corresponding active cells."

        # TODO: We might wanna do some kind of check analogous to the one being
        # done in `copy_guards()`. And here we should probably throw an error
        # if it fails because calling this method multiple times actually does
        # cause harm.

        # Short hand
        g = self.grid

        if self.dtype.names is None:
            # x-boundaries
            for ix in range(g.lbx):
                self[:, ix + g.nx] += self[:, ix]
            for ix in range(g.ubx + g.lbx - 1, g.ubx - 1, -1):
                self[:, ix - g.nx] += self[:, ix]
            # y-boundaries
            self[g.uby:, g.lbx:g.ubx] = \
                self.send_up(self[g.uby:, g.lbx:g.ubx])
            self[:g.lby, g.lbx:g.ubx] = \
                self.send_down(self[:g.lby, g.lbx:g.ubx])
            for iy in range(g.lby):
                self[iy + g.nyp, g.lbx:g.ubx] += self[iy, g.lbx:g.ubx]
            for iy in range(g.uby + g.lby - 1, g.uby - 1, -1):
                self[iy - g.nyp, g.lbx:g.ubx] += self[iy, g.lbx:g.ubx]
        else:
            # x-boundaries
            for dim in self.dtype.names:
                for ix in range(g.lbx):
                    self[:, ix + g.nx][dim] += self[:, ix][dim]
                for ix in range(g.ubx + g.lbx - 1, g.ubx - 1, -1):
                    self[:, ix - g.nx][dim] += self[:, ix][dim]
            # y-boundaries
            self[g.uby:, g.lbx:g.ubx] = \
                self.send_up(self[g.uby:, g.lbx:g.ubx])
            self[:g.lby, g.lbx:g.ubx] = \
                self.send_down(self[:g.lby, g.lbx:g.ubx])
            for dim in self.dtype.names:
                for iy in range(g.lby):
                    self[iy + g.nyp, g.lbx:g.ubx][dim] \
                            += self[iy, g.lbx:g.ubx][dim]
                for iy in range(g.uby + g.lby - 1, g.uby - 1, -1):
                    self[iy - g.nyp, g.lbx:g.ubx][dim] \
                            += self[iy, g.lbx:g.ubx][dim]

        # Erase guard cells
        # TODO: I suggest we get rid of this. The guard layes will be
        # overwritten anyway by `copy_guards()`.
        self[:g.lby, :] = 0.0
        self[g.uby:, :] = 0.0
        self[:, g.ubx:] = 0.0
        self[:, :g.lbx] = 0.0


class ShearField(Field):
    # TODO: ShearField really needs to have a ShearManifold passed, it
    # will fail if a standard Grid or Manifold is passed. Change variable name
    # to manifold or shearmanifold?

    def __new__(cls, grid, time=0.0, **kwds):

        obj = super().__new__(cls, grid, time=time, **kwds)

        # Wave numbers for real-to-complex transforms
        obj.kx = 2*np.pi*np.fft.rfftfreq(grid.nx)/grid.dx

        # Outer product of y and kx
        obj.y_kx = np.outer(grid.y, obj.kx)

        return obj

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj)

        if obj is None:
            return

        self.kx = getattr(obj, "kx", None)
        self.y_kx = getattr(obj, "y_kx", None)

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

    def add_guards(self):

        # Short hand
        g = self.grid

        # Add data from guard cells to corresponding active cells
        if self.dtype.names is None:
            self[:, g.lbx:g.lbx+g.lbx] += self[:, g.ubx:]
            self[:, g.ubx-g.lbx:g.ubx] += self[:, :g.lbx]
        else:
            for dim in self.dtype.names:
                self[:, g.lbx:g.lbx+g.lbx][dim] += self[:, g.ubx:][dim]
                self[:, g.ubx-g.lbx:g.ubx][dim] += self[:, :g.lbx][dim]

        # Translate the y-ghostzones
        if self.grid.comm.rank == self.grid.comm.size - 1:
            trans = self.grid.Ly*self.grid.S*self.time
            for iy in range(g.uby, g.uby + g.lby):
                self._translate_boundary(trans, iy)
        if self.grid.comm.rank == 0:
            trans = -self.grid.Ly*self.grid.S*self.time
            for iy in range(0, g.lby):
                self._translate_boundary(trans, iy)

        # Add data from guard cells to corresponding active cells in y
        if self.dtype.names is None:
            self[g.lby:g.lby+g.lby, :] += self.send_up(self[g.uby:, :])
            self[g.uby-g.lby:g.uby, :] += self.send_down(self[:g.lby, :])
        else:
            for dim in self.dtype.names:
                self[g.lby:g.lby+g.lby, :][dim] += \
                    self.send_up(self[g.uby:, :][dim])
                self[g.uby-g.lby:g.uby, :][dim] += \
                    self.send_down(self[:g.lby, :][dim])

        # Erase guard cells
        self[:g.lby, :] = 0.0
        self[g.uby:, :] = 0.0
        self[:, g.ubx:] = 0.0
        self[:, :g.lbx] = 0.0

    def copy_guards(self):

        msg = 'Boundaries are already set!'
        assert not self.boundaries_set, msg

        # Short hand
        g = self.grid

        # lower active cells to upper guard layers
        self[g.uby:, g.lbx:g.ubx] = \
            self.send_down(self[g.lby:g.lby+g.lby, g.lbx:g.ubx])
        # upper active cells to lower guard layers
        self[:g.lby, g.lbx:g.ubx] = \
            self.send_up(self[g.uby-g.lby:g.uby, g.lbx:g.ubx])
        # lower active cells to upper guard layers
        self[:, g.ubx:] = self[:, g.lbx:g.lbx+g.lbx]
        # upper active cells to lower guard layers
        self[:, :g.lbx] = self[:, g.ubx-g.lbx:g.ubx]

        # Translate the y-ghostzones
        if self.grid.comm.rank == self.grid.comm.size - 1:
            trans = -self.grid.Ly*self.grid.S*self.time
            for iy in range(g.uby, g.uby + g.lby):
                self._translate_boundary(trans, iy)
        if self.grid.comm.rank == 0:
            trans = +self.grid.Ly*self.grid.S*self.time
            for iy in range(0, g.lby):
                self._translate_boundary(trans, iy)

        self.boundaries_set = True

    def translate(self, time):
        """Translation using numpy's fft."""

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

        for dim in ('x', 'y', 'z'):
            # Fourier transform along x
            fx_hat = np.fft.rfft(self.trim()[dim], axis=1)

            # Translate along x by an amount -S*t*y
            fx_hat *= np.exp(1j*self.grid.S*time*self.y_kx)

            # Inverse Fourier transform along x
            self.active[dim] = np.fft.irfft(fx_hat, axis=1)

        # Set boundary condition?
        self.boundaries_set = False
