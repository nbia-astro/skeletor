from numpy import ndarray, asarray, zeros
from .cython.types import Float


class Field(ndarray):

    def __new__(cls, grid, time=0.0, **kwds):

        # Field size set accoring to number of guard layes
        obj = super().__new__(cls, shape=(grid.myp, grid.mx), **kwds)

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
        "Copy data to guard cells from corresponding active cells."

        # Make sure boundaries aren't already set
        # TODO: turn this into a warning. Other than communication overhead,
        # calling this multiple times doesn't really cause any harm.
        assert not self.boundaries_set, 'Boundaries are already set!'

        lbx = self.grid.lbx
        lby = self.grid.lby
        ubx = self.grid.ubx
        uby = self.grid.uby

        # x-boundaries
        self[uby:, lbx:ubx] = self.send_down(self[lby:lby+lby, lbx:ubx])
        self[:lby, lbx:ubx] = self.send_up(self[uby-lby:uby, lbx:ubx])

        # y-boundaries
        self[:, ubx:] = self[:, lbx:lbx+lbx]
        self[:, :lbx] = self[:, ubx-lbx:ubx]

        self.boundaries_set = True

    def add_guards(self):
        "Add data from guard cells to corresponding active cells."

        # TODO: We might wanna do some kind of check analogous to the one being
        # done in `copy_guards()`. And here we should probably throw an error
        # if it fails because calling this method multiple times actually does
        # cause harm.

        lbx = self.grid.lbx
        lby = self.grid.lby
        ubx = self.grid.ubx
        uby = self.grid.uby

        if self.dtype.names is None:
            # x-boundaries
            self[:, lbx:lbx+lbx] += self[:, ubx:]
            self[:, ubx-lbx:ubx] += self[:, :lbx]
            # y-boundaries
            self[lby:lby+lby, :] += self.send_up(self[uby:, :])
            self[uby-lby:uby, :] += self.send_down(self[:lby, :])
        else:
            for dim in self.dtype.names:
                # x-boundaries
                self[:, lbx:lbx+lbx][dim] += self[:, ubx:][dim]
                self[:, ubx-lbx:ubx][dim] += self[:, :lbx][dim]
                # y-boundaries
                # TODO: reduce number of communications. Separate
                # communications for each vector field component can't be good
                # for performance.
                self[lby:lby+lby, :][dim] += self.send_up(self[uby:, :][dim])
                self[uby-lby:uby, :][dim] += self.send_down(self[:lby, :][dim])

        # Erase guard cells
        # TODO: I suggest we get rid of this. The guard layes will be
        # overwritten anyway by `copy_guards()`.
        self[:lby, :] = 0.0
        self[uby:, :] = 0.0
        self[:, ubx:] = 0.0
        self[:, :lbx] = 0.0


class ShearField(Field):
    # TODO: ShearField really needs to have a ShearManifold passed, it
    # will fail if a standard Grid or Manifold is passed. Change variable name
    # to manifold or shearmanifold?

    def __new__(cls, grid, time=0.0, **kwds):

        from numpy.fft import rfftfreq
        from numpy import outer, pi

        obj = super().__new__(cls, grid, time=time, **kwds)

        # Wave numbers for real-to-complex transforms
        obj.kx = 2*pi*rfftfreq(grid.nx)/grid.dx

        # Outer product of y and kx
        obj.y_kx = outer(grid.y, obj.kx)

        return obj

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj)

        if obj is None:
            return

        self.kx = getattr(obj, "kx", None)
        self.y_kx = getattr(obj, "y_kx", None)

    def _translate_boundary(self, trans, iy):

        "Translation using FFTs"
        from numpy.fft import rfft, irfft
        from numpy import exp

        lbx = self.grid.lbx
        ubx = self.grid.ubx

        # Translate in real space by phase shifting in spectral space
        phase = -1j*self.kx*trans
        if self.dtype.names is None:
            self[iy, self.grid.lbx:self.grid.ubx] = \
                irfft(exp(phase)*rfft(self[iy, self.grid.lbx:self.grid.ubx]))
        else:
            for dim in self.dtype.names:
                self[iy, self.grid.lbx:self.grid.ubx][dim] = \
                    irfft(exp(phase) *
                          rfft(self[iy, self.grid.lbx:self.grid.ubx][dim]))

        # Update x-boundaries
        self[iy, ubx:] = self[iy, lbx:lbx+lbx]
        self[iy, :lbx] = self[iy, ubx-lbx:ubx]

    def add_guards(self):

        lbx = self.grid.lbx
        lby = self.grid.lby
        ubx = self.grid.ubx
        uby = self.grid.uby

        # Add data from guard cells to corresponding active cells
        if self.dtype.names is None:
            self[:, lbx:lbx+lbx] += self[:, ubx:]
            self[:, ubx-lbx:ubx] += self[:, :lbx]
        else:
            for dim in self.dtype.names:
                self[:, lbx:lbx+lbx][dim] += self[:, ubx:][dim]
                self[:, ubx-lbx:ubx][dim] += self[:, :lbx][dim]

        # Translate the y-ghostzones
        if self.grid.comm.rank == self.grid.comm.size - 1:
            trans = self.grid.Ly*self.grid.S*self.time
            for iy in range(uby, uby + lby):
                self._translate_boundary(trans, iy)
        if self.grid.comm.rank == 0:
            trans = -self.grid.Ly*self.grid.S*self.time
            for iy in range(0, lby):
                self._translate_boundary(trans, iy)

        # Add data from guard cells to corresponding active cells in y
        if self.dtype.names is None:
            self[lby:lby+lby, :] += self.send_up(self[uby:, :])
            self[uby-lby:uby, :] += self.send_down(self[:lby, :])
        else:
            for dim in self.dtype.names:
                self[lby:lby+lby, :][dim] += self.send_up(self[uby:, :][dim])
                self[uby-lby:uby, :][dim] += self.send_down(self[:lby, :][dim])

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
        ubx = self.grid.ubx
        uby = self.grid.uby

        # lower active cells to upper guard layers
        self[uby:, lbx:ubx] = self.send_down(self[lby:lby+lby, lbx:ubx])
        # upper active cells to lower guard layers
        self[:lby, lbx:ubx] = self.send_up(self[uby-lby:uby, lbx:ubx])
        # lower active cells to upper guard layers
        self[:, ubx:] = self[:, lbx:lbx+lbx]
        # upper active cells to lower guard layers
        self[:, :lbx] = self[:, ubx-lbx:ubx]

        # Translate the y-ghostzones
        if self.grid.comm.rank == self.grid.comm.size - 1:
            trans = -self.grid.Ly*self.grid.S*self.time
            for iy in range(uby, uby + lby):
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

    def translate_vector(self, time):
        """Translation using numpy's fft."""
        from numpy.fft import rfft, irfft
        from numpy import exp

        for dim in ('x', 'y', 'z'):
            # Fourier transform along x
            fx_hat = rfft(self.trim()[dim], axis=1)

            # Translate along x by an amount -S*t*y
            fx_hat *= exp(1j*self.grid.S*time*self.y_kx)

            # Inverse Fourier transform along x
            self.active[dim] = irfft(fx_hat, axis=1)

        # Set boundary condition?
        self.boundaries_set = False
