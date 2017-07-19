from .field import Field
from .cython.types import Float4


class Sources(Field):

    def __new__(cls, manifold, **kwds):
        obj = super().__new__(cls, manifold, dtype=Float4, **kwds)
        return obj

    @property
    def rho(self):
        return self['t']

    @property
    def Jx(self):
        return self['x']

    @property
    def Jy(self):
        return self['y']

    @property
    def Jz(self):
        return self['z']

    def deposit(self, particles, erase=True, set_boundaries=False):

        if particles.order == 1:
            from .cython.deposit import deposit_cic as cython_deposit
        elif particles.order == 2:
            from .cython.deposit import deposit_tsc as cython_deposit
        else:
            msg = 'Interpolation order {} not implemented.'
            raise RuntimeError(msg.format(particles.order))

        if erase:
            self.fill((0.0, 0.0, 0.0, 0.0))

        # Rate of shear
        S = getattr(self.grid, 'S', 0.0)

        cython_deposit(particles[:particles.N], self, self.grid, S)

        self.boundaries_set = False

        # self.normalize(particles)

        if set_boundaries:
            self.set_boundaries()

    def deposit_fix(self, particles):

        if particles.order == 1:
            from .cython.deposit import deposit_cic_fix as cython_deposit
        elif particles.order == 2:
            from .cython.deposit import deposit_tsc_fix as cython_deposit
        else:
            msg = 'Interpolation order {} not implemented.'
            raise RuntimeError(msg.format(particles.order))

        assert not self.boundaries_set

        if self.grid.comm.rank == 0:
            self[:self.grid.lby, :] = (0.0, 0.0, 0.0, 0.0)
        if self.grid.comm.rank == self.grid.comm.size - 1:
            self[self.grid.uby:, :] = (0.0, 0.0, 0.0, 0.0)

        # Rate of shear
        S = getattr(self.grid, 'S', 0.0)

        t = particles.time
        cython_deposit(particles[:particles.N], self, self.grid, S, t)

        self.normalize(particles)

    def normalize(self, particles):
        """Normalize the charge and current densities such that the mean charge
        density is equal to particle.n0."""
        # TODO: The total number of particles used below does not change in
        # time and thus does not need to be calculated over and over again.
        from mpi4py.MPI import SUM

        N = self.grid.comm.allreduce(particles.N, op=SUM)

        fac = particles.charge*particles.n0*self.grid.nx*self.grid.ny/N
        for dim in self.dtype.names:
            self[dim] *= fac

    def set_boundaries(self):

        # Add guards
        self.add_guards()
        # Copy guards
        self.copy_guards()

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
        # NOTE: The following would be more efficient, but is not as generic
        # because it only works for homogeneous floating point types.
        #   self.view(Float) += other.view(Float)
        return self

    def add_guards_x(self):
        """
        Add data from guard cells in x to corresponding active cells.
        Note: this acts on _all_ grid points along y (active plus guards).
        """
        g = self.grid

        for ix in range(g.lbx):
            self[:, ix + g.nx] += self[:, ix]
        for ix in range(g.ubx + g.lbx - 1, g.ubx - 1, -1):
            self[:, ix - g.nx] += self[:, ix]

    def add_guards_y(self):
        """
        Add data from guard cells in y to corresponding active cells.
        Note: this only acts on active grid points along x.
        """
        g = self.grid

        self[g.uby:, g.lbx:g.ubx] = self.send_up(self[g.uby:, g.lbx:g.ubx])
        self[:g.lby, g.lbx:g.ubx] = self.send_dn(self[:g.lby, g.lbx:g.ubx])
        for iy in range(g.lby):
            self[iy + g.nyp, g.lbx:g.ubx] += self[iy, g.lbx:g.ubx]
        for iy in range(g.uby + g.lby - 1, g.uby - 1, -1):
            self[iy - g.nyp, g.lbx:g.ubx] += self[iy, g.lbx:g.ubx]

    def add_guards(self):
        "Add data from guard cells to corresponding active cells."

        # TODO: We might wanna do some kind of check analogous to the one being
        # done in `copy_guards()`. And here we should probably throw an error
        # if it fails because calling this method multiple times actually does
        # cause harm.

        # x-boundaries
        self.add_guards_x()

        if self.shear:
            # Short hand
            g = self.grid
            # Translate the y-ghostzones
            if g.comm.rank == g.comm.size - 1:
                trans = g.Ly*g.S*self.time
                for iy in range(g.uby, g.uby + g.lby):
                    self._translate_boundary(trans, iy)
            if g.comm.rank == 0:
                trans = -g.Ly*g.S*self.time
                for iy in range(0, g.lby):
                    self._translate_boundary(trans, iy)

        # y-boundaries
        self.add_guards_y()

        # Erase guard cells
        # TODO: I suggest we get rid of this. The guard layes will be
        # overwritten anyway by `copy_guards()`.
        self[:self.grid.lby, :] = 0.0
        self[self.grid.uby:, :] = 0.0
        self[:, self.grid.ubx:] = 0.0
        self[:, :self.grid.lbx] = 0.0

    def add_guards_old(self):
        "Add data from guard cells to corresponding active cells."
        # NOTE: This is for testing purposes only!
        # NOTE: This routine does not take shear into account.

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
            self[g.uby-g.lby:g.uby, :] += self.send_dn(self[:g.lby, :])
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
                    self.send_dn(self[:g.lby, :][dim])

        # Erase guard cells
        # TODO: I suggest we get rid of this. The guard layes will be
        # overwritten anyway by `copy_guards()`.
        self[:g.lby, :] = 0.0
        self[g.uby:, :] = 0.0
        self[:, g.ubx:] = 0.0
        self[:, :g.lbx] = 0.0
