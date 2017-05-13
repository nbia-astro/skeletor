from .field import Field
from .cython.deposit import deposit as cython_deposit
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

        if erase:
            self.fill((0.0, 0.0, 0.0, 0.0))

        # Rate of shear
        S = getattr(self.grid, 'S', 0.0)

        cython_deposit(particles[:particles.N], self, self.grid, S)

        self.boundaries_set = False

        self.normalize(particles)

        if set_boundaries:
            self.set_boundaries()

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
