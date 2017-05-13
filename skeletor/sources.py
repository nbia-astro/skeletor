from .cython.deposit import deposit as cython_deposit
from .cython.types import Float4


class Sources:

    def __init__(self, manifold, **kwds):

        from .field import Field

        # Electric four-current density (rho, Jx, Jy, Jz)
        self.current = Field(manifold, dtype=Float4, **kwds)

    @property
    def rho(self):
        return self.current['t']

    @property
    def Jx(self):
        return self.current['x']

    @property
    def Jy(self):
        return self.current['y']

    @property
    def Jz(self):
        return self.current['z']

    def deposit(self, particles, erase=True, set_boundaries=False):

        if erase:
            self.current.fill((0.0, 0.0, 0.0, 0.0))

        # Short hand
        grid = self.current.grid

        # TODO: Add the manifold as attribute to this class and use this here
        # instead of self.current's grid. "S" and "shear" aren't actually
        # attributes of the Grid class
        if not grid.shear:
            S = 0.0
        else:
            S = grid.S

        cython_deposit(particles[:particles.N], self.current, grid, S)

        self.current.boundaries_set = False

        self.normalize(particles)

        if set_boundaries:
            self.set_boundaries()

    def normalize(self, particles):
        """Normalize the charge and current densities such that the mean charge
        density is equal to particle.n0."""
        # TODO: The total number of particles used below does not change in
        # time and thus does not need to be calculated over and over again.
        from mpi4py.MPI import SUM

        grid = self.current.grid

        N = grid.comm.allreduce(particles.N, op=SUM)

        fac = particles.charge*particles.n0*grid.nx*grid.ny/N
        for dim in self.current.dtype.names:
            self.current[dim] *= fac

    def set_boundaries(self):

            # Add guards
            self.current.add_guards()
            # Copy guards
            self.current.copy_guards()
