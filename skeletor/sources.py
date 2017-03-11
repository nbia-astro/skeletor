from .cython.deposit import deposit as cython_deposit
from .cython.types import Float4


class Sources:

    def __init__(self, manifold, npc, **kwds):

        # Particles per cell
        self.npc = npc

        if not manifold.shear:
            from .field import Field
        else:
            from .field import ShearField as Field

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

        # TODO: Add the manifold as attribute to this class and use this here
        # instead of self.current's grid. "S" and "shear" aren't actually
        # attributes of the Grid class
        if not self.current.grid.shear:
            S = 0.0
        else:
            S = self.current.grid.S

        cython_deposit(particles[:particles.np], self.current,
                       self.current.grid, S)

        self.current.boundaries_set = False

        self.normalize(particles.charge)

        if set_boundaries:
            self.set_boundaries()

    def normalize(self, charge):
        """
        Normalize charge and current with number of particles per cell and
        the charge per particle
        """
        for dim in self.current.dtype.names:
            self.current[dim] *= charge/self.npc

    def set_boundaries(self):

            # Add guards
            self.current.add_guards()
            # Copy guards
            self.current.copy_guards()
