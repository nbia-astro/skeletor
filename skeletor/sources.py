from .cython.deposit import deposit as cython_deposit
from .cython.types import Float, Float3


class Sources:

    def __init__(self, manifold, npc, **kwds):

        # Particles per cell
        self.npc = npc

        if not manifold.shear:
            from .field import Field
        else:
            from .field import ShearField as Field

        self.rho = Field(manifold, dtype=Float, **kwds)
        self.J = Field(manifold, dtype=Float3, **kwds)

    def deposit(self, particles, erase=True, set_boundaries=False):

        if erase:
            self.rho.fill(0.0)
            self.J.fill((0.0, 0.0, 0.0))

        if not self.rho.grid.shear:
            S = 0.0
        else:
            S = self.rho.grid.S

        cython_deposit(particles[:particles.np], self.rho, self.J,
                       self.rho.grid, S)

        self.rho.boundaries_set = False
        self.J.boundaries_set = False

        self.normalize(particles.charge)

        if set_boundaries:
            self.set_boundaries()


    def normalize(self, charge):
        """
        Normalize charge and current with number of particles per cell and
        the charge per particle
        """
        self.rho *= charge/self.npc
        for dim in ('x', 'y', 'z'):
            self.J[dim] *= charge/self.npc

    def set_boundaries(self):

            # Add guards
            self.rho.add_guards()
            self.J.add_guards()
            # Copy guards
            self.rho.copy_guards()
            self.J.copy_guards()
