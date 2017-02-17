from .cython.deposit import deposit as cython_deposit
from .cython.types import Float, Float2


class Sources:

    def __init__(self, manifold, **kwds):

        if not manifold.shear:
            from .field import Field
        else:
            from .field import ShearField as Field

        self.rho = Field(manifold, dtype=Float, **kwds)
        self.J = Field(manifold, dtype=Float2, **kwds)

    def deposit(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)
            self.J.fill((0.0, 0.0))

        if not self.rho.grid.shear:
            S = 0.0
        else:
            S = self.rho.grid.S

        cython_deposit(particles[:particles.np], self.rho, self.J,
                       particles.charge, self.rho.grid, S)

        self.rho.boundaries_set = False
        self.J.boundaries_set = False
