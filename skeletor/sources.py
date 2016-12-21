from .cython.deposit import deposit as cython_deposit
from .cython.ppic2_wrapper import cppgpost2l
from .cython.dtypes import Float, Float2


class Sources:

    def __init__(self, manifold, **kwds):

        if not manifold.shear:
            from .field import Field
            self.rho = Field(manifold, dtype=Float, **kwds)
            self.J = Field(manifold, dtype=Float2, **kwds)
        elif manifold.shear:
            from .field import ShearField
            self.rho = ShearField(manifold, dtype=Float, **kwds)
            self.J = ShearField(manifold, dtype=Float2, **kwds)

    def deposit(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)
            self.J.fill((0.0, 0.0))

        cython_deposit(
                particles[:particles.np], self.rho, self.J, particles.charge,
                self.rho.grid.noff, self.rho.grid.lbx, self.rho.grid.lby)

        self.rho.boundaries_set = False
        self.J.boundaries_set = False

    def deposit_ppic2(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cppgpost2l(
                particles, self.rho, particles.np, self.rho.grid.noff,
                particles.charge, particles.size)
