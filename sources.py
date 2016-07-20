from fields import Field, GlobalField
from deposit import deposit as cython_deposit
from ppic2_wrapper import cppgpost2l


class Sources:

    def __init__(self, grid, **kwds):

        self.rho = Field(grid, **kwds)

    def deposit(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cython_deposit(
                particles[:particles.np], self.rho, self.rho.grid.noff)

    def deposit_ppic2(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cppgpost2l(
                particles, self.rho, particles.np, self.rho.grid.noff,
                particles.size)


class GlobalSources(Sources):

    def __init__(self, grid, **kwds):

        self.rho = GlobalField(grid, **kwds)

    def deposit(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cython_deposit(particles, self.rho, 0)
