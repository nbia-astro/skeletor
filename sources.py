from fields import Field, GlobalField
from deposit import deposit as cython_deposit


class Sources:

    def __init__(self, grid, **kwds):

        self.rho = Field(grid, **kwds)
        self.noff = grid.noff

    def deposit(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cython_deposit(particles[:particles.np], self.rho, self.noff)


class GlobalSources(Sources):

    def __init__(self, grid, **kwds):

        self.rho = GlobalField(grid, **kwds)

    def deposit(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cython_deposit(particles, self.rho, 0)
