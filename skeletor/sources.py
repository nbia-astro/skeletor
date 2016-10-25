from .field import Field
from .cython.deposit import deposit as cython_deposit
from .cython.ppic2_wrapper import cppgpost2l


class Sources:

    def __init__(self, grid, comm, **kwds):

        self.rho = Field(grid, comm, **kwds)

    def deposit(self, particles, St=0, erase=True):

        if erase:
            self.rho.fill(0.0)

        grid = self.rho.grid

        cython_deposit(
                particles[:particles.np], self.rho,
                particles.charge, self.rho.grid.noff,
                grid.nx, grid.ny, St)

    def deposit_ppic2(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cppgpost2l(
                particles, self.rho, particles.np, self.rho.grid.noff,
                particles.charge, particles.size)
