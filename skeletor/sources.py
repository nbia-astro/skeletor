from .field import Field
from .cython.ppic2_wrapper import cppgpost2l
from .cython.dtypes import Float2


class Sources:

    def __init__(self, grid, order='cic', **kwds):

        self.rho = Field(grid, **kwds)

        self.J = Field(grid, dtype=Float2)

        # Interpolation order
        self.order = order

    def deposit(self, particles, erase=True):
        if self.order == 'cic':
            from .cython.deposit import deposit_cic as cython_deposit
        elif self.order == 'tsc':
            from .cython.deposit import deposit_tsc as cython_deposit
        else:
            msg = 'Interpolation order not supported. order = {}'
            raise RuntimeError(msg.format(self.order))

        if erase:
            self.rho.fill(0.0)
            self.J.fill((0.0, 0.0, 0.0))

        cython_deposit(
                particles[:particles.np], self.rho, self.J, particles.charge,
                self.rho.grid.noff, self.rho.grid.lbx, self.rho.grid.lby)

    def deposit_tsc(self, particles, erase=True):
        from .cython.deposit import deposit_tsc

        if erase:
            self.rho.fill(0.0)
            self.J.fill((0.0, 0.0, 0.0))

        deposit_tsc(
                particles[:particles.np], self.rho, particles.charge,
                self.rho.grid.noff, self.rho.grid.lbx, self.rho.grid.lby)

    def deposit_ppic2(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cppgpost2l(
                particles, self.rho, particles.np, self.rho.grid.noff,
                particles.charge, particles.size)
