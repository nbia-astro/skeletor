from .field import Field
from .cython.ppic2_wrapper import cppgpost2l


class Sources:

    def __init__(self, grid, order='cic', **kwds):

        self.rho = Field(grid, **kwds)

        # Interpolation order
        self.order = order

    def deposit(self, particles, erase=True):
        err = 'Too few bondary layers for the chosen deposition'
        # Cloud-In-Cell deposition (CIC)
        if self.order == 'cic':
            assert (self.rho.grid.nlbx >= 0 and self.rho.grid.nlby >= 0 and
                    self.rho.grid.nubx >= 1 and self.rho.grid.nuby >= 1), err
            from .cython.deposit import deposit_cic as cython_deposit
        # Triangular Shaped Cloud deposition (TSC)
        elif self.order == 'tsc':
            assert (self.rho.grid.nlbx >= 1 and self.rho.grid.nlby >= 1 and
                    self.rho.grid.nubx >= 2 and self.rho.grid.nuby >= 2), err
            from .cython.deposit import deposit_tsc as cython_deposit
        else:
            msg = 'Interpolation order not supported. order = {}'
            raise RuntimeError(msg.format(self.order))

        if erase:
            self.rho.fill(0.0)

        cython_deposit(
                particles[:particles.np], self.rho, particles.charge,
                self.rho.grid.noff, self.rho.grid.lbx, self.rho.grid.lby)

    def deposit_tsc(self, particles, erase=True):
        from .cython.deposit import deposit_tsc

        if erase:
            self.rho.fill(0.0)

        deposit_tsc(
                particles[:particles.np], self.rho, particles.charge,
                self.rho.grid.noff, self.rho.grid.lbx, self.rho.grid.lby)

    def deposit_ppic2(self, particles, erase=True):

        if erase:
            self.rho.fill(0.0)

        cppgpost2l(
                particles, self.rho, particles.np, self.rho.grid.noff,
                particles.charge, particles.size)
