class Operators:

    """Finite difference operators"""

    def __init__(self, grid, ax, ay, np):

        from math import log2

        self.indx = int(log2(grid.nx))
        self.indy = int(log2(grid.ny))

        assert grid.nx == 2**self.indx, "'nx' needs to be a power of two"
        assert grid.ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

        # Normalization constant
        self.affp = grid.nx*grid.ny/np

        # Grid
        self.grid = grid

    def gradient(self, f, grad):
        """Calculate the gradient of f"""
        from ..cython.finite_difference import gradient as cython_gradient

        cython_gradient(f, grad, self.grid.lbx, self.grid.ubx, self.grid.lby,
                        self.grid.uby)

    def ddxn(self, f):
        from ..cython.finite_difference import ddxdn
        from numpy import zeros_like

        df = zeros_like(f)

        ddxdn(f, df, self.grid.lbx, self.grid.ubx, self.grid.lby,
              self.grid.uby)

        return df

    def ddyn(self, f):
        from ..cython.finite_difference import ddydn
        from numpy import zeros_like

        df = zeros_like(f)

        ddydn(f, df, self.grid.lbx, self.grid.ubx, self.grid.lby,
              self.grid.uby)

        return df

    def curl(self, f, curl):
        """Calculate the curl of vector f"""

        curl['x'] = self.ddyn(f['z'])
        curl['y'] = -self.ddxn(f['z'])
        curl['z'] = self.ddxn(f['y']) - self.ddyn(f['x'])


    def grad_inv_del(self, qe, fxye):

        raise RuntimeError("grad_inv_del not implemented for 2nd order finite \
                           difference.")