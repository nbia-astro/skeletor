from ..grid import Grid
import warnings

class Manifold(Grid):

    """Finite difference operators"""

    def __init__(self, nx, ny, comm, ax, ay, nlbx=0, nubx=2, nlby=0, nuby=1):

        super().__init__(nx, ny, comm, nlbx, nubx, nlby, nuby)

        from math import log2

        err = 'Not enough boundary layers for second order finite difference.'
        assert (nlbx >= 1 and nlby >= 1 and nubx >= 1 and nuby >= 1), err

        self.indx = int(log2(nx))
        self.indy = int(log2(ny))

        assert nx == 2**self.indx, "'nx' needs to be a power of two"
        assert ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

    def gradient(self, f, grad, destroy_input=None):
        """Calculate the gradient of f"""
        from ..cython.finite_difference import gradient as cython_gradient

        if destroy_input is not None:
            warnings.warn("Ignoring option 'destroy_input'.")

        cython_gradient(f, grad, self.lbx, self.ubx, self.lby, self.uby)

    def grad_inv_del(self, qe, fxye):

        raise RuntimeError("grad_inv_del not implemented for 2nd order finite \
                           difference.")
