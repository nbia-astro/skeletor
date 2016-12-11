from ..grid import Grid


class Manifold(Grid):

    """Finite difference operators"""

    def __init__(
            self, nx, ny, comm,
            ax=0.0, ay=0.0, nlbx=1, nubx=1, nlby=1, nuby=1):

        super().__init__(
                nx, ny, comm, nlbx=nlbx, nubx=nubx, nlby=nlby, nuby=nuby)

        err = 'Not enough boundary layers for second order finite difference.'
        assert (nlbx >= 1 and nlby >= 1 and nubx >= 1 and nuby >= 1), err

        msg = 'Finie particle size not implemented in this manifold'
        assert ax == 0.0 and ay == 0.0, msg

    def gradient(self, f, grad):
        """Calculate the gradient of f"""
        from ..cython.finite_difference import gradient as cython_gradient

        cython_gradient(f, grad, self.lbx, self.ubx, self.lby, self.uby)

    def log(self, f):
        from numpy import log as numpy_log
        return numpy_log(f)

    def grad_inv_del(self, qe, fxye):

        msg = "grad_inv_del not implemented for 2nd order finite difference."
        raise NotImplementedError(msg)
