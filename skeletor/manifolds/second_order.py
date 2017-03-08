from ..grid import Grid


class Manifold(Grid):

    """Finite difference operators"""

    def __init__(self, nx, ny, comm, ax=0.0, ay=0.0, **grid_kwds):

        super().__init__(nx, ny, comm, **grid_kwds)

        err = 'Not enough guard layers for second order finite difference.'
        assert self.lbx >= 1 and self.lby >= 1, err

        msg = 'Finite particle size not implemented in this manifold'
        assert ax == 0.0 and ay == 0.0, msg

        # Rotation and shear is always false for this manifold
        self.shear = False
        self.rotation = False

    def gradient(self, f, grad):
        """Calculate the gradient of f"""

        msg = 'Boundaries need to be set on f for second order differences'
        assert f.boundaries_set, msg
        from ..cython.finite_difference import gradient as cython_gradient

        cython_gradient(f, grad, self)

        grad.boundaries_set = False

    def curl(self, f, curl, down=True):
        """Calculate the curl of f"""

        msg = 'Boundaries need to be set on f for second order differences'
        assert f.boundaries_set, msg

        if down:
            from ..cython.finite_difference import curl_down as cython_curl
        else:
            from ..cython.finite_difference import curl_up as cython_curl

        cython_curl(f['x'], f['y'], f['z'], curl, self)

        curl.boundaries_set = False

    def unstagger(self, f, g, set_boundaries=False):
        """Interpolate the staggered field f to cell centers"""

        msg = 'Boundaries need to be set on f for interpolation'
        assert f.boundaries_set, msg
        from ..cython.finite_difference import unstagger as cython_unstagger

        cython_unstagger(f['x'], f['y'], f['z'], g, self)

        g.boundaries_set = False

        if set_boundaries:
            g.copy_guards()

    def stagger(self, f, g, set_boundaries=False):
        """Interpolate the cell-centered field f to cell corners"""

        msg = 'Boundaries need to be set on f for interpolation'
        assert f.boundaries_set, msg
        from ..cython.finite_difference import stagger as cython_stagger

        cython_stagger(f['x'], f['y'], f['z'], g, self)

        g.boundaries_set = False

        if set_boundaries:
            g.copy_guards()

    def divergence(self, f, g):
        """Calculate the divergence of the vector field f"""

        from ..cython.finite_difference import divergence

        divergence(f['x'], f['y'], g, self)

    def log(self, f):
        from numpy import log as numpy_log

        return numpy_log(f)

    def grad_inv_del(self, qe, fxye):

        msg = "grad_inv_del not implemented for 2nd order finite difference."
        raise NotImplementedError(msg)


class ShearingManifold(Manifold):

    """Finite difference operators in the shearing sheet"""

    def __init__(self, nx, ny, comm, S=0, Omega=0, **manifold_kwds):

        super().__init__(nx, ny, comm, **manifold_kwds)

        # Shear parameter
        self.S = S
        # True if shear is turned on
        self.shear = (S != 0)

        # Angular frequency
        self.Omega = Omega
        self.rotation = (Omega != 0)
