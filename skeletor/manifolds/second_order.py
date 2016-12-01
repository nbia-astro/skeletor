from ..grid import Grid


class Manifold(Grid):

    """Finite difference operators"""

    def __init__(
            self, nx, ny, comm,
            ax=0.0, ay=0.0, nlbx=1, nubx=1, nlby=1, nuby=1, Lx=None, Ly=None):

        super().__init__(
                nx, ny, comm, nlbx=nlbx, nubx=nubx, nlby=nlby, nuby=nuby,
                Lx=Lx, Ly=Ly)

        err = 'Not enough boundary layers for second order finite difference.'
        assert (nlbx >= 1 and nlby >= 1 and nubx >= 1 and nuby >= 1), err

        msg = 'Finie particle size not implemented in this manifold'
        assert ax == 0.0 and ay == 0.0, msg

        # Rotation and shear is always false for this manifold
        self.shear = False
        self.rotation = False

    def gradient(self, f, grad):
        """Calculate the gradient of f"""

        msg = 'Boundaries need to be set on f for second order differences'
        assert f.boundaries_set, msg
        from ..cython.finite_difference import gradient as cython_gradient

        cython_gradient(f, grad, self.lbx, self.ubx, self.lby, self.uby)

        grad.boundaries_set = False

        # Divide by dx and dy to account for nx != Lx and ny != Ly
        grad['x'] /= self.dx
        grad['y'] /= self.dy


    def log(self, f):
        from numpy import log as numpy_log

        return numpy_log(f)

    def ddxn(self, f):
        from ..cython.finite_difference import ddxdn
        from numpy import zeros_like

        df = zeros_like(f)

        ddxdn(f, df, self.lbx, self.ubx, self.lby, self.uby)

        df /= self.dx

        return df

    def ddyn(self, f):
        from ..cython.finite_difference import ddydn
        from numpy import zeros_like

        df = zeros_like(f)

        ddydn(f, df, self.lbx, self.ubx, self.lby, self.uby)

        df /= self.dy

        return df

    def curl(self, f, curl):
        """Calculate the curl of vector f"""

        curl['x'] = self.ddyn(f['z'])
        curl['y'] = -self.ddxn(f['z'])
        curl['z'] = self.ddxn(f['y']) - self.ddyn(f['x'])

    def grad_inv_del(self, qe, fxye):

        msg = "grad_inv_del not implemented for 2nd order finite difference."
        raise NotImplementedError(msg)


class ShearingManifold(Manifold):

    """Finite difference operators in the shearing sheet"""

    def __init__(
            self, nx, ny, comm,
            ax=0.0, ay=0.0, nlbx=1, nubx=1, nlby=1, nuby=1, S=0, Omega=0,
            Lx=None, Ly=None):

        super().__init__(
                nx, ny, comm, nlbx=nlbx, nubx=nubx, nlby=nlby, nuby=nuby,
                Lx=Lx, Ly=Ly)

        # Shear parameter
        self.S = S
        # True if shear is turned on
        self.shear = (S != 0)

        # Angular frequency
        self.Omega = Omega
        self.rotation = (Omega != 0)
