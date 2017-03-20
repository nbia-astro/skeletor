from ..grid import Grid
import numpy as np


class Manifold(Grid):

    """Finite difference operators"""

    def __init__(self, nx, ny, comm,
                 ax=0.0, ay=0.0, custom_cppois22=True, **grid_kwds):

        super().__init__(nx, ny, comm, **grid_kwds)

        err = 'Not enough guard layers for second order finite difference.'
        assert self.lbx >= 1 and self.lby >= 1, err

        # Rotation and shear is always false for this manifold
        self.shear = False
        self.rotation = False

        # Initialize Poisson solver (see definition below)
        self.grad_inv_del = PoissonSolver(self, ax=ax, ay=ay,
                                          custom_cppois22=custom_cppois22)

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
        return np.log(f)


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


class PoissonSolver:

    def __init__(self, grid, ax=0.0, ay=0.0, custom_cppois22=True):

        from ..cython.types import Complex, Complex2, Float, Float2, Int
        from ..cython.ppic2_wrapper import cwpfft2rinit, cppois22
        from ..cython.operators import calc_form_factors

        self.indx = int(np.log2(grid.nx))
        self.indy = int(np.log2(grid.ny))

        assert grid.nx == 2**self.indx, "'nx' needs to be a power of two"
        assert grid.ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

        # Normalization constant
        self.affp = 1.0

        # Use PPIC2's cppois22() routine for solving Poisson's equation?
        # This restricts the grid to be isotropic (i.e. dx = dy)
        self.custom_cppois22 = custom_cppois22

        # Store grid
        self.grid = grid

        nxh = grid.nx//2
        nyh = (1 if 1 > grid.ny//2 else grid.ny//2)
        nxhy = (nxh if nxh > grid.ny else grid.ny)
        nxyh = (grid.nx if grid.nx > grid.ny else grid.ny)//2
        nye = grid.ny + 2
        kxp = (nxh - 1)//grid.comm.size + 1
        kyp = (grid.ny - 1)//grid.comm.size + 1

        self.qt = np.zeros((kxp, nye), Complex)
        self.fxyt = np.zeros((kxp, nye), Complex2)
        self.mixup = np.zeros(nxhy, Int)
        self.sct = np.zeros(nxyh, Complex)
        self.ffc = np.zeros((kxp, nyh), Complex)
        self.bs = np.zeros((kyp, kxp), Complex2)
        self.br = np.zeros((kyp, kxp), Complex2)

        # Declare charge and electric fields with fixed number of guard layers.
        # This is necessary for interfacing with PPIC2's C-routines.
        self.qe = np.zeros((grid.nyp+1, grid.nx+2), dtype=Float)
        self.fxye = np.zeros((grid.nyp+1, grid.nx+2), dtype=Float2)

        # Prepare fft tables
        cwpfft2rinit(self.mixup, self.sct, self.indx, self.indy)

        # Calculate form factors
        if custom_cppois22:
            kstrt = grid.comm.rank + 1
            calc_form_factors(self.qt, self.ffc,
                              ax, ay, self.affp, grid, kstrt)
        else:
            msg = "Using PPIC2's Poisson solver requires dx=dy"
            assert np.isclose(grid.dx, grid.dy), msg
            isign = 0
            cppois22(self.qt, self.fxyt, isign, self.ffc,
                     self.ax, self.ay, self.affp, grid)

    def __call__(self, rho, E):

        from ..cython.ppic2_wrapper import cppois22
        from ..cython.ppic2_wrapper import cwppfft2r, cwppfft2r2
        from ..cython.operators import grad_inv_del

        # Copy charge into pre-allocated buffer that has the right number of
        # guard layers expected by PPIC2
        self.qe[:-1, :-2] = rho.active

        # Transform charge to fourier space with standard procedure:
        # updates qt, modifies qe
        isign = -1
        ttp = cwppfft2r(self.qe, self.qt, self.bs, self.br, isign,
                        self.mixup, self.sct, self.indx, self.indy, self.grid)

        # Calculate force/charge in fourier space with standard procedure:
        # updates fxyt, we
        if self.custom_cppois22:
            kstrt = self.grid.comm.rank + 1
            we = grad_inv_del(
                    self.qt, self.fxyt, self.ffc, self.grid, kstrt)
        else:
            isign = -1
            we = cppois22(
                    self.qt, self.fxyt, isign, self.ffc,
                    self.ax, self.ay, self.affp, self.grid)
            # Scale with dx and dy (this is not done by ppic2s FFT)
            self.fxyt['x'] *= self.grid.dx
            self.fxyt['y'] *= self.grid.dy

        # Transform force to real space with standard procedure:
        # updates fxye, modifies fxyt
        isign = 1
        cwppfft2r2(self.fxye, self.fxyt, self.bs, self.br, isign,
                   self.mixup, self.sct, self.indx, self.indy, self.grid)

        # Copy electric field into an array with arbitrary number
        # of guard layers
        E['x'].active = self.fxye['x'][:-1, :-2]
        E['y'].active = self.fxye['y'][:-1, :-2]
        E['z'].active = 0.0

        E.boundaries_set = False

        return ttp, we
