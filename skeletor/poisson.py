class Poisson:

    """Solve Gauss' law ∇·E = ρ/ε0 via a discrete fourier transform."""

    def __init__(self, grid, ax, ay, np):

        from .cython.dtypes import Complex, Complex2, Int
        from .cython.ppic2_wrapper import cwpfft2rinit, cppois22
        from math import log2
        from numpy import zeros

        self.indx = int(log2(grid.nx))
        self.indy = int(log2(grid.ny))

        assert grid.nx == 2**self.indx, "'nx' needs to be a power of two"
        assert grid.ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

        # Normalization constant
        self.affp = grid.nx*grid.ny/np

        nxh = grid.nx//2
        nyh = (1 if 1 > grid.ny//2 else grid.ny//2)
        nxhy = (nxh if nxh > grid.ny else grid.ny)
        nxyh = (grid.nx if grid.nx > grid.ny else grid.ny)//2
        nye = grid.ny + 2
        kxp = (nxh - 1)//grid.nvp + 1
        kyp = (grid.ny - 1)//grid.nvp + 1

        self.qt = zeros((kxp, nye), Complex)
        self.fxyt = zeros((kxp, nye), Complex2)
        self.mixup = zeros(nxhy, Int)
        self.sct = zeros(nxyh, Complex)
        self.ffc = zeros((kxp, nyh), Complex)
        self.bs = zeros((kyp, kxp), Complex2)
        self.br = zeros((kyp, kxp), Complex2)

        # Prepare fft tables
        cwpfft2rinit(self.mixup, self.sct, self.indx, self.indy)

        # Calculate form factors
        isign = 0
        cppois22(
                self.qt, self.fxyt, isign, self.ffc,
                self.ax, self.ay, self.affp, grid)

    def __call__(self, qe, fxye, destroy_input=True):

        from .cython.ppic2_wrapper import cppois22, cwppfft2r, cwppfft2r2

        grid = qe.grid

        if destroy_input:
            qe_ = qe
        else:
            qe_ = qe.copy()

        # Transform charge to fourier space with standard procedure:
        # updates qt, modifies qe
        isign = -1
        ttp = cwppfft2r(
                qe_, self.qt, self.bs, self.br, isign, self.mixup, self.sct,
                self.indx, self.indy, grid)

        # Calculate force/charge in fourier space with standard procedure:
        # updates fxyt, we
        isign = -1
        we = cppois22(
                self.qt, self.fxyt, isign, self.ffc,
                self.ax, self.ay, self.affp, grid)

        # Transform force to real space with standard procedure:
        # updates fxye, modifies fxyt
        isign = 1
        cwppfft2r2(
                fxye, self.fxyt, self.bs, self.br, isign,
                self.mixup, self.sct, self.indx, self.indy, grid)

        return ttp, we

class PoissonMpiFFT4py:

    """Solve Gauss' law ∇·E = ρ/ε0 via a discrete fourier transform."""

    def __init__(self, grid, ax, ay, np):

        from math import log2
        from numpy import zeros, sum, where, zeros_like, array, exp
        from mpiFFT4py.line import R2C
        from mpi4py import MPI

        self.indx = int(log2(grid.nx))
        self.indy = int(log2(grid.ny))

        assert grid.nx == 2**self.indx, "'nx' needs to be a power of two"
        assert grid.ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

        # Normalization constant
        self.affp = grid.nx*grid.ny/np

        # Length vector
        self.L = array([grid.Ly, grid.Lx], dtype=float)
        # Grid size vector
        self.N = array([grid.ny, grid.nx], dtype=int)

        # Create FFT object
        self.FFT = R2C(self.N, self.L, MPI, "double")

        # Pre-allocate array for Fourier transform and force
        self.qe_hat = zeros(self.FFT.complex_shape(), dtype=self.FFT.complex)

        self.Ex_hat = zeros_like(self.qe_hat)
        self.Ey_hat = zeros_like(self.qe_hat)

        # Scaled local wavevector
        k = self.FFT.get_scaled_local_wavenumbermesh()

        # Define kx and ky (notice that they are swapped due to the grid ordering)
        self.kx = k[1]
        self.ky = k[0]

        # Local wavenumber squared
        k2 = sum(k*k, 0, dtype=float)
        # Inverse of the wavenumber squared
        self.k21 = 1 / where(k2 == 0, 1, k2).astype(float)
        # Effective inverse wave number for finite size particles
        self.k21_eff = self.k21*exp(-((self.kx*ax)**2 + (self.ky*ay)**2))


    def __call__(self, qe, E, destroy_input=True):

       # Transform charge density to Fourier space
       self.qe_hat = self.FFT.fft2(qe.trim(), self.qe_hat)

       # Solve Gauss' law in Fourier space and transform back to real space
       self.Ex_hat = -1j*self.kx*self.k21_eff*self.qe_hat
       self.Ey_hat = -1j*self.ky*self.k21_eff*self.qe_hat

       E['x'][:-1, :-2] = self.affp*self.FFT.ifft2(self.Ex_hat, E['x'][:-1, :-2])
       E['y'][:-1, :-2] = self.affp*self.FFT.ifft2(self.Ey_hat, E['y'][:-1, :-2])