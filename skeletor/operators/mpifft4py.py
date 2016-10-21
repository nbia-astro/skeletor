class Operators:

    """Differential and translation operators using mpiFFT4py"""

    def __init__(self, grid, ax, ay, np):

        from math import log2
        from numpy import zeros, sum, where, zeros_like, array, exp
        from mpiFFT4py.line import R2C
        from mpi4py import MPI
        from skeletor import Float, Complex

        self.indx = int(log2(grid.nx))
        self.indy = int(log2(grid.ny))

        assert grid.nx == 2**self.indx, "'nx' needs to be a power of two"
        assert grid.ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

        # Length vector
        L = array([grid.Ly, grid.Lx], dtype=Float)
        # Grid size vector
        N = array([grid.ny, grid.nx], dtype=int)

        # Create FFT object
        if str(Float) == 'float64':
            precision = 'double'
        else:
            precision = 'single'
        self.FFT = R2C(N, L, MPI.COMM_WORLD, precision)

        # Pre-allocate array for Fourier transform and force
        self.f_hat = zeros(self.FFT.complex_shape(), dtype=Complex)
        self.fx_hat = zeros_like(self.f_hat)
        self.fy_hat = zeros_like(self.f_hat)

        # Scaled local wavevector
        k = self.FFT.get_scaled_local_wavenumbermesh()

        # Define kx and ky (notice that they are swapped due to the grid
        # ordering)
        self.kx = k[1]
        self.ky = k[0]

        # Local wavenumber squared
        k2 = sum(k*k, 0, dtype=Float)
        # Inverse of the wavenumber squared
        self.k21 = 1 / where(k2 == 0, 1, k2).astype(Float)
        # Effective inverse wave number for finite size particles
        self.k21_eff = self.k21*exp(-((self.kx*ax)**2 + (self.ky*ay)**2))

    def gradient(self, f, grad):
        """Calculate the gradient of f"""
        self.FFT.fft2(f.trim(), self.f_hat)
        self.fx_hat[:] = 1j*self.kx*self.f_hat
        self.fy_hat[:] = 1j*self.ky*self.f_hat
        self.FFT.ifft2(self.fx_hat, grad['x'][:-1, :-2])
        self.FFT.ifft2(self.fy_hat, grad['y'][:-1, :-2])

    def grad_inv_del(self, f, grad_inv_del):
        """ """
        self.FFT.fft2(f.trim(), self.f_hat)

        self.fx_hat[:] = -1j*self.kx*self.k21_eff*self.f_hat
        self.fy_hat[:] = -1j*self.ky*self.k21_eff*self.f_hat

        self.FFT.ifft2(self.fx_hat, grad_inv_del['x'][:-1, :-2])
        self.FFT.ifft2(self.fy_hat, grad_inv_del['y'][:-1, :-2])
