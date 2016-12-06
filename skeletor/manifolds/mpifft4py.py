from ..grid import Grid
import warnings


class Manifold(Grid):

    def __init__(self, nx, ny, comm, ax, ay):

        from math import log2
        from numpy import zeros, sum, where, zeros_like, array, exp
        from mpiFFT4py.line import R2C
        from mpi4py import MPI
        from skeletor import Float, Complex

        super().__init__(nx, ny, comm)

        self.indx = int(log2(nx))
        self.indy = int(log2(ny))

        assert nx == 2**self.indx, "'nx' needs to be a power of two"
        assert ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

        # Length vector
        L = array([self.Ly, self.Lx], dtype=Float)
        # Grid size vector
        N = array([self.ny, self.nx], dtype=int)

        # Create FFT object
        if str(Float) == 'float64':
            precision = 'double'
        else:
            precision = 'single'
        # precision = 'double' if str(Float) == 'float64' else 'single'
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

    def gradient(self, f, grad, destroy_input=None):
        """Calculate the gradient of f"""
        if destroy_input is not None:
            warnings.warn("Ignoring option 'destroy_input'.")

        self.FFT.fft2(f.trim(), self.f_hat)
        self.fx_hat[:] = 1j*self.kx*self.f_hat
        self.fy_hat[:] = 1j*self.ky*self.f_hat
        self.FFT.ifft2(self.fx_hat, grad['x'][:-1, :-2])
        self.FFT.ifft2(self.fy_hat, grad['y'][:-1, :-2])

    def grad_inv_del(self, f, grad_inv_del, destroy_input=None):
        """ """
        if destroy_input is not None:
            warnings.warn("Ignoring option 'destroy_input'.")

        self.FFT.fft2(f.trim(), self.f_hat)

        self.fx_hat[:] = -1j*self.kx*self.k21_eff*self.f_hat
        self.fy_hat[:] = -1j*self.ky*self.k21_eff*self.f_hat

        self.FFT.ifft2(self.fx_hat, grad_inv_del['x'][:-1, :-2])
        self.FFT.ifft2(self.fy_hat, grad_inv_del['y'][:-1, :-2])


class ShearingManifold(Manifold):

    def __init__(self, nx, ny, comm, ax, ay):

        from numpy.fft import rfftfreq
        from numpy import outer, pi, zeros
        from skeletor import Complex

        super().__init__(nx, ny, comm, ax, ay)

        # Grid spacing
        # TODO: this should be a property of the Grid class
        dx = self.Lx/self.nx
        dy = self.Ly/self.ny

        shape = self.ny//self.comm.size, self.nx//2+1
        self.temp = zeros(shape, dtype=Complex)

        # Wave numbers for real-to-complex transforms
        kx_vec = 2*pi*rfftfreq(self.nx)/dx

        # Outer product of y and kx
        self.y_kx = outer(self.y, kx_vec)

        # Maximum value of ky
        self.ky_max = pi/dy

        # Aspect ratio of grid
        self.aspect = self.Lx/self.Ly

    def _rfft2(self, f, f_hat, phase):
        from numpy import exp
        self.FFT.rfftx(f, self.temp)
        self.temp *= exp(-1j*phase)
        self.FFT.ffty(self.temp, f_hat)

    def _irfft2(self, f_hat, f, phase):
        from numpy import exp
        self.FFT.iffty(f_hat, self.temp)
        self.temp *= exp(1j*phase)
        self.FFT.irfftx(self.temp, f)

    def gradient(self, f, grad, St):
        """Gradient in the shearing sheet using mpifft4py"""

        from numpy import pi, mod

        # We only need to know how much time has elapsed since the last time
        # the domain was strictly periodic
        St %= self.aspect

        # Time dependent part of the laboratory frame 'ky'
        dky = St*self.kx

        # Phase shift to make 'psi' strictly periodic in 'y'.
        # This is an angle, so it can be mapped into the interval [0, 2*pi)
        phase = mod(self.y_kx*St, 2*pi)

        self._rfft2(f, self.f_hat, phase)

        # Laboratory frame 'ky'.
        # Exploit periodicity in Fourier space (i.e. aliasing) and make sure
        # that  -π/δy ≤ ky < π/δy
        ky = self.ky + dky
        ky = mod(ky + self.ky_max, 2*self.ky_max) - self.ky_max

        # Take derivative
        self.fx_hat[:] = 1j*self.kx*self.f_hat
        self.fy_hat[:] = 1j*ky*self.f_hat

        # Transform back to real space
        self._irfft2(self.fx_hat, grad['x'][:-1, :-2], phase)
        self._irfft2(self.fy_hat, grad['y'][:-1, :-2], phase)

    def grad_inv_del(self, f, grad_inv_del):
        raise 'grad_inv_del not implemented in shearing sheet'
