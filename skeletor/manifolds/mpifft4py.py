from ..grid import Grid


class Manifold(Grid):

    def __init__(
            self, nx, ny, comm,
            ax=0.0, ay=0.0, nlbx=0, nubx=2, nlby=0, nuby=1):

        from math import log2
        from numpy import zeros, sum, where, zeros_like, array, exp
        from mpiFFT4py.line import R2C
        from mpi4py import MPI
        from skeletor import Float, Complex

        super().__init__(
                nx, ny, comm, nlbx=nlbx, nubx=nubx, nlby=nlby, nuby=nuby)

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
        self.gx_hat = zeros_like(self.f_hat)
        self.gy_hat = zeros_like(self.f_hat)

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

    def gradient(self, f, g):
        """Calculate the gradient of f"""

        self.FFT.fft2(f.active, self.f_hat)
        self.gx_hat[:] = 1j*self.kx*self.f_hat
        self.gy_hat[:] = 1j*self.ky*self.f_hat
        self.FFT.ifft2(self.gx_hat, g.active['x'])
        self.FFT.ifft2(self.gy_hat, g.active['y'])

        g.boundaries_set = False

    def log(self, f):
        """Custom log function that works on the
            active cells of skeletor fields"""
        from numpy import log as numpy_log
        g = f.copy()
        g[f.grid.lby:f.grid.uby, f.grid.lbx:f.grid.ubx] = numpy_log(f.active)

        g.boundaries_set = False
        return g

    def grad_inv_del(self, f, g):

        self.FFT.fft2(f.active, self.f_hat)

        self.gx_hat[:] = -1j*self.kx*self.k21_eff*self.f_hat
        self.gy_hat[:] = -1j*self.ky*self.k21_eff*self.f_hat

        self.FFT.ifft2(self.gx_hat, g.active['x'])
        self.FFT.ifft2(self.gy_hat, g.active['y'])

        g.boundaries_set = False


class ShearingManifold(Manifold):

    def __init__(
            self, nx, ny, comm,
            ax=0.0, ay=0.0, nlbx=0, nubx=2, nlby=0, nuby=1, S=0, Omega=0):

        from numpy.fft import rfftfreq
        from numpy import outer, pi, zeros
        from skeletor import Complex

        super().__init__(
                nx, ny, comm, nlbx=nlbx, nubx=nubx, nlby=nlby, nuby=nuby)

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

        # Shear parameter
        self.S = S
        # True if shear is turned on
        self.shear = (S != 0)

        # Angular frequency
        self.Omega = Omega
        self.rotation = (Omega != 0)

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

    def gradient(self, f, g, St):
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
        self.gx_hat[:] = 1j*self.kx*self.f_hat
        self.gy_hat[:] = 1j*ky*self.f_hat

        # Transform back to real space
        self._irfft2(self.gx_hat, g.active['x'], phase)
        self._irfft2(self.gy_hat, g.active['y'], phase)

    def grad_inv_del(self, f, grad_inv_del):
        raise 'grad_inv_del not implemented in shearing sheet'
