class Ohm:

    """Solve Ohm's law via a discrete Fourier transform."""

    def __init__(self, grid, npc, charge=1.0, temperature=0.0, eta=0.0):

        from numpy import zeros, zeros_like, array
        from math import log2
        from mpiFFT4py.line import R2C
        from mpi4py import MPI

        self.indx = int(log2(grid.nx))
        self.indy = int(log2(grid.ny))

        assert grid.nx == 2**self.indx, "'nx' needs to be a power of two"
        assert grid.ny == 2**self.indy, "'ny' needs to be a power of two"

        # Normalization constant
        self.affp = 1/npc

        # Grid dimensions
        self.N = array([grid.ny, grid.nx], dtype=int)
        self.L = array([grid.Ly, grid.Lx], dtype=float)

        # Create FFT object
        self.FFT = R2C(self.N, self.L, MPI, "double")

        # Pre-allocate array for Fourier transform and force
        self.lnrho_hat = zeros(
                self.FFT.complex_shape(), dtype=self.FFT.complex)
        self.Ex_hat = zeros_like(self.lnrho_hat)
        self.Ey_hat = zeros_like(self.lnrho_hat)

        # Scaled local wavevector
        k = self.FFT.get_scaled_local_wavenumbermesh()

        # Define kx and ky (notice swapping due to the grid ordering)
        self.kx = k[1]
        self.ky = k[0]

        # Charge
        self.charge = charge

        # Temperature
        self.temperature = temperature

        # Ratio of temperature to charge
        self.alpha = self.temperature/self.charge

        # Resistivity
        self.eta = eta

    def __call__(self, rho, E, destroy_input=True):
        from numpy import log

        # Transform log of charge density to Fourier space
        self.lnrho_hat[:] = self.FFT.fft2(log(rho.trim()), self.lnrho_hat)

        # Pressure term of Ohm's law
        # Notice that we multiply the charge to get the force
        self.Ex_hat[:] = -self.alpha*1j*self.kx*self.lnrho_hat
        self.Ey_hat[:] = -self.alpha*1j*self.ky*self.lnrho_hat

        # Transform back to real space
        E["x"][:-1, :-2] = self.FFT.ifft2(self.Ex_hat, E["x"][:-1, :-2])
        E["y"][:-1, :-2] = self.FFT.ifft2(self.Ey_hat, E["y"][:-1, :-2])

        # Add resitivity
        # TODO
