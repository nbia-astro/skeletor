import numpy as np


class InitialCondition():

    def __init__(self, npc, quiet=False, vt=0.0):

        # Quiet start?
        self.quiet = quiet

        # Particles per cell
        self.npc = npc

        # Ion thermal velocity
        self.vt = vt

    def __call__(self, manifold, ions):

        # Total number particles in one MPI domain
        N = manifold.nx*manifold.nyp*self.npc

        if self.quiet:
            # Uniform distribution of particle positions (quiet start)
            sqrt_npc = int(np.sqrt(self.npc))
            assert sqrt_npc**2 == self.npc
            npx = manifold.nx*sqrt_npc
            npy = manifold.nyp*sqrt_npc
            x1 = (np.arange(npx) + 0.5)/sqrt_npc
            y1 = (np.arange(npy) + 0.5)/sqrt_npc + manifold.edges[0]
            x, y = [xy.flatten() for xy in np.meshgrid(x1, y1)]
        else:
            x = manifold.nx*np.random.uniform(size=N)
            y = manifold.nyp*np.random.uniform(size=N) + manifold.edges[0]

        # Set initial position
        ions['x'][:N] = x
        ions['y'][:N] = y

        # Draw particle velocities from a normal distribution
        # with zero mean and width 'vt'
        ions['vx'][:N] = self.vt*np.random.normal(size=N)
        ions['vy'][:N] = self.vt*np.random.normal(size=N)
        ions['vz'][:N] = self.vt*np.random.normal(size=N)

        ions.N = N


class DensityPertubation(InitialCondition):

    def __init__(self, npc, ikx, iky, ampl, vt=0):

        # Particles per cell
        self.npc = npc

        # Wavenumber mode numbers
        self.ikx = ikx
        self.iky = iky

        # Amplitude of perturbation
        self.ampl = ampl

        # Ion thermal velocity
        self.vt = vt

        if self.ikx == 0:
            msg = """This class unfortunately cannot currently handle density
            perturbations that do not have an x-dependence. The reason is
            that particle positions are assumed to be uniformly placed along x.
            The density perturbations are created by varying the interparticle
            distance in the y-direction only."""
            raise RuntimeError(msg)

    def __call__(self, manifold, ions):

        from scipy.optimize import newton

        self.Lx = manifold.Lx
        self.Ly = manifold.Ly

        self.kx = self.ikx*2*np.pi/self.Lx
        self.ky = self.iky*2*np.pi/self.Ly

        sqrt_npc = int(np.sqrt(self.npc))
        assert sqrt_npc**2 == self.npc
        npx = manifold.nx*sqrt_npc
        npy = manifold.nyp*sqrt_npc
        # Uniformly distributed numbers from 0 to 1
        U = (np.arange(npx) + 0.5)/npx
        # Particle y positions
        y1 = ((np.arange(npy) + 0.5)/sqrt_npc + manifold.edges[0])*manifold.dy

        self.X = np.empty_like(U)

        # Find cdf
        self.find_cdf()

        # Store newton solver for easy access
        self.newton = newton
        self.npx = npx
        self.npy = npy

        N = npx*npy
        x = np.empty(N)
        y = np.empty(N)

        # Calculate particle x-positions
        for k in range(0, self.npy):
            self.find_X(U, y1[k])
            x[k*npx:(k+1)*npx] = self.X
            y[k*npx:(k+1)*npx] = y1[k]

        # Set initial positions
        ions['x'][:N] = x/manifold.dx
        ions['y'][:N] = y/manifold.dy

        # Draw particle velocities from a normal distribution
        # with zero mean and width 'vt'
        ions['vx'][:N] = self.vt*np.random.normal(size=N)
        ions['vy'][:N] = self.vt*np.random.normal(size=N)
        ions['vz'][:N] = self.vt*np.random.normal(size=N)

        ions.N = N

    def find_cdf(self, phase=0.0):
        """
        This function symbolically calculates the cdf for the density
        distribution.
        """
        import sympy as sym

        # Define symbols
        x, y = sym.symbols("x, y")

        # Density distribution
        n = 1 + self.ampl*sym.cos(self.kx*x + self.ky*y + phase)

        # Analytic density distribution as numpy function
        self.f = sym.lambdify((x, y), n, "numpy")

        # Symbolic pdf and cdf
        pdf_sym = n/sym.integrate(n, (x, 0, self.Lx))
        cdf_sym = sym.integrate(pdf_sym, (x, 0, x))

        # Turn sympy function into numpy function
        self.cdf = sym.lambdify((x, y), cdf_sym, "numpy")

    def find_X(self, U, y):
        """
        Find a row of y-values for each value of x.
        """
        self.X[0] = self.newton(lambda x: self.cdf(x, y) - U[0], 0)
        for i in range(1, self.npx):
            self.X[i] = self.newton(lambda x: self.cdf(x, y) - U[i],
                                    self.X[i-1])
