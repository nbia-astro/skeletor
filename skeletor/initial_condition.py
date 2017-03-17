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

    def __init__(self, npc, ikx, iky, ampl, **kwds):

        # kwds['quiet'] = True
        super().__init__(npc, **kwds)

        # Wavenumber mode numbers
        self.ikx = ikx
        self.iky = iky

        # Amplitude of perturbation
        self.ampl = ampl

        if self.ikx == 0:
            msg = """This class unfortunately cannot currently handle density
            perturbations that do not have an x-dependence. The reason is
            that particle positions are assumed to be uniformly placed along x.
            The density perturbations are created by varying the interparticle
            distance in the y-direction only."""
            raise RuntimeError(msg)

    def __call__(self, manifold, ions):
        # TODO: Add more documentation

        from scipy.optimize import newton

        # This initialize uniformly distributed particle positions
        # and normally distributed particle velocities
        super().__call__(manifold, ions)

        self.Lx = manifold.Lx
        self.Ly = manifold.Ly

        self.kx = self.ikx*2*np.pi/self.Lx
        self.ky = self.iky*2*np.pi/self.Ly

        # x-coordinate in units of the box size
        x = ions['x'][:ions.N]/manifold.nx
        # y-coordinate in "physical" units
        y = ions['y'][:ions.N]*manifold.dy

        # Find cdf
        cdf = self.find_cdf()

        for ip in range(ions.N):
            # This guess is exact if the self.ampl is zero and it's presumably
            # still a decent guess if self.ampl is small compared to unity.
            guess = x[ip]*manifold.Lx
            x[ip] = newton(lambda X: cdf(X, y[ip]) - x[ip], guess)

        # Set initial positions
        ions['x'][:ions.N] = x/manifold.dx
        ions['y'][:ions.N] = y/manifold.dy

    def find_cdf(self, phase=0.0):
        """
        This function symbolically calculates the cdf for the density
        distribution.
        """
        import sympy

        # Define symbols
        x, y = sympy.symbols("x, y")

        # Density distribution
        n = 1 + self.ampl*sympy.cos(self.kx*x + self.ky*y + phase)

        # Analytic density distribution as numpy function
        self.f = sympy.lambdify((x, y), n, "numpy")

        # Symbolic pdf and cdf
        pdf_sym = n/sympy.integrate(n, (x, 0, self.Lx))
        cdf_sym = sympy.integrate(pdf_sym, (x, 0, x))

        # Turn sympy function into numpy function
        return sympy.lambdify((x, y), cdf_sym, "numpy")
