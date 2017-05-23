import numpy as np


class InitialCondition():

    def __init__(self, npc, quiet=False, vt=0.0, global_init=False):

        # Quiet start?
        self.quiet = quiet

        # Particles per cell
        self.npc = npc

        # Ion thermal velocity
        self.vt = vt

        # Initialize particles "globally" on each processor?
        self.global_init = global_init

    def __call__(self, manifold, ions):

        nx = manifold.nx
        ny = manifold.ny if self.global_init else manifold.nyp

        # Number of particles on all (one) processor(s)
        # if global_init is True (False)
        N = nx*ny*self.npc

        # Uniform distribution of particle positions
        if self.quiet:
            # Particles are placed on a regular grid (quiet start)
            sqrt_npc = int(np.sqrt(self.npc))
            assert sqrt_npc**2 == self.npc
            x1 = (np.arange(nx*sqrt_npc) + 0.5)/sqrt_npc
            y1 = (np.arange(ny*sqrt_npc) + 0.5)/sqrt_npc
            x, y = [xy.flatten() for xy in np.meshgrid(x1, y1)]
        else:
            # Particles are placed randomly (noisy start)
            x = nx*np.random.uniform(size=N)
            y = ny*np.random.uniform(size=N)

        # Draw particle velocities from a normal distribution
        # with zero mean and width 'vt'
        vx = self.vt*np.random.normal(size=N)
        vy = self.vt*np.random.normal(size=N)
        vz = self.vt*np.random.normal(size=N)

        if self.global_init:
            # Distributed particles across subdomains
            x = manifold.x0 + x*manifold.dx
            y = manifold.y0 + y*manifold.dy
            ions.initialize(x, y, vx, vy, vz)
        else:
            # Set initial position
            ions['x'][:N] = x
            ions['y'][:N] = y + manifold.edges[0]

            ions['vx'][:N] = vx
            ions['vy'][:N] = vy
            ions['vz'][:N] = vz

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
