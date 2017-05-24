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
        self.x0 = manifold.x0
        self.y0 = manifold.y0

        self.kx = self.ikx*2*np.pi/self.Lx
        self.ky = self.iky*2*np.pi/self.Ly

        # x-coordinate in units of the box size
        x = ions['x'][:ions.N]/manifold.nx
        # y-coordinate in "physical" units
        y = self.y0 + ions['y'][:ions.N]*manifold.dy

        # Find cdf
        cdf = self.find_cdf()

        for ip in range(ions.N):
            # This guess is exact if the self.ampl is zero and it's presumably
            # still a decent guess if self.ampl is small compared to unity.
            guess = self.x0 + x[ip]*manifold.Lx
            x[ip] = newton(lambda X: cdf(X, y[ip]) - x[ip], guess)

        # Set initial positions
        ions['x'][:ions.N] = (x - self.x0)/manifold.dx
        ions['y'][:ions.N] = (y - self.y0)/manifold.dy

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
        pdf_sym = n/sympy.integrate(n, (x, self.x0, self.x0 + self.Lx))
        cdf_sym = sympy.integrate(pdf_sym, (x, self.x0, x))

        # Turn sympy function into numpy function
        return sympy.lambdify((x, y), cdf_sym, "numpy")


class QuietMaxwellian:
    """
    Initialize a plasma with uniform density and a Maxwellian velocity
    distribution. The initialization is 'quiet' in phase-space.
    """
    def __init__(self, npc, vtx, vty, vtz, global_init=False, shuffle=True,
                 seed=1928346143):

        # Particles per cell
        self.npc = npc

        # Initialize particles "globally" on each processor?
        self.global_init = global_init

        # Thermal velocities
        self.vtx = vtx
        self.vty = vty
        self.vtz = vtz

        # Shuffle the velocities within each cell?
        self.shuffle = shuffle

        # Seed for generation of random numbers
        self.seed = seed

    def __call__(self, manifold, ions):
        from scipy.special import erfinv
        import numpy as np
        from skeletor.cython.misc import assemple_arrays

        nx = manifold.nx
        ny = manifold.ny if self.global_init else manifold.nyp

        # Number of particles on all (one) processor(s)
        # if global_init is True (False)
        N = nx*ny*self.npc

        # Uniform distribution of particle positions within a single cell
        # (quiet start)
        sqrt_npc = int(np.sqrt(self.npc))
        assert sqrt_npc**2 == self.npc
        a = (np.arange(sqrt_npc) + 0.5)/sqrt_npc
        x_cell, y_cell = np.meshgrid(a, a)
        x_cell = x_cell.flatten()
        y_cell = y_cell.flatten()

        # Quiet Maxwellian using the inverse error function
        R = (np.arange(self.npc) + 0.5)/self.npc
        vx_cell = erfinv(2*R - 1)*np.sqrt(2)*self.vtx
        vy_cell = erfinv(2*R - 1)*np.sqrt(2)*self.vty
        vz_cell = erfinv(2*R - 1)*np.sqrt(2)*self.vtz

        # Shuffle the velocities to remove some of introduced order
        # We use the same shuffling in every cell such that there is still a
        # high degree of artifical order in the system. In order to ensure
        # this, the seeds have to be the same on every processor.
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(vx_cell)
            np.random.seed(self.seed+1)
            np.random.shuffle(vy_cell)
            np.random.seed(self.seed+2)
            np.random.shuffle(vz_cell)

        # Initialize arrays with particle positions and velocities
        # The result has x and y in grid distance units and the velocities in
        # 'physical' units.
        # Cython is used to bring down the speed of a triple loop.
        x = np.empty(N)
        y = np.empty(N)
        vx = np.empty(N)
        vy = np.empty(N)
        vz = np.empty(N)
        assemple_arrays(x_cell, y_cell, vx_cell, vy_cell, vz_cell,
                        x, y, vx, vy, vz, self.npc, manifold)

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
