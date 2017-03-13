class InitialCondition():

    def __init__(self, npc, quiet=False, vt=0.0):

        # Quiet start?
        self.quiet = quiet

        # Particles per cell
        self.npc = npc

        # Ion thermal velocity
        self.vt = vt

    def __call__(self, manifold, ions):

        from numpy import sqrt, arange, meshgrid
        from numpy.random import uniform, normal

        # Total number particles in one MPI domain
        np = manifold.nx*manifold.nyp*self.npc

        if self.quiet:
            # Uniform distribution of particle positions (quiet start)
            sqrt_npc = int(sqrt(self.npc))
            assert sqrt_npc**2 == self.npc
            npx = manifold.nx*sqrt_npc
            npy = manifold.nyp*sqrt_npc
            x1 = manifold.Lx*(arange(npx) + 0.5)/npx
            y1 = manifold.edges[0]*manifold.dy + \
                 manifold.Ly/manifold.comm.size*(arange(npy) + 0.5)/npy
            x, y =  meshgrid(x1, y1)
            x = x.flatten()
            y = y.flatten()
        else:
            x = manifold.Lx*uniform(size=np)
            y = manifold.edges[0]*manifold.dy + \
                manifold.Ly/manifold.comm.size*uniform(size=np)

        # Set initial position
        ions['x'][:np] = x/manifold.dx
        ions['y'][:np] = y/manifold.dy

        # Draw particle velocities from a normal distribution
        # with zero mean and width 'vt'
        ions['vx'][:np] = self.vt*normal (size=np)
        ions['vy'][:np] = self.vt*normal (size=np)
        ions['vz'][:np] = self.vt*normal (size=np)

        ions.np = np

        ions.units = True

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
        from numpy import pi, sqrt, arange, empty, empty_like
        from numpy.random import normal

        self.Lx = manifold.Lx
        self.Ly = manifold.Ly

        self.kx = self.ikx*2*pi/self.Lx
        self.ky = self.iky*2*pi/self.Ly

        sqrt_npc = int(sqrt(self.npc))
        assert sqrt_npc**2 == self.npc
        npx = manifold.nx*sqrt_npc
        npy = manifold.nyp*sqrt_npc
        Ux = (arange(npx) + 0.5)/npx
        Uy = manifold.edges[0]*manifold.dy + \
             manifold.Ly/manifold.comm.size*(arange(npy) + 0.5)/npy

        self.X = empty_like(Ux)

        # Find cdf
        self.find_cdf()

        # Store newton solver for easy access
        self.newton = newton
        self.npx = npx
        self.npy = npy

        np = npx*npy
        x = empty(np)
        y = empty(np)

        # Calculate particle positions
        for k in range (0, self.npy):
            self.find_X(Ux, Uy[k])
            x[k*npx:(k+1)*npx] = self.X
            y[k*npx:(k+1)*npx] = Uy[k]

        # Set initial positions
        ions['x'][:np] = x/manifold.dx
        ions['y'][:np] = y/manifold.dy

        # Draw particle velocities from a normal distribution
        # with zero mean and width 'vt'
        ions['vx'][:np] = self.vt*normal (size=np)
        ions['vy'][:np] = self.vt*normal (size=np)
        ions['vz'][:np] = self.vt*normal (size=np)

        ions.np = np

        ions.units = True

    def find_cdf(self, phase=0.0):
        """
        This function symbolically calculates the cdf for the density
        distribution.
        """
        import sympy as sym

        # Define symbols
        x, y = sym.symbols ("x, y")

        # Density distribution
        n = 1 + self.ampl*sym.cos(self.kx*x + self.ky*y + phase)

        # Analytic density distribution as numpy function
        self.f = sym.lambdify((x, y), n, "numpy")

        # Symbolic pdf and cdf
        pdf_sym = n/sym.integrate(n, (x, 0, self.Lx))
        cdf_sym = sym.integrate(pdf_sym, (x, 0, x))

        # Turn sympy function into numpy function
        self.cdf = sym.lambdify((x, y), cdf_sym, "numpy")

    def find_X(self, Ux, y):
        """
        Find a row of y-values for each value of x.
        """
        self.X[0] = self.newton(lambda x: self.cdf(x, y) - Ux[0], 0)
        for i in range (1, self.npx):
            self.X[i] = self.newton(lambda x: self.cdf(x, y) - Ux[i],
                                    self.X[i-1])
