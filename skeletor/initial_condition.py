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
        ions['x'][:np] = x
        ions['y'][:np] = y

        # Draw particle velocities from a normal distribution
        # with zero mean and width 'vt'
        ions['vx'][:np] = self.vt*normal (size=np)
        ions['vy'][:np] = self.vt*normal (size=np)

        ions.np = np

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
            msg = """This class unfortunately cannot handle density
            perturbations that do not have an x-dependence."""
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
        ions['x'][:np] = x
        ions['y'][:np] = y

        # Draw particle velocities from a normal distribution
        # with zero mean and width 'vt'
        ions['vx'][:np] = self.vt*normal (size=np)
        ions['vy'][:np] = self.vt*normal (size=np)

        ions.np = np

    def find_cdf(self, option=1):
        """
        This function symbolically calculates the cdf for the density
        distribution.
        """
        import sympy as sym

        # Define symbols
        x, y = sym.symbols ("x, y")

        # Density distribution
        # Cosine
        if option == 1:
            n = 1 + self.ampl*sym.cos(self.kx*x + self.ky*y)
        # Sine
        elif option == 2:
            n = 1 + self.ampl*sym.sin(self.kx*x + self.ky*y)

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


def uniform_density(nx, ny, npc, quiet):
    """Return Uniform distribution of particle positions"""

    if quiet:
        # Quiet start
        from numpy import sqrt, arange, meshgrid
        sqrt_npc = int(sqrt(npc))
        assert (sqrt_npc**2 == npc), 'npc need to be the square of an integer'
        dx = dy = 1/sqrt_npc
        x, y = meshgrid(arange(0, nx, dx), arange(0, ny, dy))
        x = x.flatten()
        y = y.flatten()
    else:
        # Random positions
        np = nx*ny*npc
        x = nx*numpy.random.uniform(size=np).astype(Float)
        y = ny*numpy.random.uniform(size=np).astype(Float)

    return (x, y)

def velocity_perturbation(x, y, kx, ky, ampl_vx, ampl_vy, vtx, vty):
    from numpy import random, sin
    from skeletor import Float

    # Perturbation to particle velocities
    vx = ampl_vx*sin(kx*x+ky*y)
    vy = ampl_vy*sin(kx*x+ky*y)

    # Number of particles
    np = x.shape[0]

    # Add thermal velocity
    vx += vtx*random.normal(size=np).astype(Float)
    vy += vty*random.normal(size=np).astype(Float)

    return (vx, vy)
