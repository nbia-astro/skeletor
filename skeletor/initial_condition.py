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

        # Units are on
        ions.units = True


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
