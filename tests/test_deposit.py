from skeletor import Float, Particles, Sources
from skeletor.manifolds.second_order import Manifold

from mpi4py.MPI import COMM_WORLD as comm, SUM
import numpy as np

# Number of grid points in x- and y-direction (nx = 2**indx, ...)
nx, ny = 512, 512

# Average number of particles
npc = 32

# Particle charge and mass
charge = -1.0
mass = 1.0

# Create numerical grid
manifold = Manifold(nx, ny, comm)

# # Initialize sources
sources = Sources(manifold)

# Total number of particles
N = npc*nx*ny
# Maximum number of particles in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
particles = Particles(manifold, Nmax, charge=charge, mass=mass)

# Synchronize random number generator across ALL processes
np.random.set_state(comm.bcast(np.random.get_state()))

# Uniform distribution of particle positions
x = manifold.Lx*np.random.uniform(size=N).astype(Float)
y = manifold.Ly*np.random.uniform(size=N).astype(Float)
# Normal distribution of particle velocities
vx = np.empty(N, Float)
vy = np.empty(N, Float)
vz = np.empty(N, Float)

# Assign particles to subdomains
particles.initialize(x, y, vx, vy, vz)


# def test_initialize():
#     """
#     Check that after the particles have been distributed across subdomains,
#     summing over all subdomains gives back the total number of particles.
#     """
#     # TODO: Figure out why this test sometimes fails.
#     assert comm.allreduce(particles.N, op=SUM) == N


def test_deposit():
    """
    The deposited charge summed over active *and* guard cells must be equal to
    the number of particles in each subdomain times the particle charge.
    """
    sources.deposit(particles)
    assert np.isclose(sources.rho.sum(), charge*particles.N/npc)


def test_add_guards():
    """
    After the charge deposited in the guard cells has been added to the
    corresponding active cells, summing the charge density over all active
    cells and all subdomains must yield the *total* number of particles times
    the particle charge.
    """
    sources.add_guards()
    assert np.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=SUM), N*charge/npc)
