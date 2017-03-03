from skeletor import Float, Grid, Particles, Sources
from skeletor.manifolds.second_order import Manifold

from mpi4py.MPI import COMM_WORLD as comm, SUM
import numpy

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
sources = Sources(manifold, npc)

# Total number of particles
np = npc*nx*ny
# Maximum number of particles in each partition
npmax = int(1.5*np/comm.size)

# Create particle array
particles = Particles(manifold, npmax, charge=charge, mass=mass)

# Synchronize random number generator across ALL processes
numpy.random.set_state(comm.bcast(numpy.random.get_state()))

# Uniform distribution of particle positions
x = nx*numpy.random.uniform(size=np).astype(Float)
y = ny*numpy.random.uniform(size=np).astype(Float)
# Normal distribution of particle velocities
vx = numpy.empty(np, Float)
vy = numpy.empty(np, Float)
vz = numpy.empty(np, Float)

# Assign particles to subdomains
particles.initialize(x, y, vx, vy, vz)


# def test_initialize():
#     """
#     Check that after the particles have been distributed across subdomains,
#     summing over all subdomains gives back the total number of particles.
#     """
#     # TODO: Figure out why this test sometimes fails.
#     assert comm.allreduce(particles.np, op=SUM) == np


def test_deposit():
    """
    The deposited charge summed over active *and* guard cells must be equal to
    the number of particles in each subdomain times the particle charge.
    """
    sources.deposit(particles)
    assert numpy.isclose(sources.rho.sum(), charge*particles.np/npc)


def test_add_guards():
    """
    After the charge deposited in the guard cells has been added to the
    corresponding active cells, summing the charge density over all active
    cells and all subdomains must yield the *total* number of particles times
    the particle charge.
    """
    sources.rho.add_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=SUM), np*charge/npc)
