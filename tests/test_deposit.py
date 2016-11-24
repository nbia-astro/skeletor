from skeletor import Float, Grid, Particles, Sources, cppinit

from mpi4py.MPI import COMM_WORLD as comm, SUM
import numpy

# Number of grid points in x- and y-direction (nx = 2**indx, ...)
nx, ny = 512, 512

# Average number of particles
npc = 32

# Particle charge and mass
charge = -1.0
mass = 1.0

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid
grid = Grid(nx, ny, comm)

# # Initialize sources
sources = Sources(grid, dtype=Float)
sources_ppic2 = Sources(grid, dtype=Float)

# Total number of particles
np = npc*nx*ny
# Maximum number of particles in each partition
npmax = int(1.5*np/nvp)

# Create particle array
particles = Particles(npmax, charge, mass)

# Synchronize random number generator across ALL processes
numpy.random.set_state(comm.bcast(numpy.random.get_state()))

# Uniform distribution of particle positions
x = nx*numpy.random.uniform(size=np).astype(Float)
y = ny*numpy.random.uniform(size=np).astype(Float)
# Normal distribution of particle velocities
vx = numpy.empty(np, Float)
vy = numpy.empty(np, Float)

# Assign particles to subdomains
particles.initialize(x, y, vx, vy, grid)


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
    assert numpy.isclose(sources.rho.sum(), charge*particles.np)


def test_deposit_ppic2():
    """
    Make sure that PPIC2's deposit routine gives the same charge density.
    """
    sources_ppic2.deposit_ppic2(particles)
    assert numpy.allclose(sources.rho, sources_ppic2.rho)


def test_add_guards():
    """
    After the charge deposited in the guard cells has been added to the
    corresponding active cells, summing the charge density over all active
    cells and all subdomains must yield the *total* number of particles times
    the particle charge.
    """
    sources.rho.add_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=SUM), np*charge)


def test_add_guards_ppic2():
    """
    Make sure that PPIC2's add_guards routine gives the same charge density.
    """
    sources_ppic2.rho.add_guards_ppic2()
    assert numpy.allclose(sources.rho.trim(), sources_ppic2.rho.trim())
