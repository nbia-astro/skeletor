from ppic2_wrapper import cppinit
from dtypes import Float, Float2
from grid import Grid
from fields import Field
from particles import Particles
from sources import Sources, GlobalSources
import numpy
from mpi4py import MPI


def allclose_sorted(a, b, **kwargs):
    """This function first sorts the input arrays 'a' and 'b' (out of place)
    and returns true if the sorted arrays are almost equal to each other
    (element by element)."""
    from numpy import allclose, sort
    return allclose(sort(a), sort(b), **kwargs)


def mpi_allsum(A, comm=MPI.COMM_WORLD):
    """Sum over all processes. The result is available on all processes."""
    return comm.allreduce(A, op=MPI.SUM)


def mpi_allconcatenate(A, comm=MPI.COMM_WORLD):
    """Concatenate the array A on each process into one large continguous
    array. The result is available on all processes."""
    from numpy import concatenate
    return concatenate(comm.allgather(A))


# Number of grid points in x- and y-direction
nx, ny = 32, 32

# Average number of particles per cell
npc = 256

# Thermal velocity of electrons in x- and y-direction
vtx, vty = 1.0, 1.0
# Drift velocity of electrons in x- and y-direction
vdx, vdy = 0.0, 0.0

# Timestep
dt = 0.1
# Number of timesteps to run for
nt = 10

# Synchronize random number generator across ALL processes
numpy.random.set_state(MPI.COMM_WORLD.bcast(numpy.random.get_state()))

# Start parallel processing. Calling this function necessary even though
# `MPI.Init()` has already been called by importing `MPI` from `mpi4py`. The
# reason is that `cppinit()` sets a number of global variables in the C library
# source file (`ppic2/pplib2.c`). The returned variables `idproc` and `nvp` are
# simply the MPI rank (i.e. processor id) and size (i.e. total number of
# processes), respectively.
idproc, nvp = cppinit()

# Create numerical grid. This contains information about the extent of the
# subdomain assigned to each processor.
grid = Grid(nx, ny)

# Total number of particles in simulation
np = npc*grid.nx*grid.ny

# Maximum number of electrons in each partition
npmax = int(1.5*np/nvp)

# Uniform distribution of particle positions
x = grid.nx*numpy.random.uniform(size=np).astype(Float)
y = grid.nx*numpy.random.uniform(size=np).astype(Float)
# Normal distribution of particle velocities
vx = vdx + vtx*numpy.random.normal(size=np).astype(Float)
vy = vdy + vty*numpy.random.normal(size=np).astype(Float)

# Assign particles to subdomains
ind = numpy.logical_and(y >= grid.edges[0], y < grid.edges[1])
electrons = Particles(x[ind], y[ind], vx[ind], vy[ind], npmax)
# Make sure the numbers of particles in each subdomain add up to the total
# number of particles
assert mpi_allsum(electrons.np) == np

# Sanity check.
# Combine particles from all processes into a single array and make sure that
# the result agrees with the global particle array
global_electrons = mpi_allconcatenate(electrons[:electrons.np])
assert allclose_sorted(x, global_electrons["x"])
assert allclose_sorted(y, global_electrons["y"])
assert allclose_sorted(vx, global_electrons["vx"])
assert allclose_sorted(vy, global_electrons["vy"])

# Set the force to zero (this will of course change in the future).
fxy = Field(grid, dtype=Float2)
fxy.fill(0.0)

for it in range(nt):

    # Push particles on each processor. This call also sends and receives
    # particles to and from other processors/subdomains. The latter is the only
    # non-trivial step in the entire code so far.
    electrons.push(fxy, dt)

    # Push global particle array and apply boundary conditions.
    x += vx*dt
    y += vy*dt
    x %= nx
    y %= ny

# Combine particles from all processes into a single array and make sure that
# the result agrees with the global particle array
global_electrons = mpi_allconcatenate(electrons[:electrons.np])
assert allclose_sorted(x, global_electrons["x"])
assert allclose_sorted(y, global_electrons["y"])
assert allclose_sorted(vx, global_electrons["vx"])
assert allclose_sorted(vy, global_electrons["vy"])

# Check charge deposition
global_sources = GlobalSources(grid, dtype=Float)
global_sources.deposit(global_electrons)
assert numpy.isclose(global_sources.rho.sum(), np)
global_sources.rho.add_guards()
assert numpy.isclose(global_sources.rho.trim().sum(), np)

sources = Sources(grid, dtype=Float)
sources.deposit(electrons)

sources2 = Sources(grid, dtype=Float)
sources2.deposit_ppic2(electrons)
assert numpy.allclose(sources.rho, sources2.rho)

assert numpy.isclose(sources.rho.sum(), electrons.np)

rho = sources.rho.copy()
rho2 = sources.rho.copy()

sources.rho.add_guards()
rho.add_guards2()
rho2.add_guards_ppic2()

assert numpy.isclose(mpi_allsum(rho2.trim().sum()), np)
assert numpy.isclose(mpi_allsum(rho.trim().sum()), np)
assert numpy.isclose(mpi_allsum(sources.rho.trim().sum()), np)

assert numpy.allclose(rho.trim(), sources.rho.trim())
assert numpy.allclose(rho2.trim(), sources.rho.trim())

sources.rho.copy_guards()
rho.copy_guards2()
rho2.copy_guards_ppic2()

assert numpy.allclose(rho, sources.rho)
assert numpy.allclose(rho2, sources.rho)

assert numpy.allclose(
    mpi_allconcatenate(sources.rho.trim()), global_sources.rho.trim())
