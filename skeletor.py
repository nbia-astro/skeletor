from ppic2_wrapper import cppinit
from dtypes import Float, Float2
from grid import Grid
from fields import Field
from particles import Particles
from sources import Sources
import numpy
from mpi4py import MPI


def allclose_sorted(a, b, **kwargs):
    """This function first sorts the input arrays 'a' and 'b' (out of place)
    and returns true if the sorted arrays are almost equal to each other
    (element by element)."""
    from numpy import allclose, sort
    return allclose(sort(a), sort(b), **kwargs)


def mpi_allsum(A):
    """Sum over all processes. The result is available on all processes."""
    return comm.allreduce(A, op=MPI.SUM)


def mpi_allconcatenate(A):
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

# Total number of particles in simulation
np = npc*nx*ny

# Uniform distribution of particle positions
x = nx*numpy.random.uniform(size=np).astype(Float)
y = nx*numpy.random.uniform(size=np).astype(Float)
# Normal distribution of particle velocities
vx = vdx + vtx*numpy.random.normal(size=np).astype(Float)
vy = vdy + vty*numpy.random.normal(size=np).astype(Float)

global_electrons = []
global_rho = []

for comm in [MPI.COMM_SELF, MPI.COMM_WORLD]:

    # Start parallel processing. Calling this function necessary even though
    # `MPI.Init()` has already been called by importing `MPI` from `mpi4py`.
    # The reason is that `cppinit()` sets a number of global variables in the C
    # library source file (`ppic2/pplib2.c`). The returned variables `idproc`
    # and `nvp` are simply the MPI rank (i.e. processor id) and size (i.e.
    # total number of processes), respectively.
    idproc, nvp = cppinit(comm)

    # Create numerical grid. This contains information about the extent of the
    # subdomain assigned to each processor.
    grid = Grid(nx, ny, comm)

    # Maximum number of electrons in each partition
    npmax = int(1.5*np/nvp)

    # Assign particles to subdomains
    ind = numpy.logical_and(y >= grid.edges[0], y < grid.edges[1])
    electrons = Particles(x[ind], y[ind], vx[ind], vy[ind], npmax, comm)
    # Make sure the numbers of particles in each subdomain add up to the total
    # number of particles
    assert mpi_allsum(electrons.np) == np

    # Set the force to zero (this will of course change in the future).
    fxy = Field(grid, comm, dtype=Float2)
    fxy.fill(0.0)

    for it in range(nt):

        # Push particles on each processor. This call also sends and receives
        # particles to and from other processors/subdomains. The latter is the
        # only non-trivial step in the entire code so far.
        electrons.push(fxy, dt)

    # Combine particles from all processes into a single array and make sure
    # that the result agrees with the global particle array
    global_electrons += [mpi_allconcatenate(electrons[:electrons.np])]

    sources = Sources(grid, comm, dtype=Float)
    sources2 = Sources(grid, comm, dtype=Float)

    sources.deposit(electrons)
    sources2.deposit_ppic2(electrons)

    assert numpy.allclose(sources.rho, sources2.rho)
    assert numpy.isclose(sources.rho.sum(), electrons.np)

    sources.rho.add_guards()
    sources2.rho.add_guards_ppic2()

    sources.rho.copy_guards()
    sources2.rho.copy_guards_ppic2()

    assert numpy.allclose(sources.rho, sources2.rho)
    assert numpy.isclose(mpi_allsum(sources.rho.trim().sum()), np)

    global_rho += [mpi_allconcatenate(sources.rho.trim())]

for component in ["x", "y", "vx", "vy"]:
    assert allclose_sorted(
        global_electrons[0][component],
        global_electrons[1][component])

assert numpy.allclose(global_rho[0], global_rho[1])
