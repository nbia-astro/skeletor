from ppic2_wrapper import cppinit, cppexit
from dtypes import Float
from grid import Grid
from particles import Particles
import numpy
from mpi4py import MPI

def isclose_sorted (a, b, **kwargs):
    """This function first sorts the input arrays 'a' and 'b' (out of place)
    and returns true if the sorted arrays are almost equal to each other
    (element by element)."""
    from numpy import isclose, sort
    return all (isclose (sort (a), sort (b), **kwargs))

def sync_rng (comm=MPI.COMM_WORLD):
    """Synchronize random number generator across processes."""
    from numpy.random import get_state, set_state
    set_state (comm.bcast (get_state ()))

def mpi_allsum (A, comm=MPI.COMM_WORLD):
    """Sum over all processes. The result is available on all processes."""
    return comm.allreduce (A, op=MPI.SUM)

def mpi_allconcatenate (A, comm=MPI.COMM_WORLD):
    """Concatenate the array A on each process into one large continguous
    array. The result is available on all processes."""
    from numpy import concatenate
    return concatenate (comm.allgather (A))

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

# Start parallel processing. Calling this function necessary even though
# `MPI.Init()` has already been called by importing `MPI` from `mpi4py`. The
# reason is that `cppinit()` sets a number of global variables in the C library
# source file (`ppic2/pplib2.c`). The returned variables `idproc` and `nvp` are
# simply the MPI rank (i.e. processor id) and size (i.e. total number of
# processes), respectively.
idproc, nvp = cppinit ()

# Synchronize random number generator across processes
sync_rng ()

# Create numerical grid. This contains information about the extent of the
# subdomain assigned to each processor.
grid = Grid (nx, ny)

# Total number of particles in simulation
np = npc*grid.nx*grid.ny

# Maximum number of electrons in each partition
npmax = int (1.5*np/nvp)

# Uniform distribution of particle positions
x = grid.nx*numpy.random.uniform (size=np).astype (Float)
y = grid.nx*numpy.random.uniform (size=np).astype (Float)
# Normal distribution (possibly with shift) of particle positions
vx = vdx + vtx*numpy.random.normal (size=np).astype (Float)
vy = vdy + vty*numpy.random.normal (size=np).astype (Float)

# Assign particles to subdomains
ind = numpy.logical_and (y >= grid.edges[0], y < grid.edges[1])
electrons = Particles (x[ind], y[ind], vx[ind], vy[ind], npmax)
# Make sure the number of particles in each subdomain add up to the total
# number of particles
assert mpi_allsum (electrons.np) == np

# Sanity check.
# Combine particles from all processes into a single array and make sure that
# the result agrees with the global particle array
all_electrons = mpi_allconcatenate (electrons[:electrons.np])
assert isclose_sorted (x, all_electrons["x"])
assert isclose_sorted (y, all_electrons["y"])
assert isclose_sorted (vx, all_electrons["vx"])
assert isclose_sorted (vy, all_electrons["vy"])

for it in range (nt):

    # Push particles on each processor. This call also sends and receives
    # particles to and from other processors/subdomains. The latter is the only
    # non-trivial step in the entire code so far.
    electrons.push (grid, dt)

    # Push global particle array and apply boundary conditions.
    x += vx*dt
    y += vy*dt
    x %= nx
    y %= ny

# Combine particles from all processes into a single array and make sure that
# the result agrees with the global particle array
all_electrons = mpi_allconcatenate (electrons[:electrons.np])
assert isclose_sorted (x, all_electrons["x"])
assert isclose_sorted (y, all_electrons["y"])
assert isclose_sorted (vx, all_electrons["vx"])
assert isclose_sorted (vy, all_electrons["vy"])

# End parallel processing
cppexit ()
