from dtypes import Float, Float2
from fields import Field
from grid import Grid
from particles import Particles
from particle_sort import ParticleSort
from poisson import Poisson
from ppic2_wrapper import cppinit
from sources import Sources

from mpi4py.MPI import COMM_WORLD as comm, SUM
import numpy

# Number of grid points in x- and y-direction (nx = 2**indx, ...)
nx, ny = 1 << 9, 1 << 9
# nx, ny = 1 << 5, 1 << 5

# Total number of electrons
np = 3072*3072
# np = 256*256

# Time at end of simulation, in units of plasma frequency
tend = 10.0
# Time interval between successive calculations
dt = 0.1

# Sort every 'nt_sort' timesteps
nt_sort = 50

# Particle charge and mass
charge = -1.0
mass = 1.0

# Thermal velocity of electrons
vtx, vty = 1.0, 1.0
# Drift velocity of electrons
vdx, vdy = 0.0, 0.0

# Smoothed particle size
ax, ay = 0.912871, 0.912871

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid
grid = Grid(nx, ny, comm)

# Initialize Poisson solver
poisson = Poisson(grid, comm, ax, ay, np)

# Initialize particle sort
sort = ParticleSort(grid)

# Initialize sources
sources = Sources(grid, comm, dtype=Float)

# Intialize force field
fxy = Field(grid, comm, dtype=Float2)

# Maximum number of electrons in each partition
npmax = int(1.5*np/nvp)

# Create particle array
electrons = Particles(npmax, charge, mass)
# Create another particle array. This is needed for out-of-place sorting.
electrons2 = Particles(npmax, charge, mass)

# Synchronize random number generator across ALL processes
numpy.random.set_state(comm.bcast(numpy.random.get_state()))

# Uniform distribution of particle positions
x = nx*numpy.random.uniform(size=np).astype(Float)
y = nx*numpy.random.uniform(size=np).astype(Float)
# Normal distribution of particle velocities
vx = vdx + vtx*numpy.random.normal(size=np).astype(Float)
vy = vdy + vty*numpy.random.normal(size=np).astype(Float)

# Assign particles to subdomains
ind = numpy.logical_and(y >= grid.edges[0], y < grid.edges[1])
electrons.initialize(x[ind], y[ind], vx[ind], vy[ind])
assert comm.allreduce(electrons.np, op=SUM) == np

# Total number of time steps
nt = int(tend/dt + 1e-4)

# Start integration
for it in range(nt):

    # Deposit charge
    sources.deposit(electrons)
    # Add charge from guard cells
    sources.rho.add_guards_ppic2()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=SUM), np*charge)

    # Solve Gauss' law
    poisson(sources.rho, fxy)
    # Apply boundary condition
    fxy.copy_guards_ppic2()

    # Push particles
    electrons.push(fxy, dt)

    # Sort particles
    if nt_sort > 0:
        if it % nt_sort == 0:
            sort(electrons, electrons2)
            # Exchange "pointers"
            tmp = electrons
            electrons = electrons2
            electrons2 = tmp
